///////////////////////////////////////////////////////////////////////////////
//
// File: Operator.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>

#include <LibUtilities/BasicUtils/MiscUtils.hpp>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/Communication/Comm.h>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <MultiRegions/ExpList.h>

#include "Operators/Common/Spaces.hpp"
#include "Operators/Field/Field.hpp"
#include <Operators/Common/OperatorsDeclspec.hpp>

namespace Nektar::LibUtilities
{
/**
 * Partial specialisation for memory region
 */
template <class elemT> class CommDataTypeTraits<MemoryRegion<elemT>>
{
public:
    static CommDataType &GetDataType()
    {
        return CommDataTypeTraits<elemT>::GetDataType();
    }
    static void *GetPointer(MemoryRegion<elemT> &val)
    {
        return val.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
    }
    static const void *GetPointer(const MemoryRegion<elemT> &val)
    {
        return val.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
    }
    static size_t GetCount(const MemoryRegion<elemT> &val)
    {
        return val.size();
    }
    const static bool IsVector = true;
};

} // namespace Nektar::LibUtilities

namespace Nektar::Operators
{

extern OPERATORS_EXPORT std::string g_OpExecSpace;
extern OPERATORS_EXPORT std::string g_OpImpl;

template <bool B, class T> struct simd_type_if
{
    typedef tinysimd::scalarT<T> type;
};

template <class T> struct simd_type_if<true, T>
{
    typedef tinysimd::simd<T> type;
};

// Core implementation types
class StdMat
{
};

class SumFac
{
};

class SumFacQP
{
};

// Use typenames to define available implementations to
// allow extension by users without modifying library
using default_fp_type = double;

// Forward-declare the Operator base class so we can define the factory
template <typename TData> class Operator;

// Typename alias for the factory
template <typename TData>
using OperatorFactory =
    Nektar::LibUtilities::NekFactory<std::string, Operator<TData>,
                                     const MultiRegions::ExpListSharedPtr &>;

// Operator factory singleton
template <typename TData> OperatorFactory<TData> &GetOperatorFactory();

template <typename TData> class Operator
{
public:
    virtual ~Operator() = default;

    Operator(const MultiRegions::ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList)
    {
        std::shared_ptr<LibUtilities::SessionReader> session =
            m_expansionList->GetSession();

        // Command-line specified execution space
        if (session->DefinesCmdLineArgument("opExecSpace"))
        {
            std::string cmdValue =
                session->GetCmdLineArgument<std::string>("opExecSpace");

            if (cmdValue == "AVX")
            {
                g_OpExecSpace = "AVX";
            }
            else if (cmdValue == "CUDA")
            {
                g_OpExecSpace = "CUDA";
            }
            else if (cmdValue == "HIP")
            {
                g_OpExecSpace = "HIP";
            }
            else if (cmdValue == "Kokkos")
            {
                g_OpExecSpace = "Kokkos";
            }
            else if (cmdValue == "SYCL")
            {
                g_OpExecSpace = "SYCL";
            }
            else if (cmdValue == "Serial")
            {
                g_OpExecSpace = "Serial";
            }
            else
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "Bad command line argument for opExecSpace:" +
                             cmdValue);
            }
        }

        // Command-line specified implementation
        if (session->DefinesCmdLineArgument("opImpl"))
        {
            std::string cmdValue =
                session->GetCmdLineArgument<std::string>("opImpl");

            if (cmdValue == "SumFac")
            {
                g_OpImpl = "SumFac";
            }
            else if (cmdValue == "SumFacQP")
            {
                g_OpImpl = "SumFacQP";
            }
            else if (cmdValue == "StdMat")
            {
                g_OpImpl = "StdMat";
            }
            else
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "Bad command line argument for opImpl:" + cmdValue);
            }
        }
    }

    template <typename TDescriptor, typename ExecSpace, typename Implementation>
    static std::shared_ptr<typename TDescriptor::class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        // The TDescriptor name contains the namespace as well as the <TData>
        // of <FieldState, TData> which needs to be removed.
        std::string descript = Nektar::demangleTypeName(typeid(TDescriptor));
        std::string descriptStr(descript);

        // Check for a Nektar::Operators:: root.
        if (Nektar::stripString(descriptStr, "Nektar::Operators::"))
        {
            size_t found = descriptStr.find("<");
            if (found != std::string::npos)
            {
                descriptStr.erase(found);
            }
            else
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "malformed operator template descriptor.");
            }
        }
        else
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "malformed operator template descriptor.");
        }

#if defined(_MSC_VER)
        // Check for a struct root.
        Nektar::stripString(descriptStr, "struct ");
#endif

        // The TDescriptor name may contain the FieldState, if so get
        // the enum and corresponding string.
        std::string fieldStateStr(descript);

        size_t found0 = fieldStateStr.find("0");
        size_t found1 = fieldStateStr.find("1");
        if (found0 != std::string::npos)
        {
            fieldStateStr = FieldStateString(FieldState(0));
        }
        else if (found1 != std::string::npos)
        {
            fieldStateStr = FieldStateString(FieldState(1));
        }
        else
        {
            fieldStateStr.clear();
        }

        // The ExecSpace name contains the namespace which needs to be
        // removed.
        std::string execStr = Nektar::demangleTypeName(typeid(ExecSpace));

        Nektar::stripString(execStr, "NektarSpaces::");

        if (execStr == "KOKKOS")
        {
            execStr = "Kokkos";
        }

#if defined(_MSC_VER)
        // Check for a class root.
        Nektar::stripString(execStr, "class ");
#endif

        // Overide with command-line specified execution space, if necessary
        if (g_OpExecSpace != "")
        {
            execStr = g_OpExecSpace;
        }

        // The Implementation name contains the namespace which needs
        // to be removed.
        std::string implStr = Nektar::demangleTypeName(typeid(Implementation));

        Nektar::stripString(implStr, "Nektar::Operators::");

#if defined(_MSC_VER)
        // Check for a class root.
        Nektar::stripString(implStr, "class ");
#endif

        // Overide with command-line specified implementation, if necessary
        if (g_OpImpl != "")
        {
            implStr = g_OpImpl;
        }

        std::string requestedKey =
            descriptStr + fieldStateStr + execStr + implStr;
        std::string key = requestedKey;

        OperatorFactory<TData> &factory = GetOperatorFactory<TData>();

        bool notFound = true;

#if !defined(OPERATOR_ENABLE_DEFAULTING)
        constexpr size_t nOpTests = 1;
#else
        constexpr size_t nOpTests = 3;
#endif
        for (size_t i = 0; i < nOpTests; ++i)
        {
            switch (i)
            {
                case 0:
                    // Find the operator with the requested ExecSpace and the
                    // same implementation.
                    key = descriptStr + fieldStateStr + execStr + implStr;
                    break;
                case 1:
                    // Find the operator with the "Serial" ExecSpace and the
                    // same implementation.
                    // key = descriptStr + fieldStateStr + "Serial" + implStr;
                    break;
                case 2:
                    // Find the operator with the "Serial" ExecSpace and
                    // the "StdMat" implementation.
                    key = descriptStr + fieldStateStr + "Serial" + "StdMat";
                    break;
                default:
                    break;
            }

            if (factory.ModuleExists(key))
            {
                if (key != requestedKey)
                {
                    std::string msg;
                    msg += "The requested operator: " + requestedKey +
                           " was not found. Using operator: " + key +
                           " instead";

                    WARNINGL0(false, msg);
                }

                notFound = false;

                break;
            }
        }

        // No suitible operator was found.
        if (notFound)
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey
                << " and no default operator: " << key << ". Descriptor is "
                << descript << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<typename TDescriptor::class_name>(
            factory.CreateInstance(key, expansionList));
    }

protected:
    size_t GetGeometricFactorSize(const std::vector<BlockAttributes> &blocks,
                                  const size_t width)
    {
        size_t gfSize = 0;
        size_t exp_id = 0;

        for (size_t blk = 0; blk < blocks.size(); ++blk)
        {
            size_t num_elmt_groups = blocks[blk].GetNumElmtGroups(width);

            const auto expPtr = this->m_expansionList->GetExp(exp_id);

            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                gfSize += num_elmt_groups * expPtr->GetTotPoints();
            }
            else
            {
                gfSize += num_elmt_groups;
            }

            exp_id += blocks[blk].num_elements;
        }

        return gfSize * width;
    }

    std::shared_ptr<std::vector<TData>> SetJacobian(
        size_t jacSize, std::vector<BlockAttributes> &blocks, size_t width)
    {
        // Allocate memory for the jacobian
        std::vector<TData> jac;
        jac.resize(jacSize);

        size_t exp_id = 0;
        size_t jac_id = 0;

        for (size_t blk = 0; blk < blocks.size(); ++blk)
        {
            size_t num_elements    = blocks[blk].num_elements;
            size_t num_elmt_groups = blocks[blk].GetNumElmtGroups(width);

            auto expPtr = this->m_expansionList->GetExp(exp_id);

            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                Array<OneD, Array<OneD, NekDouble>> jacArray(width);

                for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
                {
                    for (size_t i = 0; i < width; ++i, ++el)
                    {
                        if (el < num_elements)
                        {
                            jacArray[i] =
                                this->m_expansionList->GetExp(exp_id++)
                                    ->GetMetricInfo()
                                    ->GetJac(expPtr->GetPointsKeys());
                        }
                        else
                        {
                            jacArray[i] = Array<OneD, NekDouble>(
                                expPtr->GetTotPoints(), 0.0);
                        }
                    }

                    for (size_t pt = 0; pt < expPtr->GetTotPoints(); ++pt)
                    {
                        for (size_t i = 0; i < width; ++i, ++jac_id)
                        {
                            jac[jac_id] = jacArray[i][pt];
                        }
                    }
                }
            }
            else // regular geometry
            {
                for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
                {
                    for (size_t i = 0; i < width; ++i, ++el, ++jac_id)
                    {
                        if (el < num_elements)
                        {
                            auto &auxJac =
                                this->m_expansionList->GetExp(exp_id++)
                                    ->GetMetricInfo()
                                    ->GetJac(expPtr->GetPointsKeys());
                            jac[jac_id] = auxJac[0];
                        }
                        else
                        {
                            jac[jac_id] = 0.0;
                        }
                    }
                }
            }
        }

        return MemoryManager<std::vector<TData>>::AllocateSharedPtr(jac);
    }

    std::shared_ptr<std::vector<TData>> SetDerivativeFactor(
        size_t dfSize, std::vector<BlockAttributes> &blocks, size_t width,
        bool transpose = false)
    {
        // Allocate memory for the derivative factor
        size_t nDim   = this->m_expansionList->GetShapeDimension();
        size_t nCoord = this->m_expansionList->GetCoordim(0);

        std::vector<TData> derivFac;
        derivFac.resize(nDim * nCoord * dfSize);

        size_t exp_id = 0;
        size_t df_id  = 0;

        for (size_t blk = 0; blk < blocks.size(); ++blk)
        {
            size_t num_elements    = blocks[blk].num_elements;
            size_t num_elmt_groups = blocks[blk].GetNumElmtGroups(width);
            auto expPtr            = this->m_expansionList->GetExp(exp_id);

            size_t range1 = transpose ? nDim * nCoord : expPtr->GetTotPoints();
            size_t range2 = transpose ? expPtr->GetTotPoints() : nDim * nCoord;

            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
                {
                    for (size_t index1 = 0; index1 < range1; ++index1)
                    {
                        for (size_t index2 = 0; index2 < range2; ++index2)
                        {
                            for (size_t i = 0; i < width; ++i, ++df_id)
                            {
                                if (el + i < num_elements)
                                {
                                    size_t d  = transpose ? index1 : index2;
                                    size_t pt = transpose ? index2 : index1;

                                    auto &df = this->m_expansionList
                                                   ->GetExp(exp_id + i)
                                                   ->GetMetricInfo()
                                                   ->GetDerivFactors(
                                                       expPtr->GetPointsKeys());
                                    derivFac[df_id] = df[d][pt];
                                }
                                else
                                {
                                    derivFac[df_id] = 0.0;
                                }
                            }
                        }
                    }

                    if (el < num_elements)
                    {
                        exp_id += std::min(width, num_elements - el);
                    }
                    el += width;
                }
            }
            else
            {
                for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
                {
                    for (size_t d = 0; d < nDim * nCoord; ++d)
                    {
                        for (size_t i = 0; i < width; ++i, ++df_id)
                        {
                            if (el + i < num_elements)
                            {
                                auto &df =
                                    this->m_expansionList->GetExp(exp_id + i)
                                        ->GetMetricInfo()
                                        ->GetDerivFactors(
                                            expPtr->GetPointsKeys());
                                derivFac[df_id] = df[d][0];
                            }
                            else
                            {
                                derivFac[df_id] = 0.0;
                            }
                        }
                    }

                    if (el < num_elements)
                    {
                        exp_id += std::min(width, num_elements - el);
                    }
                    el += width;
                }
            }
        }

        return MemoryManager<std::vector<NekDouble>>::AllocateSharedPtr(
            derivFac);
    }

    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
