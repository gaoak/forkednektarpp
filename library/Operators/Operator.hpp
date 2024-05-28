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
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <MultiRegions/ExpList.h>
#include <Operators/OperatorsDeclspec.hpp>

#include "Operators/Field.hpp"
#include "Operators/Spaces.hpp"

#define FLAG_QP false

namespace Nektar::Operators
{

extern OPERATORS_EXPORT std::string g_ExecSpace;
extern OPERATORS_EXPORT std::string g_Impl;

// Core implementation types
class StdMat
{
};

class SumFac
{
};

// Use typenames to define available implementations to
// allow extension by users without modifying library
using default_fp_type = double;
using vec_t           = tinysimd::simd<double>;

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

        if (!Nektar::stripString(execStr, "NektarSpaces::"))
        {
            // Check for a Kokkos:: root.
            size_t found = execStr.find("Kokkos::");
            if (found != std::string::npos)
            {
                execStr = "Kokkos";
            }
        }

#if defined(_MSC_VER)
        // Check for a class root.
        Nektar::stripString(execStr, "class ");
#endif

        // The Implementation name contains the namespace which needs
        // to be removed.
        std::string implStr = Nektar::demangleTypeName(typeid(Implementation));

        Nektar::stripString(implStr, "Nektar::Operators::");

#if defined(_MSC_VER)
        // Check for a class root.
        Nektar::stripString(implStr, "class ");
#endif

        std::string requestedKey =
            descriptStr + fieldStateStr + execStr + implStr;
        std::string key = requestedKey;

        OperatorFactory<TData> &factory = GetOperatorFactory<TData>();

        bool notFound = true;

        for (size_t i = 0; i < 7; ++i)
        {
            switch (i)
            {
                case 0:
                    // Find the operator with the same ExecSpace and the
                    // same implementation.
                    key = descriptStr + fieldStateStr + execStr + implStr;
                    break;
                case 1:
                    // Find the operator with the same ExecSpace and the default
                    // implementation.
                    key = descriptStr + fieldStateStr + execStr + g_Impl;
                    break;
                case 2:
                    // Find the operator with the same ExecSpace and the
                    // universal implementation.
                    key = descriptStr + fieldStateStr + execStr + "Universal";
                    break;
                case 3:
                    // Find the operator with the default ExecSpace and the
                    // default implementation.
                    key = descriptStr + fieldStateStr + g_ExecSpace + g_Impl;
                    break;
                case 4:
                    // Find the operator with the default ExecSpace and the
                    // universal implementation.
                    key =
                        descriptStr + fieldStateStr + g_ExecSpace + "Universal";
                    break;
                case 5:
                    // Find the operator with the "Serial" ExecSpace
                    // and the universal implementation.
                    key = descriptStr + fieldStateStr + "Serial" + "Universal";
                    break;
                case 6:
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
    // Standard Get/Set methods
    size_t GetGeometricFactorSize(void)
    {
        size_t gfSize    = 0;
        size_t nTotElmts = this->m_expansionList->GetNumElmts();

        // Calculate the jacobian array size
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            // Determine shape and type of the element
            auto const expPtr = this->m_expansionList->GetExp(e);
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                gfSize += expPtr->GetTotPoints();
            }
            else
            {
                gfSize++;
            }
        }

        return gfSize;
    }

    Array<OneD, TData> SetJacobian(size_t jacSize)
    {
        // Allocate memory for the jacobian
        Array<OneD, TData> jac(jacSize, 0.0);

        // Initialise jacobian.
        size_t index     = 0;
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            auto expPtr = this->m_expansionList->GetExp(e);
            auto &auxJac =
                expPtr->GetMetricInfo()->GetJac(expPtr->GetPointsKeys());
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                size_t nqe = expPtr->GetTotPoints();
                for (size_t i = 0; i < nqe; ++i)
                {
                    jac[index++] = auxJac[i];
                }
            }
            else
            {
                jac[index++] = auxJac[0];
            }
        }

        return jac;
    }

    Array<OneD, Array<OneD, TData>> SetDerivativeFactor(size_t dfSize)
    {
        // Allocate memory for the derivative factor
        size_t nDim   = this->m_expansionList->GetShapeDimension();
        size_t nCoord = this->m_expansionList->GetCoordim(0);
        Array<OneD, Array<OneD, TData>> derivFac(nDim * nCoord);
        for (size_t d = 0; d < nDim * nCoord; d++)
        {
            derivFac[d] = Array<OneD, TData>(dfSize);
        }

        // Initialise derivative factor.
        size_t dfindex   = 0;
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            auto expPtr = this->m_expansionList->GetExp(e);
            auto &df    = expPtr->GetMetricInfo()->GetDerivFactors(
                expPtr->GetPointsKeys());
            size_t nqTot = expPtr->GetTotPoints();
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                for (size_t d = 0; d < nDim * nCoord; d++)
                {
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        derivFac[d][dfindex + i] = df[d][i];
                    }
                }
                dfindex += nqTot;
            }
            else
            {
                for (size_t d = 0; d < nDim * nCoord; d++)
                {
                    derivFac[d][dfindex] = df[d][0];
                }
                dfindex += 1;
            }
        }

        return derivFac;
    }

    // Vectorized Get/Set methods
    size_t GetGeometricFactorSize(const std::vector<BlockAttributes> &blocks)
    {
        size_t gfSize = 0;
        size_t exp_id = 0;

        for (size_t blk = 0; blk < blocks.size(); ++blk)
        {
            size_t num_metaBlocks =
                (blocks[blk].num_elements + blocks[blk].num_padding_elements) /
                vec_t::width;

            auto const expPtr = this->m_expansionList->GetExp(exp_id);

            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                gfSize += num_metaBlocks * expPtr->GetTotPoints();
            }
            else
            {
                gfSize += num_metaBlocks;
            }

            exp_id += blocks[blk].num_elements;
        }

        return gfSize;
    }

    std::shared_ptr<std::vector<vec_t, tinysimd::allocator<vec_t>>> SetJacobian(
        size_t jacSize, std::vector<BlockAttributes> &blocks)
    {
        // Allocate memory for the jacobian
        std::vector<vec_t, tinysimd::allocator<vec_t>> jac;
        jac.resize(jacSize);

        size_t exp_id = 0;
        size_t jac_id = 0;

        for (size_t blk = 0; blk < blocks.size(); ++blk)
        {
            size_t num_elements         = blocks[blk].num_elements;
            size_t num_padding_elements = blocks[blk].num_padding_elements;
            size_t num_metaBlocks =
                (num_elements + num_padding_elements) / vec_t::width;

            auto expPtr = this->m_expansionList->GetExp(exp_id);

            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                Array<OneD, Array<OneD, NekDouble>> jacArray(vec_t::width);
                alignas(vec_t::alignment) NekDouble tmp[vec_t::width];
                for (size_t e = 0; e < num_metaBlocks - 1; ++e)
                {
                    for (size_t i = 0; i < vec_t::width; ++i)
                    {
                        jacArray[i] = this->m_expansionList->GetExp(exp_id++)
                                          ->GetMetricInfo()
                                          ->GetJac(expPtr->GetPointsKeys());
                    }

                    for (size_t pt = 0; pt < expPtr->GetTotPoints(); ++pt)
                    {
                        for (size_t i = 0; i < vec_t::width; ++i)
                        {
                            tmp[i] = jacArray[i][pt];
                        }
                        jac[jac_id++].load(&tmp[0]);
                    }
                }

                // Last block: may have padding elements
                for (size_t i = 0; i < vec_t::width - num_padding_elements; ++i)
                {
                    jacArray[i] = this->m_expansionList->GetExp(exp_id++)
                                      ->GetMetricInfo()
                                      ->GetJac(expPtr->GetPointsKeys());
                }

                for (size_t pt = 0; pt < expPtr->GetTotPoints(); ++pt)
                {
                    for (size_t i = 0; i < vec_t::width - num_padding_elements;
                         ++i)
                    {
                        tmp[i] = jacArray[i][pt];
                    }
                    for (size_t i = vec_t::width - num_padding_elements;
                         i < vec_t::width; ++i)
                    {
                        tmp[i] = 0.0;
                    }
                    jac[jac_id++].load(&tmp[0]);
                }
            }
            else // regular geometry
            {
                alignas(vec_t::alignment) NekDouble tmp[vec_t::width];
                for (size_t e = 0; e < num_metaBlocks - 1; ++e)
                {
                    for (size_t i = 0; i < vec_t::width; ++i)
                    {
                        auto &auxJac = this->m_expansionList->GetExp(exp_id++)
                                           ->GetMetricInfo()
                                           ->GetJac(expPtr->GetPointsKeys());
                        tmp[i] = auxJac[0];
                    }
                    jac[jac_id++].load(&tmp[0]);
                }
                // last block: may have padding elements
                for (size_t i = 0; i < vec_t::width - num_padding_elements; ++i)
                {
                    auto &auxJac = this->m_expansionList->GetExp(exp_id++)
                                       ->GetMetricInfo()
                                       ->GetJac(expPtr->GetPointsKeys());
                    tmp[i] = auxJac[0];
                }
                for (size_t i = vec_t::width - num_padding_elements;
                     i < vec_t::width; ++i)
                {
                    tmp[i] = 0.0;
                }
                jac[jac_id++].load(&tmp[0]);
            }
        }

        return MemoryManager<std::vector<vec_t, tinysimd::allocator<vec_t>>>::
            AllocateSharedPtr(jac);
    }

    std::shared_ptr<std::vector<vec_t, tinysimd::allocator<vec_t>>>
    SetDerivativeFactor(size_t dfSize, std::vector<BlockAttributes> &blocks)
    {
        // Allocate memory for the derivative factor
        size_t nDim   = this->m_expansionList->GetShapeDimension();
        size_t nCoord = this->m_expansionList->GetCoordim(0);
        std::vector<vec_t, tinysimd::allocator<vec_t>> derivFac;
        derivFac.resize(nDim * nCoord * dfSize);
        // derivFac storage order: vec->dim->coord->point->element->block
        // original df: point->dim->coord->element->block
        size_t exp_id = 0;
        size_t jac_id = 0;

        for (size_t blk = 0; blk < blocks.size(); ++blk)
        {
            size_t num_elements         = blocks[blk].num_elements;
            size_t num_padding_elements = blocks[blk].num_padding_elements;
            size_t num_metaBlocks =
                (num_elements + num_padding_elements) / vec_t::width;

            auto expPtr = this->m_expansionList->GetExp(exp_id);

            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                alignas(vec_t::alignment) NekDouble tmp[vec_t::width];
                // loop over meta-blocks: except last one
                for (size_t e = 0; e < num_metaBlocks - 1; ++e)
                {
                    for (size_t pt = 0; pt < expPtr->GetTotPoints(); ++pt)
                    {
                        for (size_t d = 0; d < nDim * nCoord; ++d)
                        {
                            for (size_t i = 0; i < vec_t::width; ++i)
                            {
                                auto &df =
                                    this->m_expansionList->GetExp(exp_id + i)
                                        ->GetMetricInfo()
                                        ->GetDerivFactors(
                                            expPtr->GetPointsKeys());
                                tmp[i] = df[d][pt];
                            }
                            derivFac[jac_id++].load(&tmp[0]);
                        }
                    }
                    exp_id += vec_t::width;
                }
                // Last block: may have padding elements
                for (size_t pt = 0; pt < expPtr->GetTotPoints(); ++pt)
                {
                    for (size_t d = 0; d < nDim * nCoord; ++d)
                    {
                        for (size_t i = 0;
                             i < vec_t::width - num_padding_elements; ++i)
                        {
                            auto &df =
                                this->m_expansionList->GetExp(exp_id + i)
                                    ->GetMetricInfo()
                                    ->GetDerivFactors(expPtr->GetPointsKeys());
                            tmp[i] = df[d][pt];
                        }
                        for (size_t i = vec_t::width - num_padding_elements;
                             i < vec_t::width; ++i)
                        {
                            tmp[i] = 0.0;
                        }
                        derivFac[jac_id++].load(&tmp[0]);
                    }
                }
                exp_id += vec_t::width - num_padding_elements;
            }
            else
            {
                alignas(vec_t::alignment) NekDouble tmp[vec_t::width];
                // loop over meta-blocks: except last one
                for (size_t e = 0; e < num_metaBlocks - 1; ++e)
                {
                    for (size_t d = 0; d < nDim * nCoord; ++d)
                    {
                        for (size_t i = 0; i < vec_t::width; ++i)
                        {
                            auto &df =
                                this->m_expansionList->GetExp(exp_id + i)
                                    ->GetMetricInfo()
                                    ->GetDerivFactors(expPtr->GetPointsKeys());
                            tmp[i] = df[d][0];
                        }
                        derivFac[jac_id++].load(&tmp[0]);
                    }
                    exp_id += vec_t::width;
                }

                // Last block: may have padding elements
                for (size_t d = 0; d < nDim * nCoord; ++d)
                {
                    for (size_t i = 0; i < vec_t::width - num_padding_elements;
                         ++i)
                    {
                        auto &df =
                            this->m_expansionList->GetExp(exp_id + i)
                                ->GetMetricInfo()
                                ->GetDerivFactors(expPtr->GetPointsKeys());
                        tmp[i] = df[d][0];
                    }
                    for (size_t i = vec_t::width - num_padding_elements;
                         i < vec_t::width; ++i)
                    {
                        tmp[i] = 0.0;
                    }
                    derivFac[jac_id++].load(&tmp[0]);
                }
                exp_id += vec_t::width - num_padding_elements;
            }
        }

        return MemoryManager<std::vector<vec_t, tinysimd::allocator<vec_t>>>::
            AllocateSharedPtr(derivFac);
    }

    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
