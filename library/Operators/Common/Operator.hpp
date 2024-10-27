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

template <bool B, typename T> struct simd_type_if
{
    typedef tinysimd::scalarT<T> type;
};

template <typename T> struct simd_type_if<true, T>
{
    typedef tinysimd::simd<T> type;
};

// Core implementation types.
struct StdMat
{
};

struct SumFac
{
};

struct SumFacQP
{
};

struct Generic
{
};

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

            if (cmdValue == "Serial")
            {
                g_OpExecSpace = "Serial";
            }
            else if (cmdValue == "AVX")
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
            else if (cmdValue == "SYCL")
            {
                g_OpExecSpace = "SYCL";
            }
            else if (cmdValue == "Kokkos")
            {
                g_OpExecSpace = "Kokkos";
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

            if (cmdValue == "StdMat")
            {
                g_OpImpl = "StdMat";
            }
            else if (cmdValue == "SumFac")
            {
                g_OpImpl = "SumFac";
            }
            else if (cmdValue == "SumFacQP")
            {
                g_OpImpl = "SumFacQP";
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
        // Check for a struct root.
        Nektar::stripString(execStr, "struct ");
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
        // Check for a struct root.
        Nektar::stripString(implStr, "struct ");
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
        constexpr size_t nOpTests = 2;
#else
        constexpr size_t nOpTests = 4;
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
                    // Find the operator with the requested ExecSpace and a
                    // general implementation.
                    key = descriptStr + fieldStateStr + execStr + "Generic";
                    break;
                case 2:
                    // Find the operator with the "Serial" ExecSpace and the
                    // same implementation.
                    key = descriptStr + fieldStateStr + "Serial" + implStr;
                    break;
                case 3:
                    // Find the operator with the "Serial" ExecSpace and
                    // the "StdMat" implementation.
                    key = descriptStr + fieldStateStr + "Serial" + "StdMat";
                    break;
                default:
                    break;
            }

            if (factory.ModuleExists(key))
            {
                if (key != requestedKey && i != 1)
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
    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
