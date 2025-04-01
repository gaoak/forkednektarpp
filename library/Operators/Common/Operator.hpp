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

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/MiscUtils.hpp>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <LibUtilities/Communication/Comm.h>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <MultiRegions/ExpList.h>

#include "Operators/Common/BasisDataWarehouse.hpp"
#include "Operators/Common/GeometricDataWarehouse.hpp"
#include "Operators/Common/ModeIndexDataWarehouse.hpp"
#include "Operators/Common/NekDataWarehouse.hpp"
#include "Operators/Common/OperatorsDeclspec.hpp"
#include "Operators/Common/Spaces.hpp"
#include "Operators/Common/StdMatDataWarehouse.hpp"
#include "Operators/Field/Field.hpp"

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

template <bool B, typename TData> struct simd_type_if
{
    typedef tinysimd::scalarT<TData> type;
};

template <typename TData> struct simd_type_if<true, TData>
{
    typedef tinysimd::simd<TData> type;
};

// Core implementation types.
struct StdMat
{
    static constexpr char name[] = "StdMat";
};

struct SumFac
{
    static constexpr char name[] = "SumFac";
};

struct SumFacQP
{
    static constexpr char name[] = "SumFacQP";
};

struct Generic
{
    static constexpr char name[] = "Generic";
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
        : m_expansionList(expansionList),
          m_dataWarehouse(expansionList->GetDataWarehouseSharedPtr())
    {
    }

    template <typename TOperator>
    static std::shared_ptr<TOperator> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string execStr)
    {
        std::string descriptStr  = TOperator::name;
        std::string requestedKey = descriptStr + execStr;

        OperatorFactory<TData> &factory = GetOperatorFactory<TData>();

        bool notFound = true;

        if (factory.ModuleExists(requestedKey))
        {
            notFound = false;
        }

        // No suitible operator was found.
        if (notFound)
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<TOperator>(
            factory.CreateInstance(requestedKey, expansionList));
    }

protected:
    MultiRegions::ExpListSharedPtr m_expansionList;
    NekDataWarehouseSharedPtr m_dataWarehouse;
};

template <typename Implementation>
NEK_FORCE_INLINE static unsigned int GetDeviceBlockSize(
    [[maybe_unused]] const unsigned int blockSize)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        return NektarSpaces::Device::defaultBlockSize;
    }
    else
    {
        auto warpsize = NektarSpaces::vector_width<double>::value;
        return std::min(((blockSize + warpsize - 1u) / warpsize) * warpsize,
                        NektarSpaces::Device::defaultBlockSize);
    }
}

template <typename Implementation>
NEK_FORCE_INLINE static unsigned int GetDeviceGridSize(const unsigned int nelmt)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int blocksize =
            NektarSpaces::Device::defaultBlockSize;
        return std::min((nelmt + blocksize - 1u) / blocksize, 2147483647u);
    }
    else
    {
        return std::min(nelmt, 2147483647u);
    }
}

} // namespace Nektar::Operators
