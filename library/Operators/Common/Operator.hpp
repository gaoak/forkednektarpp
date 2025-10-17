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
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <LibUtilities/Communication/Comm.h>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <MultiRegions/ExpList.h>

#include "Operators/Common/BasisDataWarehouse.hpp"
#include "Operators/Common/GeometricDataWarehouse.hpp"
#include "Operators/Common/LocalToGlobalDataWarehouse.hpp"
#include "Operators/Common/ModeIndexDataWarehouse.hpp"
#include "Operators/Common/NekDataWarehouse.hpp"
#include "Operators/Common/OperatorsDeclspec.hpp"
#include "Operators/Common/Spaces.hpp"
#include "Operators/Common/StdMatDataWarehouse.hpp"
#include "Operators/Field/Field.hpp"

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
    static inline const std::string name = "StdMat";
};

struct SumFac
{
    static inline const std::string name = "SumFac";
};

struct SumFacTOP
{
    static inline const std::string name = "SumFacTOP";
};

struct SumFacMat
{
    static constexpr char name[] = "SumFacMat";
};

struct Generic
{
    static inline const std::string name = "Generic";
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

    template <template <typename> typename TOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string execStr)
    {
        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? session->GetCmdLineArgument<std::string>("opExecSpace")
                : execStr;

        std::string requestedKey = TOperator<TData>::name + execStr0;

        OperatorFactory<TData> &factory = GetOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<TOperator<TData>>(
            factory.CreateInstance(requestedKey, expansionList));
    }

protected:
    MultiRegions::ExpListSharedPtr m_expansionList;
    NekDataWarehouseSharedPtr m_dataWarehouse;
};

template <typename Implementation>
NEK_FORCE_INLINE static constexpr unsigned int GetDeviceBlockSize(
    [[maybe_unused]] const unsigned int blockSize)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr auto warpsize = NektarSpaces::vector_width<double>::value;
        return warpsize;
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        constexpr auto warpsize = NektarSpaces::vector_width<double>::value;
        return std::min(((blockSize + warpsize - 1u) / warpsize) * warpsize,
                        NektarSpaces::Device::defaultBlockSize);
    }
    else
    {
        return 0;
    }
}

template <typename Implementation>
NEK_FORCE_INLINE static constexpr unsigned int GetDeviceGridSize(
    const size_t nelmt)
{
    size_t maxGridSize = 2147483647;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr auto warpsize = NektarSpaces::vector_width<double>::value;
        return std::min((nelmt + warpsize - 1u) / warpsize, maxGridSize);
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        return std::min(nelmt, maxGridSize);
    }
    else
    {
        return 0;
    }
}

} // namespace Nektar::Operators
