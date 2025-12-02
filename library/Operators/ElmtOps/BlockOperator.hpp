///////////////////////////////////////////////////////////////////////////////
//
// File: BlockOperator.hpp
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

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LocalRegions/Expansion.h>

#include "Operators/Common/BasisDataWarehouse.hpp"
#include "Operators/Common/GeometricDataWarehouse.hpp"
#include "Operators/Common/ModeIndexDataWarehouse.hpp"
#include "Operators/Common/NekDataWarehouse.hpp"
#include "Operators/Common/StdMatDataWarehouse.hpp"

#include "Operators/Field/Field.hpp"

namespace Nektar::Operators
{

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

// Forward-declare the BlockOperator base class so we can define the factory
template <typename TData> class BlockOperator;

// BlockOperator factory singleton
template <typename TData>
using BlockOperatorFactory = Nektar::LibUtilities::NekFactory<
    std::string, BlockOperator<TData>, const unsigned int,
    const LocalRegions::ExpansionSharedPtr &, NekDataWarehouseSharedPtr>;

// BlockOperator factory singleton
template <typename TData>
BlockOperatorFactory<TData> &GetBlockOperatorFactory();

template <typename TData> class BlockOperator
{
public:
    virtual ~BlockOperator() = default;

    template <template <typename> typename TOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse, std::string execStr,
        std::string implStr)
    {
        std::string requestedKey = TOperator<TData>::name + execStr + implStr;

        BlockOperatorFactory<TData> &factory = GetBlockOperatorFactory<TData>();

        // No suitible operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<TOperator<TData>>(
            factory.CreateInstance(requestedKey, block_idx, exp,
                                   dataWarehouse));
    }

    void Apply(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

    void operator()(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

protected:
    unsigned int m_block_idx;
    LocalRegions::ExpansionSharedPtr m_exp;
    NekDataWarehouseSharedPtr m_dataWarehouse;

    BlockOperator(const unsigned int block_idx,
                  const LocalRegions::ExpansionSharedPtr &exp,
                  NekDataWarehouseSharedPtr dataWarehouse)
        : m_block_idx(block_idx), m_exp(exp), m_dataWarehouse(dataWarehouse)
    {
    }

    virtual void v_Apply(BlockAccessor<TData> &inblock,
                         BlockAccessor<TData> &outblock) = 0;
};

template <typename Implementation>
NEK_FORCE_INLINE static constexpr unsigned int GetDeviceBlockSize(
    [[maybe_unused]] const unsigned int blockSize)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return warpsize;
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
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
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
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
