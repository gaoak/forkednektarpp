///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorIProductWRTBase.hpp
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

#include "Operators/ElmtOps/OperatorElmt.hpp"

namespace Nektar::Operators
{

template <typename TData> struct IProductWRTBase;

template <typename TData>
class BlockOperatorIProductWRTBase : public BlockOperator<TData>
{
public:
    BlockOperatorIProductWRTBase(const LocalRegions::ExpansionSharedPtr &exp,
                                 NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(exp, dataWarehouse)
    {
    }

    ~BlockOperatorIProductWRTBase() override = default;

    virtual void apply(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock) = 0;

    virtual void operator()(BlockAccessor<TData> &inblock,
                            BlockAccessor<TData> &outblock)
    {
        this->apply(inblock, outblock);
    }

    void SetScale(TData scale)
    {
        m_scale = scale;
    }

protected:
    TData m_scale = 1.0;
};

// Descriptor / traits class for BlockIProductWRTBase
template <typename TData> struct BlockIProductWRTBase
{
    using class_name = BlockOperatorIProductWRTBase<TData>;

    BlockIProductWRTBase() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return BlockOperator<TData>::template Create<
            BlockIProductWRTBase<TData>, ExecSpace, Impl>(exp, dataWarehouse);
    }
};

// IProductWRTBase base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorIProductWRTBase
    : public OperatorElmt<FieldState::Phys, FieldState::Coeff, TData>
{
    friend struct IProductWRTBase<TData>;

public:
    OperatorIProductWRTBase(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorElmt<FieldState::Phys, FieldState::Coeff, TData>(
              expansionList)
    {
    }

    ~OperatorIProductWRTBase() override = default;

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOperator.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOperator[blk]->apply(inblock, outblock);
        }
    }

    virtual void operator()(Field<TData, FieldState::Phys> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        this->apply(in, out);
    }

    void SetScale(TData scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOperator.size(); ++blk)
        {
            this->m_blockOperator[blk]->SetScale(scale);
        }
    }

protected:
    std::vector<std::shared_ptr<BlockOperatorIProductWRTBase<TData>>>
        m_blockOperator;
};

// Descriptor / traits class for IProductWRTBase
template <typename TData> struct IProductWRTBase
{
    using class_name = OperatorIProductWRTBase<TData>;

    IProductWRTBase() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        auto IProductWRTBaseOp =
            Operator<TData>::template Create<IProductWRTBase<TData>, ExecSpace,
                                             Impl>(expansionList);

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);

        // Loop over the blocks.
        for (auto &block : blocks)
        {
            IProductWRTBaseOp->m_blockOperator.push_back(
                BlockIProductWRTBase<TData>::template Create<ExecSpace, Impl>(
                    expansionList->GetExp(block.GetExpIdx()),
                    expansionList->GetDataWarehouseSharedPtr()));
        }

        return IProductWRTBaseOp;
    }
};

} // namespace Nektar::Operators

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }
};

} // namespace Nektar::Operators::detail
