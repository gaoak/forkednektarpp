///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorIProductWRTDerivBase.hpp
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

template <typename TData> struct IProductWRTDerivBase;

template <typename TData>
class BlockOperatorIProductWRTDerivBase : public BlockOperator<TData>
{
public:
    BlockOperatorIProductWRTDerivBase(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(exp, dataWarehouse)
    {
    }

    ~BlockOperatorIProductWRTDerivBase() override = default;

    virtual void apply(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock) = 0;

    virtual void operator()(BlockAccessor<TData> &inblock,
                            BlockAccessor<TData> &outblock)
    {
        this->apply(inblock, outblock);
    }

    void SetAppend(bool append)
    {
        m_append = append;
    }

protected:
    bool m_append = false;
};

// Descriptor / traits class for BlockIProductWRTDerivBase
template <typename TData> struct BlockIProductWRTDerivBase
{
    using class_name = BlockOperatorIProductWRTDerivBase<TData>;

    BlockIProductWRTDerivBase() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return BlockOperator<TData>::template Create<
            BlockIProductWRTDerivBase<TData>, ExecSpace, Impl>(exp,
                                                               dataWarehouse);
    }
};

// IProductWRTDerivBase base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorIProductWRTDerivBase
    : public OperatorElmt<FieldState::Phys, FieldState::Coeff, TData>
{
    friend struct IProductWRTDerivBase<TData>;

public:
    OperatorIProductWRTDerivBase(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorElmt<FieldState::Phys, FieldState::Coeff, TData>(
              expansionList)
    {
    }

    ~OperatorIProductWRTDerivBase() override = default;

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(out.GetNumComponents() ==
                     in.GetNumComponents() /
                         this->m_expansionList->GetCoordim(0),
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

    void SetAppend(bool append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOperator.size(); ++blk)
        {
            this->m_blockOperator[blk]->SetAppend(append);
        }
    }

protected:
    std::vector<std::shared_ptr<BlockOperatorIProductWRTDerivBase<TData>>>
        m_blockOperator;
};

// Descriptor / traits class for IProductWRTDerivBase
template <typename TData> struct IProductWRTDerivBase
{
    using class_name = OperatorIProductWRTDerivBase<TData>;

    IProductWRTDerivBase() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        auto IProductWRTDerivBaseOp =
            Operator<TData>::template Create<IProductWRTDerivBase<TData>,
                                             ExecSpace, Impl>(expansionList);

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);

        // Loop over the blocks.
        for (auto &block : blocks)
        {
            IProductWRTDerivBaseOp->m_blockOperator.push_back(
                BlockIProductWRTDerivBase<TData>::template Create<ExecSpace,
                                                                  Impl>(
                    expansionList->GetExp(block.GetExpIdx()),
                    expansionList->GetDataWarehouseSharedPtr()));
        }

        return IProductWRTDerivBaseOp;
    }
};

} // namespace Nektar::Operators

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorIProductWRTDerivBaseImpl
    : public OperatorIProductWRTDerivBase<TData>
{
public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }
};

} // namespace Nektar::Operators::detail
