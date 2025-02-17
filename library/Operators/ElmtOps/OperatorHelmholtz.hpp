///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorHelmholtz.hpp
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

template <typename TData> struct Helmholtz;

template <typename TData>
class BlockOperatorHelmholtz : public BlockOperator<TData>
{
public:
    BlockOperatorHelmholtz(const LocalRegions::ExpansionSharedPtr &exp,
                           NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(exp, dataWarehouse)
    {
    }

    ~BlockOperatorHelmholtz() override = default;

    virtual void apply(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock) = 0;

    virtual void operator()(BlockAccessor<TData> &inblock,
                            BlockAccessor<TData> &outblock)
    {
        this->apply(inblock, outblock);
    }

    void SetLambda(TData lambda)
    {
        m_lambda = lambda;
    }

protected:
    TData m_lambda = 1.0;
};

// Descriptor / traits class for BlockHelmholtz
template <typename TData> struct BlockHelmholtz
{
    using class_name = BlockOperatorHelmholtz<TData>;

    BlockHelmholtz() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return BlockOperator<TData>::template Create<BlockHelmholtz<TData>,
                                                     ExecSpace, Impl>(
            exp, dataWarehouse);
    }
};

// Helmholtz base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorHelmholtz
    : public OperatorElmt<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend struct Helmholtz<TData>;

public:
    OperatorHelmholtz(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorElmt<FieldState::Coeff, FieldState::Coeff, TData>(
              expansionList)
    {
    }

    ~OperatorHelmholtz() override = default;

    void apply(Field<TData, FieldState::Coeff> &in,
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

    virtual void operator()(Field<TData, FieldState::Coeff> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        this->apply(in, out);
    }

    void SetLambda(TData lambda)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOperator.size(); ++blk)
        {
            this->m_blockOperator[blk]->SetLambda(lambda);
        }
    }

protected:
    std::vector<std::shared_ptr<BlockOperatorHelmholtz<TData>>> m_blockOperator;
};

// Descriptor / traits class for Helmholtz
template <typename TData> struct Helmholtz
{
    using class_name = OperatorHelmholtz<TData>;

    Helmholtz() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        auto HelmholtzOp =
            Operator<TData>::template Create<Helmholtz<TData>, ExecSpace, Impl>(
                expansionList);

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);

        // Loop over the blocks.
        for (auto &block : blocks)
        {
            HelmholtzOp->m_blockOperator.push_back(
                BlockHelmholtz<TData>::template Create<ExecSpace, Impl>(
                    expansionList->GetExp(block.GetExpIdx()),
                    expansionList->GetDataWarehouseSharedPtr()));
        }

        return HelmholtzOp;
    }
};

} // namespace Nektar::Operators

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorHelmholtzImpl : public OperatorHelmholtz<TData>
{
public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtz<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorHelmholtzImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }
};

} // namespace Nektar::Operators::detail
