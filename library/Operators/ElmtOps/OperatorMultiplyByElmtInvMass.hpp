///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorMultiplyByElmtInvMass.hpp
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

template <typename TData> struct MultiplyByElmtInvMass;

template <typename TData>
class BlockOperatorMultiplyByElmtInvMass : public BlockOperator<TData>
{
public:
    BlockOperatorMultiplyByElmtInvMass(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(exp, dataWarehouse)
    {
    }

    ~BlockOperatorMultiplyByElmtInvMass() override = default;

    virtual void apply(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock) = 0;

    virtual void operator()(BlockAccessor<TData> &inblock,
                            BlockAccessor<TData> &outblock)
    {
        this->apply(inblock, outblock);
    }

    void SetInvMassMatrix(std::vector<TData> &dmat)
    {
        v_SetInvMassMatrix(dmat);
    }

protected:
    virtual void v_SetInvMassMatrix(std::vector<TData> &dmat) = 0;
};

// Descriptor / traits class for BlockMultiplyByElmtInvMass
template <typename TData> struct BlockMultiplyByElmtInvMass
{
    using class_name = BlockOperatorMultiplyByElmtInvMass<TData>;

    BlockMultiplyByElmtInvMass() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return BlockOperator<TData>::template Create<
            BlockMultiplyByElmtInvMass<TData>, ExecSpace, Impl>(exp,
                                                                dataWarehouse);
    }
};

// MultiplyByElmtInvMass base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorMultiplyByElmtInvMass
    : public OperatorElmt<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend struct MultiplyByElmtInvMass<TData>;

public:
    OperatorMultiplyByElmtInvMass(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorElmt<FieldState::Coeff, FieldState::Coeff, TData>(
              expansionList)
    {
    }

    ~OperatorMultiplyByElmtInvMass() override = default;

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

protected:
    std::vector<std::shared_ptr<BlockOperatorMultiplyByElmtInvMass<TData>>>
        m_blockOperator;
};

// Descriptor / traits class for MultiplyByElmtInvMass
template <typename TData> struct MultiplyByElmtInvMass
{
    using class_name = OperatorMultiplyByElmtInvMass<TData>;

    MultiplyByElmtInvMass() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        auto MultiplyByElmtInvMassOp =
            Operator<TData>::template Create<MultiplyByElmtInvMass<TData>,
                                             ExecSpace, Impl>(expansionList);

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);

        // Loop over the blocks.
        std::vector<TData> dmat;
        for (auto &block : blocks)
        {
            const auto exp   = expansionList->GetExp(block.GetExpIdx());
            const auto nmTot = exp->GetNcoeffs();
            const auto deformed =
                exp->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
            const auto nElmts = block.GetNumElements();

            if (deformed)
            {
                dmat.resize(nElmts * nmTot * nmTot);
                auto dmatptr = dmat.data();
                for (unsigned int e = 0; e < nElmts; ++e)
                {
                    const auto exp =
                        expansionList->GetExp(block.GetExpIdx() + e);
                    const auto &InvMass =
                        exp->GetLocMatrix(StdRegions::eInvMass);
                    std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot, dmatptr);
                    dmatptr += nmTot * nmTot;
                }
            }

            MultiplyByElmtInvMassOp->m_blockOperator.push_back(
                BlockMultiplyByElmtInvMass<TData>::template Create<ExecSpace,
                                                                   Impl>(
                    expansionList->GetExp(block.GetExpIdx()),
                    expansionList->GetDataWarehouseSharedPtr()));

            MultiplyByElmtInvMassOp->m_blockOperator.back()->SetInvMassMatrix(
                dmat);
        }

        return MultiplyByElmtInvMassOp;
    }
};

} // namespace Nektar::Operators

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorMultiplyByElmtInvMassImpl
    : public OperatorMultiplyByElmtInvMass<TData>
{
public:
    OperatorMultiplyByElmtInvMassImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMultiplyByElmtInvMass<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorMultiplyByElmtInvMassImpl<
            ExecSpace, Implementation, TData>>(expansionList);
    }
};

} // namespace Nektar::Operators::detail
