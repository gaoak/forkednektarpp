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

template <typename TData>
class BlockOperatorIProductWRTBase : public BlockOperator<TData>
{
public:
    static std::shared_ptr<BlockOperatorIProductWRTBase<TData>> Create(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse, std::string execStr,
        std::string implStr)
    {
        return BlockOperator<TData>::template Create<
            BlockOperatorIProductWRTBase<TData>>(exp, dataWarehouse, execStr,
                                                 implStr);
    }

    static constexpr char name[] = "BlockIProductWRTBase";

    void Apply(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

    void operator()(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

    void SetScale(TData scale)
    {
        m_scale = scale;
    }

protected:
    TData m_scale = 1.0;

    BlockOperatorIProductWRTBase(const LocalRegions::ExpansionSharedPtr &exp,
                                 NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(exp, dataWarehouse)
    {
    }

    ~BlockOperatorIProductWRTBase() override = default;

    virtual void v_Apply(BlockAccessor<TData> &inblock,
                         BlockAccessor<TData> &outblock) = 0;
};

// IProductWRTBase base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorIProductWRTBase
    : public OperatorElmt<FieldState::Phys, FieldState::Coeff, TData>
{
public:
    static std::shared_ptr<OperatorIProductWRTBase<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? session->GetCmdLineArgument<std::string>("opExecSpace")
                : execStr;
        std::string implStr0 =
            (implStr == "") ? session->GetCmdLineArgument<std::string>("opImpl")
                            : implStr;

        auto IProductWRTBaseOp =
            Operator<TData>::template Create<OperatorIProductWRTBase<TData>>(
                expansionList, execStr0);

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);

        // Loop over the blocks.
        for (auto &block : blocks)
        {
            IProductWRTBaseOp->m_blockOperator.push_back(
                BlockOperatorIProductWRTBase<TData>::Create(
                    expansionList->GetExp(block.GetExpIdx()),
                    expansionList->GetDataWarehouseSharedPtr(), execStr0,
                    implStr0));
        }

        return IProductWRTBaseOp;
    }

    static constexpr char name[] = "IProductWRTBase";

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

    OperatorIProductWRTBase(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorElmt<FieldState::Phys, FieldState::Coeff, TData>(
              expansionList)
    {
    }

    ~OperatorIProductWRTBase() override = default;

    void v_Apply(Field<TData, FieldState::Phys> &in,
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

            this->m_blockOperator[blk]->Apply(inblock, outblock);
        }
    }
};

} // namespace Nektar::Operators
