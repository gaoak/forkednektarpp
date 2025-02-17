///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorPhysInterp1DScaled.hpp
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

template <typename TData> struct PhysInterp1DScaled;

template <typename TData>
class BlockOperatorPhysInterp1DScaled : public BlockOperator<TData>
{
public:
    BlockOperatorPhysInterp1DScaled(const LocalRegions::ExpansionSharedPtr &exp,
                                    NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(exp, dataWarehouse)
    {
    }

    ~BlockOperatorPhysInterp1DScaled() override = default;

    virtual void apply(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock) = 0;

    virtual void operator()(BlockAccessor<TData> &inblock,
                            BlockAccessor<TData> &outblock)
    {
        this->apply(inblock, outblock);
    }

    void SetScaleFactor(TData scale)
    {
        m_scale = scale;
    }

protected:
    TData m_scale = -1.0; // scaling factor
};

// Descriptor / traits class for BlockPhysInterp1DScaled
template <typename TData> struct BlockPhysInterp1DScaled
{
    using class_name = BlockOperatorPhysInterp1DScaled<TData>;

    BlockPhysInterp1DScaled() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return BlockOperator<TData>::template Create<
            BlockPhysInterp1DScaled<TData>, ExecSpace, Impl>(exp,
                                                             dataWarehouse);
    }
};

// PhysInterp1DScaled base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorPhysInterp1DScaled
    : public OperatorElmt<FieldState::Phys, FieldState::Phys, TData>
{
    friend struct PhysInterp1DScaled<TData>;

public:
    OperatorPhysInterp1DScaled(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorElmt<FieldState::Phys, FieldState::Phys, TData>(expansionList)
    {
    }

    ~OperatorPhysInterp1DScaled() override = default;

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
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
                            Field<TData, FieldState::Phys> &out)
    {
        this->apply(in, out);
    }

    void SetScaleFactor(TData scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOperator.size(); ++blk)
        {
            this->m_blockOperator[blk]->SetScaleFactor(scale);
        }
    }

protected:
    std::vector<std::shared_ptr<BlockOperatorPhysInterp1DScaled<TData>>>
        m_blockOperator;
};

// Descriptor / traits class for PhysInterp1DScaled
template <typename TData> struct PhysInterp1DScaled
{
    using class_name = OperatorPhysInterp1DScaled<TData>;

    PhysInterp1DScaled() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        auto PhysInterp1DScaledOp =
            Operator<TData>::template Create<PhysInterp1DScaled<TData>,
                                             ExecSpace, Impl>(expansionList);

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);

        // Loop over the blocks.
        for (auto &block : blocks)
        {
            PhysInterp1DScaledOp->m_blockOperator.push_back(
                BlockPhysInterp1DScaled<TData>::template Create<ExecSpace,
                                                                Impl>(
                    expansionList->GetExp(block.GetExpIdx()),
                    expansionList->GetDataWarehouseSharedPtr()));
        }

        return PhysInterp1DScaledOp;
    }
};

} // namespace Nektar::Operators

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorPhysInterp1DScaledImpl : public OperatorPhysInterp1DScaled<TData>
{
public:
    OperatorPhysInterp1DScaledImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysInterp1DScaled<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorPhysInterp1DScaledImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }
};

} // namespace Nektar::Operators::detail
