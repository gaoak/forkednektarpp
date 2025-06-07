///////////////////////////////////////////////////////////////////////////////
//
// File: MassSerialStdMatXSMM.hpp
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

#include "Operators/ElmtOps/Mass/OperatorMass.hpp"

#include "Operators/ElmtOps/BwdTrans/OperatorBwdTrans.hpp"
#include "Operators/ElmtOps/IProductWRTBase/OperatorIProductWRTBase.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorMassImpl : public BlockOperatorMass<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorMassImpl(const LocalRegions::ExpansionSharedPtr &exp,
                          NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorMass<TData>(exp, dataWarehouse)
    {
        this->m_BwdTransOp = BlockOperatorBwdTrans<TData>::Create(
            this->m_exp, this->m_dataWarehouse, ExecSpace::name,
            Implementation::name);
        this->m_IProductWRTBaseOp = BlockOperatorIProductWRTBase<TData>::Create(
            this->m_exp, this->m_dataWarehouse, ExecSpace::name,
            Implementation::name);
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BlockOperatorMassImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    MemoryRegion<TData> m_bwd;

    std::shared_ptr<BlockOperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<BlockOperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        auto CompSize = inblock.GetNumComponents();

        // Initialise bwd storage space if not for correct number of components.
        auto size = inblock.GetNumElementsWithPadding() *
                    this->m_exp->GetTotPoints() * CompSize;
        if (this->m_bwd.size() != size)
        {
            this->m_bwd = MemoryRegion<TData>::Create("Mass bwd", size,
                                                      ExecSpace::alignment);
        }

        auto bwd = BlockAccessor(inblock.GetExpIdx(), inblock.GetNumElements(),
                                 inblock.GetNumElementsWithPadding(),
                                 this->m_exp->GetTotPoints(), 1, this->m_bwd,
                                 CompSize, 0);

        // Step 1: BwdTrans.
        this->m_BwdTransOp->Apply(inblock, bwd);

        // Step 2: Inner product for mass matrix operation.
        this->m_IProductWRTBaseOp->Apply(bwd, outblock);
    }
};

} // namespace Nektar::Operators::detail
