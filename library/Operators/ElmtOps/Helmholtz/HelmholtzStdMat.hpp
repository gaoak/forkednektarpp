///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzStdMat.hpp
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

#include "Operators/ElmtOps/Helmholtz/OperatorHelmholtz.hpp"

#include "Operators/ElmtOps/BwdTrans/OperatorBwdTrans.hpp"
#include "Operators/ElmtOps/IProductWRTBase/OperatorIProductWRTBase.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/OperatorIProductWRTDerivBase.hpp"
#include "Operators/ElmtOps/PhysDeriv/OperatorPhysDeriv.hpp"

#include "Operators/ElmtOps/Helmholtz/HelmholtzStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorHelmholtzImpl : public BlockOperatorHelmholtz<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorHelmholtzImpl(const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorHelmholtz<TData>(exp, dataWarehouse),
          m_diffCoeff(MemoryRegion<TData>::Create(
              "Helmholtz diffCoeff", exp->GetCoordim() * exp->GetCoordim(),
              ExecSpace::alignment))
    {
        auto ncoord = this->m_exp->GetCoordim();

        m_diffCoeff.template Initialize<MemSpace>(0);

        TData *diffCoeff =
            m_diffCoeff.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

        for (unsigned int d = 0; d < ncoord; d++)
        {
            diffCoeff[d * ncoord + d] = 1.0; // temporary solution
        }

        this->m_BwdTransOp = BlockOperatorBwdTrans<TData>::Create(
            this->m_exp, this->m_dataWarehouse, ExecSpace::name,
            Implementation::name);
        this->m_PhysDerivOp = BlockOperatorPhysDeriv<TData>::Create(
            this->m_exp, this->m_dataWarehouse, ExecSpace::name,
            Implementation::name);
        this->m_IProductWRTBaseOp = BlockOperatorIProductWRTBase<TData>::Create(
            this->m_exp, this->m_dataWarehouse, ExecSpace::name,
            Implementation::name);
        this->m_IProductWRTDerivBaseOp =
            BlockOperatorIProductWRTDerivBase<TData>::Create(
                this->m_exp, this->m_dataWarehouse, ExecSpace::name,
                Implementation::name);

        this->m_IProductWRTDerivBaseOp->SetAppend(true);
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BlockOperatorHelmholtzImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    MemoryRegion<TData> m_bwd;
    MemoryRegion<TData> m_deriv;

    std::shared_ptr<BlockOperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<BlockOperatorPhysDeriv<TData>> m_PhysDerivOp;
    std::shared_ptr<BlockOperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    std::shared_ptr<BlockOperatorIProductWRTDerivBase<TData>>
        m_IProductWRTDerivBaseOp;

    MemoryRegion<TData> m_diffCoeff;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        auto CompSize = inblock.GetNumComponents();
        auto ncoords  = this->m_exp->GetCoordim();

        // initialise bwd storage space if not for correct number of components
        auto size = inblock.GetNumElementsWithPadding() *
                    this->m_exp->GetTotPoints() * CompSize;
        if (this->m_bwd.size() != size)
        {
            this->m_bwd   = MemoryRegion<TData>::Create("Helmholtz bwd", size,
                                                        ExecSpace::alignment);
            this->m_deriv = MemoryRegion<TData>::Create(
                "Helmholtz deriv", size * ncoords, ExecSpace::alignment);
        }

        auto bwd = BlockAccessor(inblock.GetExpIdx(), inblock.GetNumElements(),
                                 inblock.GetNumElementsWithPadding(),
                                 this->m_exp->GetTotPoints(), 1, this->m_bwd,
                                 CompSize, 0);
        auto deriv = BlockAccessor(
            inblock.GetExpIdx(), inblock.GetNumElements(),
            inblock.GetNumElementsWithPadding(), this->m_exp->GetTotPoints(), 1,
            this->m_deriv, CompSize * ncoords, 0);

        // Step 1: BwdTrans.
        this->m_BwdTransOp->Apply(inblock, bwd);

        // Step 2: PhysDeriv.
        this->m_PhysDerivOp->Apply(bwd, deriv);

        // Step 3: Inner product for mass matrix operation.
        this->m_IProductWRTBaseOp->Apply(bwd, outblock);

        // Step 4: Multiply by diffusion coefficient.
        DiffusionCoeff(deriv);

        // Step 5: Inner product.
        this->m_IProductWRTDerivBaseOp->Apply(deriv, outblock);
    }

    void DiffusionCoeff(BlockAccessor<TData> &deriv)
    {
        // Initialize pointers.
        TData *diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadWrite>();

        // Initialize pointers.
        auto derivPtr = deriv.template GetPtr<MemSpace, ReadWrite>();

        // Determine shape and type of the element.
        const auto ncoord = this->m_exp->GetCoordim();
        const auto nelmt  = deriv.GetNumElementsWithPadding();
        const auto nqTot  = deriv.GetNumData();
        const auto ncomp  = deriv.GetNumComponents() / ncoord;

        MultiplyByDiffusionCoeff<ExecSpace>(nelmt, nqTot, ncoord, ncomp,
                                            diffCoeffPtr, derivPtr);
    }

    void v_SetLambda(const TData &lambda) override
    {
        this->m_lambda = lambda;

        // Update IProductWRTBase operator
        m_IProductWRTBaseOp->SetScale(lambda);
    }
};

} // namespace Nektar::Operators::detail
