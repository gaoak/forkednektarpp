///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzSerialStdMat.hpp
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

#include "Operators/ElmtOps/OperatorHelmholtz.hpp"

#include "Operators/ElmtOps/OperatorBwdTrans.hpp"
#include "Operators/ElmtOps/OperatorIProductWRTBase.hpp"
#include "Operators/ElmtOps/OperatorIProductWRTDerivBase.hpp"
#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorHelmholtzImpl : public OperatorHelmholtz<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtz<TData>(expansionList),
          m_diffCoeff(MemoryRegion<TData>::template Create<MemSpace>(
              "Helmholtz diffCoeff",
              expansionList->GetCoordim(0) * expansionList->GetCoordim(0),
              ExecSpace::alignment))
    {
        auto nCoord = this->m_expansionList->GetCoordim(0);

        m_diffCoeff.template Initialize<MemSpace>(0);

        TData *diffCoeff =
            m_diffCoeff.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

        for (size_t d = 0; d < nCoord; d++)
        {
            diffCoeff[d * nCoord + d] = 1.0; // temporary solution
        }

        m_BwdTransOp =
            BwdTrans<TData>::template Create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_PhysDerivOp =
            PhysDeriv<TData>::template Create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_IProductWRTBaseOp =
            IProductWRTBase<TData>::template Create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_IProductWRTDerivBaseOp = IProductWRTDerivBase<TData>::template Create<
            ExecSpace, Implementation>(this->m_expansionList);

        m_IProductWRTDerivBaseOp->SetAppend(true);

        m_PhysBlockAttributes =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        int CompSize = in.GetNumComponents();
        int nCoords  = this->m_expansionList->GetCoordim(0);

        // initialise bwd storage space if not for correct number of components
        if (m_bwd.GetNumComponents() != CompSize)
        {
            m_bwd = Field<TData, FieldState::Phys>::template Create<MemSpace>(
                "Helmholtz tmp", m_PhysBlockAttributes, CompSize,
                ExecSpace::alignment);
            m_deriv = Field<TData, FieldState::Phys>::template Create<MemSpace>(
                "Helmholtz bwd", m_PhysBlockAttributes, CompSize * nCoords,
                ExecSpace::alignment);
        }

        // Step 1: BwdTrans.
        this->m_BwdTransOp->apply(in, m_bwd);

        // Step 2: PhysDeriv.
        this->m_PhysDerivOp->apply(m_bwd, m_deriv);

        // Step 3: Inner product for mass matrix operation.
        this->m_IProductWRTBaseOp->apply(m_bwd, out);

        // Step 4: Multiply by diffusion coefficient.
        DiffusionCoeff(m_deriv);

        // Step 5: Inner product.
        this->m_IProductWRTDerivBaseOp->apply(m_deriv, out);
    }

    void DiffusionCoeff(Field<TData, FieldState::Phys> &deriv)
    {
        // Initialize pointers.
        TData *diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadWrite>();

        // Initialize index.
        size_t exp_idx = 0;

        for (size_t blk = 0; blk < deriv.GetBlocks().size(); ++blk)
        {
            // Block dependent.
            auto &derivblock  = deriv.GetBlocks()[blk];
            const auto nElmts = derivblock.GetNumElements();

            // Initialize pointers.
            auto derivPtr = derivblock.template GetPtr<MemSpace, ReadWrite>();

            // Determine shape and type of the element.
            const auto expPtr = this->m_expansionList->GetExp(exp_idx);
            auto nCoord       = expPtr->GetCoordim();
            auto nqTot        = expPtr->GetTotPoints();

            auto store = std::vector<Array<OneD, TData>>(nCoord);

            // Multiply by diffusion coefficient.
            for (size_t d = 0; d < nCoord; d++)
            {
                store[d] = Array<OneD, TData>(nqTot * nElmts);

                Vmath::Smul(nqTot * nElmts, diffCoeffPtr[d * nCoord], derivPtr,
                            1, store[d].data(), 1);

                for (size_t l = 1; l < nCoord; l++)
                {
                    Vmath::Svtvp(nqTot * nElmts, diffCoeffPtr[d * nCoord + l],
                                 derivPtr + l * derivblock.size(), 1,
                                 store[d].data(), 1, store[d].data(), 1);
                }
            }

            for (size_t d = 0; d < nCoord; d++)
            {
                Vmath::Vcopy(nqTot * nElmts, store[d].data(), 1,
                             derivPtr + d * derivblock.size(), 1);
            }

            // Increment index for next element type.
            exp_idx += nElmts;
        }
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

    void SetLambda(TData lambda) override
    {
        this->m_lambda = lambda;
        m_IProductWRTBaseOp->SetScale(this->m_lambda);
    }

private:
    Field<TData, FieldState::Phys> m_bwd;
    Field<TData, FieldState::Phys> m_deriv;

    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorPhysDeriv<TData>> m_PhysDerivOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    std::shared_ptr<OperatorIProductWRTDerivBase<TData>>
        m_IProductWRTDerivBaseOp;

    MemoryRegion<TData> m_diffCoeff;

    std::vector<BlockAttributes> m_PhysBlockAttributes;
};

} // namespace Nektar::Operators::detail
