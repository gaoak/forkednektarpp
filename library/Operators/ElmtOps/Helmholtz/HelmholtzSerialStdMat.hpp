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

#include "ElmtOps/OperatorHelmholtz.hpp"

#include "ElmtOps/OperatorBwdTrans.hpp"
#include "ElmtOps/OperatorIProductWRTBase.hpp"
#include "ElmtOps/OperatorIProductWRTDerivBase.hpp"
#include "ElmtOps/OperatorPhysDeriv.hpp"

#include "Common/OperatorHelper.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::StdMat>::value>::type>
class OperatorHelmholtzImpl : public OperatorHelmholtz<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtz<TData>(expansionList),
          m_bwd(Field<TData, FieldState::Phys>::template create<MemSpace>(
              "Helmholtz bwd",
              GetBlockAttributes(FieldState::Phys, expansionList,
                                 ExecSpace::width),
              1, ExecSpace::alignment)),

          m_deriv(Field<TData, FieldState::Phys>::template create<MemSpace>(
              "Helmholtz deriv",
              GetBlockAttributes(FieldState::Phys, expansionList,
                                 ExecSpace::width),
              expansionList->GetCoordim(0), ExecSpace::alignment)),
          m_derivCoeff(
              Field<TData, FieldState::Phys>::template create<MemSpace>(
                  GetBlockAttributes(FieldState::Phys, expansionList,
                                     ExecSpace::width),
                  expansionList->GetCoordim(0), ExecSpace::alignment)),
          m_diffCoeff(MemoryRegion<TData>::template create<MemSpace>(
              "Helmholtz diffCoeff",
              expansionList->GetCoordim(0) * expansionList->GetCoordim(0),
              ExecSpace::alignment))
    {
        auto nCoord = this->m_expansionList->GetCoordim(0);

        m_diffCoeff = MemoryRegion<TData>::template create<MemSpace>(
            "Helmholtz diffCoeff", nCoord * nCoord, ExecSpace::alignment);

        m_diffCoeff.initialize(0);

        TData *diffCoeff =
            m_diffCoeff.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

        for (size_t d = 0; d < nCoord; d++)
        {
            diffCoeff[d * nCoord + d] = 1.0; // temporary solution
        }

        m_BwdTransOp =
            BwdTrans<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_PhysDerivOp =
            PhysDeriv<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_IProductWRTBaseOp =
            IProductWRTBase<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_IProductWRTDerivBaseOp = IProductWRTDerivBase<TData>::template create<
            ExecSpace, Implementation>(this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Step 1: BwdTrans
        this->m_BwdTransOp->apply(in, this->m_bwd);

        // Step 2: PhysDeriv
        this->m_PhysDerivOp->apply(this->m_bwd, this->m_deriv);

        // Step 3: Inner product for mass matrix operation
        this->m_IProductWRTBaseOp->apply(this->m_bwd, out, this->m_lambda);

        // Step 4: Multiply by diffusion coefficient
        DiffusionCoeff(this->m_deriv, m_derivCoeff);

        // Step 5: Inner product
        this->m_IProductWRTDerivBaseOp->apply(m_derivCoeff, out, true);
    }

    void DiffusionCoeff(Field<TData, FieldState::Phys> &deriv,
                        Field<TData, FieldState::Phys> &derivCoeff)
    {
        ASSERTL1(deriv.GetVecWidth() == derivCoeff.GetVecWidth(),
                 "Input and output widths are different but kernel is not "
                 "setup for this (yet)");

        // Initialize pointers.
        TData *diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadWrite>();

        auto *derivPtr0 = deriv.template GetPtr<MemSpace, ReadWrite>();
        auto *derivPtr1 = derivPtr0 + deriv.GetFieldSize();
        auto *derivPtr2 = derivPtr1 + deriv.GetFieldSize();

        auto *derivCoeffPtr0 =
            derivCoeff.template GetPtr<MemSpace, ReadWrite>();
        auto *derivCoeffPtr1 = derivCoeffPtr0 + derivCoeff.GetFieldSize();
        auto *derivCoeffPtr2 = derivCoeffPtr1 + derivCoeff.GetFieldSize();

        std::vector<TData *> derivPtr{derivPtr0, derivPtr1, derivPtr2};
        std::vector<TData *> derivCoeffPtr{derivCoeffPtr0, derivCoeffPtr1,
                                           derivCoeffPtr2};

        // Initialize index.
        size_t exp_idx = 0;

        for (const auto &block : deriv.GetBlocks())
        {
            const auto nElmts    = block.num_elements;
            const auto nElmtsPad = block.num_elmt_groups * block.width;

            // Determine shape and type of the element.
            const auto expPtr = this->m_expansionList->GetExp(exp_idx);
            auto nCoord       = expPtr->GetCoordim();
            auto nqTot        = expPtr->GetTotPoints();

            // Multiply by diffusion coefficient.
            for (size_t d = 0; d < nCoord; d++)
            {
                Vmath::Smul(nqTot * nElmts, diffCoeffPtr[d * nCoord],
                            derivPtr[0], 1, derivCoeffPtr[d], 1);

                for (size_t l = 1; l < nCoord; l++)
                {
                    Vmath::Svtvp(nqTot * nElmts, diffCoeffPtr[d * nCoord + l],
                                 derivPtr[l], 1, derivCoeffPtr[d], 1,
                                 derivCoeffPtr[d], 1);
                }
            }

            // Increment pointer and index for next element type.
            for (size_t d = 0; d < nCoord; d++)
            {
                derivPtr[d] += nqTot * nElmtsPad;
                derivCoeffPtr[d] += nqTot * nElmtsPad;
            }

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

private:
    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorPhysDeriv<TData>> m_PhysDerivOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    std::shared_ptr<OperatorIProductWRTDerivBase<TData>>
        m_IProductWRTDerivBaseOp;

    Field<TData, FieldState::Phys> m_bwd;
    Field<TData, FieldState::Phys> m_deriv;
    Field<TData, FieldState::Phys> m_derivCoeff;

    MemoryRegion<TData> m_diffCoeff;
};

} // namespace Nektar::Operators::detail
