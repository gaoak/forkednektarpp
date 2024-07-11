///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzImplShared.hpp
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

#include "Operators/Helmholtz/HelmholtzImplBase.hpp"

#include "Operators/Helmholtz/HelmholtzCUDASumFacKernels.cuh"
#include "Operators/Helmholtz/HelmholtzKokkosStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
               std::is_same<Implementation, Operators::StdMat>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)>::type>
class OperatorHelmholtzImpl
    : public OperatorHelmholtzImplBase<ExecSpace, Implementation, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtzImplBase<ExecSpace, Implementation, TData>(
              expansionList),
          m_bwd(Field<TData, FieldState::Phys>::template create<MemSpace>(
              "Helmholtz bwd",
              GetBlockAttributes(FieldState::Phys, expansionList))),

          m_deriv(Field<TData, FieldState::Phys>::template create<MemSpace>(
              "Helmholtz deriv",
              GetBlockAttributes(FieldState::Phys, expansionList),
              expansionList->GetCoordim(0)))
    {
        if constexpr ((std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
                       std::is_same<Implementation, Operators::StdMat>::value))
        {
            m_derivCoeff =
                Field<TData, FieldState::Phys>::template create<MemSpace>(
                    GetBlockAttributes(FieldState::Phys, expansionList),
                    expansionList->GetCoordim(0));
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

        // CUDA and Kokkos "in-place" version
        if constexpr ((std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
                       std::is_same<Implementation,
                                    Operators::SumFac>::value) ||
                      (std::is_same<ExecSpace,
                                    Kokkos::DefaultExecutionSpace>::value &&
                       std::is_same<Implementation, Operators::StdMat>::value))
        {
            // Step 4: Multiply by diffusion coefficient
            DiffusionCoeff(this->m_deriv);

            // Step 5: Inner product
            this->m_IProductWRTDerivBaseOp->apply(this->m_deriv, out, true);
        }
        // Serial version
        else if constexpr ((std::is_same<ExecSpace,
                                         NektarSpaces::Serial>::value &&
                            std::is_same<Implementation,
                                         Operators::StdMat>::value))
        {
            // Step 4: Multiply by diffusion coefficient
            DiffusionCoeff(this->m_deriv, m_derivCoeff);

            // Step 5: Inner product
            this->m_IProductWRTDerivBaseOp->apply(m_derivCoeff, out, true);
        }
    }

    // CUDA and Kokkos "in-place" version
    void DiffusionCoeff(Field<TData, FieldState::Phys> &deriv)
    {
        // Initialize pointers.
        TData *diffCoeffPtr = this->m_diffCoeff.template GetPtr<MemSpace>();

        auto *derivPtr0 = deriv.template GetPtr<MemSpace>();
        auto *derivPtr1 = derivPtr0 + deriv.GetFieldSize();
        auto *derivPtr2 = derivPtr1 + deriv.GetFieldSize();

        // Initialize index.
        size_t exp_idx = 0;

        for (auto const &block : deriv.GetBlocks())
        {
            // Determine shape and type of the element.
            auto nElmts = block.num_elements;

            auto const expPtr = this->m_expansionList->GetExp(exp_idx);
            auto nCoord       = expPtr->GetCoordim();
            auto nqTot        = expPtr->GetTotPoints();

            // Multiply by diffusion coefficient.
            if (nCoord == 1)
            {
                DiffusionCoeff1DKernel<ExecSpace, TData>(
                    nqTot * nElmts, diffCoeffPtr, derivPtr0);

                derivPtr0 += nqTot * nElmts;
            }
            else if (nCoord == 2)
            {
                DiffusionCoeff2DKernel<ExecSpace, TData>(
                    nqTot * nElmts, diffCoeffPtr, derivPtr0, derivPtr1);

                derivPtr0 += nqTot * nElmts;
                derivPtr1 += nqTot * nElmts;
            }
            else
            {
                DiffusionCoeff3DKernel<ExecSpace, TData>(
                    nqTot * nElmts, diffCoeffPtr, derivPtr0, derivPtr1,
                    derivPtr2);

                derivPtr0 += nqTot * nElmts;
                derivPtr1 += nqTot * nElmts;
                derivPtr2 += nqTot * nElmts;
            }

            exp_idx += nElmts;
        }
    }

    // Serial version
    void DiffusionCoeff(Field<TData, FieldState::Phys> &deriv,
                        Field<TData, FieldState::Phys> &derivCoeff)
    {
        // Initialize pointers.
        TData *diffCoeffPtr = this->m_diffCoeff.template GetPtr<MemSpace>();

        auto *derivPtr0 = deriv.template GetPtr<MemSpace>();
        auto *derivPtr1 = derivPtr0 + deriv.GetFieldSize();
        auto *derivPtr2 = derivPtr1 + deriv.GetFieldSize();

        auto *derivCoeffPtr0 = derivCoeff.template GetPtr<MemSpace>();
        auto *derivCoeffPtr1 = derivCoeffPtr0 + derivCoeff.GetFieldSize();
        auto *derivCoeffPtr2 = derivCoeffPtr1 + derivCoeff.GetFieldSize();

        std::vector<TData *> derivPtr{derivPtr0, derivPtr1, derivPtr2};
        std::vector<TData *> derivCoeffPtr{derivCoeffPtr0, derivCoeffPtr1,
                                           derivCoeffPtr2};

        // Initialize index.
        size_t exp_idx = 0;

        for (auto const &block : deriv.GetBlocks())
        {
            auto nElmts = block.num_elements;

            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(exp_idx);
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
                derivPtr[d] += nqTot * nElmts;
                derivCoeffPtr[d] += nqTot * nElmts;
            }

            exp_idx += nElmts;
        }
    }

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
    Field<TData, FieldState::Phys> m_derivCoeff; // Serial Only
};

} // namespace Nektar::Operators::detail
