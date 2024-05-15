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
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)
#if defined(NEKTAR_ENABLE_CUDA)
              || (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
                  std::is_same<Implementation, Operators::SumFac>::value)
#endif
              >::type>
class OperatorHelmholtzImpl
    : public OperatorHelmholtzImplBase<ExecSpace, Implementation, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtzImplBase<ExecSpace, Implementation, TData>(
              expansionList)
    {
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
        DiffusionCoeff(this->m_deriv);

        // Step 5: Inner product
        this->m_IProductWRTDerivBaseOp->apply(this->m_deriv, out, true);
    }

    void DiffusionCoeff(Field<TData, FieldState::Phys> &deriv)
    {
        // Initialize pointers.
        TData *diffCoeffptr = this->m_diffCoeff.template GetPtr<MemSpace>();

        auto *derivptr0 = deriv.template GetPtr<MemSpace>();
        auto *derivptr1 = derivptr0 + deriv.GetFieldSize();
        auto *derivptr2 = derivptr1 + deriv.GetFieldSize();

        // Initialize index.
        size_t expIdx = 0;

        for (auto const &block : deriv.GetBlocks())
        {
            // Determine shape and type of the element.
            auto nElmts = block.num_elements;

            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nCoord       = expPtr->GetCoordim();
            auto nqTot        = expPtr->GetTotPoints();

            // Deterime CUDA grid size.
            m_gridSize = GetCUDAGridSize(nqTot * nElmts, m_blockSize);

            // Multiply by diffusion coefficient.
            if (nCoord == 1)
            {
                DiffusionCoeff1DKernel<ExecSpace, TData>(
                    m_gridSize, m_blockSize, nqTot * nElmts, diffCoeffptr,
                    derivptr0);

                derivptr0 += nqTot * nElmts;
            }
            else if (nCoord == 2)
            {
                DiffusionCoeff2DKernel<ExecSpace, TData>(
                    m_gridSize, m_blockSize, nqTot * nElmts, diffCoeffptr,
                    derivptr0, derivptr1);

                derivptr0 += nqTot * nElmts;
                derivptr1 += nqTot * nElmts;
            }
            else
            {
                DiffusionCoeff3DKernel<ExecSpace, TData>(
                    m_gridSize, m_blockSize, nqTot * nElmts, diffCoeffptr,
                    derivptr0, derivptr1, derivptr2);

                derivptr0 += nqTot * nElmts;
                derivptr1 += nqTot * nElmts;
                derivptr2 += nqTot * nElmts;
            }

            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorHelmholtzImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
