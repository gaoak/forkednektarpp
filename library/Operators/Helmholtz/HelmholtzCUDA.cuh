///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzCUDA.cuh
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

#include "Operators/BwdTrans/BwdTransCUDA.cuh"
#include "Operators/Helmholtz/HelmholtzCUDAKernels.cuh"
#include "Operators/IProductWRTBase/IProductWRTBaseCUDA.cuh"
#include "Operators/IProductWRTDerivBase/IProductWRTDerivBaseCUDA.cuh"
#include "Operators/OperatorHelmholtz.hpp"
#include "Operators/PhysDeriv/PhysDerivCUDA.cuh"

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorHelmholtzImpl<TData, ImplCUDA> : public OperatorHelmholtz<TData>
{
public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtz<TData>(expansionList),
          m_bwd(
              Field<TData, FieldState::Phys>::template create<MemoryRegionCUDA>(
                  GetBlockAttributes(FieldState::Phys, expansionList))),
          m_deriv(
              Field<TData, FieldState::Phys>::template create<MemoryRegionCUDA>(
                  GetBlockAttributes(FieldState::Phys, expansionList),
                  expansionList->GetCoordim(0)))
    {
        auto nCoord = this->m_expansionList->GetCoordim(0);

        m_BwdTransOp  = BwdTrans<>::create(this->m_expansionList, "CUDA");
        m_PhysDerivOp = PhysDeriv<>::create(this->m_expansionList, "CUDA");
        m_IProductWRTBaseOp =
            IProductWRTBase<>::create(this->m_expansionList, "CUDA");
        m_IProductWRTDerivBaseOp =
            IProductWRTDerivBase<>::create(this->m_expansionList, "CUDA");

        Array<OneD, TData> diffCoeff(nCoord * nCoord, 0.0);
        for (size_t d = 0; d < nCoord; d++)
        {
            diffCoeff[d * nCoord + d] = 1.0; // Temporary solution
        }
        cudaMalloc((void **)&m_diffCoeff, sizeof(TData) * nCoord * nCoord);
        cudaMemcpy(m_diffCoeff, diffCoeff.get(),
                   sizeof(TData) * nCoord * nCoord, cudaMemcpyHostToDevice);
    }

    ~OperatorHelmholtzImpl(void)
    {
        cudaFree(m_diffCoeff);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Step 1: BwdTrans
        m_BwdTransOp->apply(in, m_bwd);

        // Step 2: PhysDeriv
        m_PhysDerivOp->apply(m_bwd, m_deriv);

        // Step 3: Inner product for mass matrix operation
        m_IProductWRTBaseOp->apply(m_bwd, out, this->m_lambda);

        // Step 4: Multiply by diffusion coefficient
        DiffusionCoeff(m_deriv);

        // Step 5: Inner product
        m_IProductWRTDerivBaseOp->apply(m_deriv, out, true);
    }

    void DiffusionCoeff(Field<TData, FieldState::Phys> &deriv)
    {
        // Initialize pointers.
        auto *derivptr0 =
            deriv.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *derivptr1 = derivptr0 + deriv.GetFieldSize();
        auto *derivptr2 = derivptr1 + deriv.GetFieldSize();

        // Initialize index.
        size_t expIdx = 0;

        for (size_t block_idx = 0; block_idx < deriv.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = deriv.GetBlocks()[block_idx].num_elements;
            auto nCoord       = expPtr->GetCoordim();
            auto nqTot        = expPtr->GetTotPoints();

            // Deterime CUDA grid size.
            m_gridSize = GetCUDAGridSize(nqTot * nElmts, m_blockSize);

            // Multiply by diffusion coefficient.
            if (nCoord == 1)
            {
                DiffusionCoeff1DKernel<<<m_gridSize, m_blockSize>>>(
                    nqTot * nElmts, m_diffCoeff, derivptr0);
                derivptr0 += nqTot * nElmts;
            }
            else if (nCoord == 2)
            {
                DiffusionCoeff2DKernel<<<m_gridSize, m_blockSize>>>(
                    nqTot * nElmts, m_diffCoeff, derivptr0, derivptr1);
                derivptr0 += nqTot * nElmts;
                derivptr1 += nqTot * nElmts;
            }
            else
            {
                DiffusionCoeff3DKernel<<<m_gridSize, m_blockSize>>>(
                    nqTot * nElmts, m_diffCoeff, derivptr0, derivptr1,
                    derivptr2);
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
        return std::make_unique<OperatorHelmholtzImpl<TData, ImplCUDA>>(
            expansionList);
    }

    static std::string className;

private:
    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorPhysDeriv<TData>> m_PhysDerivOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    std::shared_ptr<OperatorIProductWRTDerivBase<TData>>
        m_IProductWRTDerivBaseOp;
    Field<TData, FieldState::Phys> m_bwd;
    Field<TData, FieldState::Phys> m_deriv;
    TData *m_diffCoeff;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
