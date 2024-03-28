///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseCUDA.cuh
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

#include "Operators/IProductWRTBase/IProductWRTBaseCUDA.cuh"
#include "Operators/IProductWRTDerivBase/IProductWRTDerivBaseCUDAKernels.cuh"
#include "Operators/MemoryRegionCUDA.hpp"
#include "Operators/OperatorHelper.hpp"
#include "Operators/OperatorIProductWRTDerivBase.hpp"

#define FLAG_QP false

namespace Nektar::Operators::detail
{

template <typename TData, bool DEFORMED = false>
void IProductWRTDerivBase1DKernel(
    const unsigned int gridSize, const unsigned int blockSize,
    const unsigned int nq0, const unsigned int nCoord,
    const unsigned int nElmts, const unsigned int nSize,
    const unsigned int dfSize, const TData *df, const TData *in, TData *out);

template <typename TData, bool DEFORMED = false>
void IProductWRTDerivBase2DKernel(
    const unsigned int gridSize, const unsigned int blockSize,
    LibUtilities::ShapeType shapetype, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nCoord,
    const unsigned int nElmts, const unsigned int nSize,
    const unsigned int dfSize, const TData *Z0, const TData *Z1,
    const TData *df, const TData *in, TData *out);

template <typename TData, bool DEFORMED = false>
void IProductWRTDerivBase3DKernel(
    const unsigned int gridSize, const unsigned int blockSize,
    LibUtilities::ShapeType shapetype, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nCoord,
    const unsigned int nElmts, const unsigned int nSize,
    const unsigned int dfSize, const TData *Z0, const TData *Z1,
    const TData *Z2, const TData *df, const TData *in, TData *out);

// IProductWRTDerivBase implementation
template <typename TData>
class OperatorIProductWRTDerivBaseImpl<TData, ImplCUDA>
    : public OperatorIProductWRTDerivBase<TData>
{
public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
        size_t nDim   = this->m_expansionList->GetShapeDimension();
        size_t nCoord = this->m_expansionList->GetCoordim(0);
        m_dfSize      = Operator<TData>::GetGeometricFactorSize();

        // Initialise jacobian.
        auto jac = Operator<TData>::SetJacobian(m_dfSize);
        cudaMalloc((void **)&m_jac, sizeof(TData) * m_dfSize);
        cudaMemcpy(m_jac, jac.get(), sizeof(TData) * m_dfSize,
                   cudaMemcpyHostToDevice);

        // Initialise derivative factor.
        auto derivFac = Operator<TData>::SetDerivativeFactor(m_dfSize);
        cudaMalloc((void **)&m_derivFac,
                   sizeof(TData) * nDim * nCoord * m_dfSize);
        for (size_t d = 0; d < nDim * nCoord; d++)
        {
            auto hostPtr   = derivFac[d].get();
            auto devicePtr = m_derivFac + d * m_dfSize;
            cudaMemcpy(devicePtr, hostPtr, sizeof(TData) * m_dfSize,
                       cudaMemcpyHostToDevice);
        }

        // Initialize basis.
        m_basis = GetBasisDataCUDA<TData>(expansionList);

        // Initialize basis derivative.
        m_dbasis = GetDeriveBasisDataCUDA<TData>(expansionList);

        // Initialize weight.
        m_weight = GetWeightDataCUDA<TData>(expansionList);

        // Initialize points.
        m_Z = GetPointDataCUDA<TData>(expansionList);

        // Initialize derivative matrix.
        m_D = GetDerivativeDataCUDA<TData>(expansionList);

        // Initialize workspace memory.
        auto nStorage = this->m_expansionList->GetTotPoints();
        cudaMalloc((void **)&m_tmp, sizeof(TData) * nStorage * nCoord);
    }

    ~OperatorIProductWRTDerivBaseImpl(void)
    {
        DeallocateDataCUDA<TData>(m_basis);
        DeallocateDataCUDA<TData>(m_dbasis);
        DeallocateDataCUDA<TData>(m_weight);
        DeallocateDataCUDA<TData>(m_Z);
        DeallocateDataCUDA<TData>(m_D);
        cudaFree(m_jac);
        cudaFree(m_derivFac);
        cudaFree(m_tmp);
        if (m_wsp != nullptr)
        {
            cudaFree(m_wsp);
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               bool APPEND = false) override
    {
        // Copy memory to GPU, if necessary and get raw pointers.
        auto *inptr  = in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *jacptr = m_jac;
        auto *dfptr  = m_derivFac;
        auto *tmpptr = m_tmp;
        auto nSize   = in.GetFieldSize();

        // Initialize index.
        size_t expIdx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Zero output.
        if (!APPEND)
        {
            size_t ndata = out.template GetStorage<MemoryRegionCUDA>().size();
            cudaMemset(outptr, 0, sizeof(TData) * ndata);
        }

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
             ++block_idx)
        {
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = out.GetBlocks()[block_idx].num_elements;
            auto nqTot        = expPtr->GetTotPoints();
            auto nmTot        = expPtr->GetNcoeffs();
            auto nDim         = expPtr->GetShapeDimension();
            auto nCoord       = expPtr->GetCoordim();
            auto shape        = expPtr->DetShapeType();
            auto deformed     = expPtr->GetMetricInfo()->GetGtype() ==
                            SpatialDomains::eDeformed;

            // Deterime CUDA grid size.
            m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);

            // Flag for collapsed coordinate correction.
            bool correct = expPtr->GetBasis(0)->GetBasisType() ==
                           LibUtilities::eModified_A;

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < nDim; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Function call to kernel functions.
            if (nDim == 1)
            {
                auto dbasis0 = m_dbasis[basisKeys][0];
                auto w0      = m_weight[basisKeys][0];
                auto D0      = m_D[basisKeys][0];
                auto nm0     = expPtr->GetBasisNumModes(0);
                auto nq0     = expPtr->GetNumPoints(0);
                if (deformed)
                {
                    IProductWRTDerivBase1DKernel<TData, true>(
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, nSize,
                        m_dfSize, dfptr, inptr, tmpptr);
                    IProductWRTBase1DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, nm0, nq0, nElmts, dbasis0, w0,
                        jacptr, tmpptr, outptr);
                }
                else
                {
                    IProductWRTDerivBase1DKernel<TData, false>(
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, nSize,
                        m_dfSize, dfptr, inptr, tmpptr);
                    IProductWRTBase1DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, nm0, nq0, nElmts, dbasis0, w0,
                        jacptr, tmpptr, outptr);
                }
            }
            else if (nDim == 2)
            {
                auto basis0  = m_basis[basisKeys][0];
                auto basis1  = m_basis[basisKeys][1];
                auto dbasis0 = m_dbasis[basisKeys][0];
                auto dbasis1 = m_dbasis[basisKeys][1];
                auto w0      = m_weight[basisKeys][0];
                auto w1      = m_weight[basisKeys][1];
                auto D0      = m_D[basisKeys][0];
                auto D1      = m_D[basisKeys][1];
                auto Z0      = m_Z[basisKeys][0];
                auto Z1      = m_Z[basisKeys][1];
                auto nm0     = expPtr->GetBasisNumModes(0);
                auto nm1     = expPtr->GetBasisNumModes(1);
                auto nq0     = expPtr->GetNumPoints(0);
                auto nq1     = expPtr->GetNumPoints(1);
                if constexpr (!FLAG_QP)
                {
                    size_t wspsize = 0;
                    if (shape == LibUtilities::Quad)
                    {
                        wspsize = nq1 * nElmts;
                    }
                    else if (shape == LibUtilities::Tri)
                    {
                        wspsize = nq0 * nElmts;
                    }

                    if (m_wspsize < wspsize)
                    {
                        if (m_wsp != nullptr)
                        {
                            cudaFree(m_wsp);
                        }

                        m_wspsize = wspsize;
                        cudaMalloc((void **)&m_wsp, sizeof(TData) * m_wspsize);
                    }
                }
                if (deformed)
                {
                    IProductWRTDerivBase2DKernel<TData, true>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nCoord,
                        nElmts, nSize, m_dfSize, Z0, Z1, dfptr, inptr, tmpptr);
                    IProductWRTBase2DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                        nElmts, correct, dbasis0, basis1, w0, w1, jacptr, m_wsp,
                        tmpptr, outptr);
                    IProductWRTBase2DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                        nElmts, correct, basis0, dbasis1, w0, w1, jacptr, m_wsp,
                        tmpptr + nSize, outptr);
                }
                else
                {
                    IProductWRTDerivBase2DKernel<TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nCoord,
                        nElmts, nSize, m_dfSize, Z0, Z1, dfptr, inptr, tmpptr);
                    IProductWRTBase2DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                        nElmts, correct, dbasis0, basis1, w0, w1, jacptr, m_wsp,
                        tmpptr, outptr);
                    IProductWRTBase2DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                        nElmts, correct, basis0, dbasis1, w0, w1, jacptr, m_wsp,
                        tmpptr + nSize, outptr);
                }
            }
            else if (nDim == 3)
            {
                auto basis0  = m_basis[basisKeys][0];
                auto basis1  = m_basis[basisKeys][1];
                auto basis2  = m_basis[basisKeys][2];
                auto dbasis0 = m_dbasis[basisKeys][0];
                auto dbasis1 = m_dbasis[basisKeys][1];
                auto dbasis2 = m_dbasis[basisKeys][2];
                auto w0      = m_weight[basisKeys][0];
                auto w1      = m_weight[basisKeys][1];
                auto w2      = m_weight[basisKeys][2];
                auto D0      = m_D[basisKeys][0];
                auto D1      = m_D[basisKeys][1];
                auto D2      = m_D[basisKeys][2];
                auto Z0      = m_Z[basisKeys][0];
                auto Z1      = m_Z[basisKeys][1];
                auto Z2      = m_Z[basisKeys][2];
                auto nm0     = expPtr->GetBasisNumModes(0);
                auto nm1     = expPtr->GetBasisNumModes(1);
                auto nm2     = expPtr->GetBasisNumModes(2);
                auto nq0     = expPtr->GetNumPoints(0);
                auto nq1     = expPtr->GetNumPoints(1);
                auto nq2     = expPtr->GetNumPoints(2);
                if constexpr (!FLAG_QP)
                {
                    size_t wspsize = 0;
                    if (shape == LibUtilities::Hex)
                    {
                        wspsize = (nq2 * nq1 + nq2) * nElmts;
                    }
                    else if (shape == LibUtilities::Tet)
                    {
                        wspsize = (nq2 * nq1 + nq2 + nm2) * nElmts;
                    }
                    else if (shape == LibUtilities::Prism)
                    {
                        wspsize = (nq2 * nq1 + nq2 + nm1) * nElmts;
                    }
                    else if (shape == LibUtilities::Pyr)
                    {
                        wspsize = (nq2 * nq1 + nq2) * nElmts;
                    }

                    if (m_wspsize < wspsize)
                    {
                        if (m_wsp != nullptr)
                        {
                            cudaFree(m_wsp);
                        }

                        m_wspsize = wspsize;
                        cudaMalloc((void **)&m_wsp, sizeof(TData) * m_wspsize);
                    }
                }
                if (deformed)
                {
                    IProductWRTDerivBase3DKernel<TData, true>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nCoord,
                        nElmts, nSize, m_dfSize, Z0, Z1, Z2, dfptr, inptr,
                        tmpptr);
                    IProductWRTBase3DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, dbasis0, basis1, basis2, w0, w1,
                        w2, jacptr, m_wsp, tmpptr, outptr);
                    IProductWRTBase3DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, basis0, dbasis1, basis2, w0, w1,
                        w2, jacptr, m_wsp, tmpptr + nSize, outptr);
                    IProductWRTBase3DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, basis0, basis1, dbasis2, w0, w1,
                        w2, jacptr, m_wsp, tmpptr + 2 * nSize, outptr);
                }
                else
                {
                    IProductWRTDerivBase3DKernel<TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nCoord,
                        nElmts, nSize, m_dfSize, Z0, Z1, Z2, dfptr, inptr,
                        tmpptr);
                    IProductWRTBase3DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, dbasis0, basis1, basis2, w0, w1,
                        w2, jacptr, m_wsp, tmpptr, outptr);
                    IProductWRTBase3DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, basis0, dbasis1, basis2, w0, w1,
                        w2, jacptr, m_wsp, tmpptr + nSize, outptr);
                    IProductWRTBase3DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, basis0, basis1, dbasis2, w0, w1,
                        w2, jacptr, m_wsp, tmpptr + 2 * nSize, outptr);
                }
            }

            // Increment pointer and index for next element type.
            jacptr += deformed ? nqTot * nElmts : nElmts;
            dfptr += deformed ? nqTot * nElmts : nElmts;
            inptr += nqTot * nElmts;
            tmpptr += nqTot * nElmts;
            outptr += nmTot * nElmts;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        MultiRegions::ExpListSharedPtr expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTDerivBaseImpl<TData, ImplCUDA>>(expansionList);
    }

    static std::string className;

private:
    TData *m_wsp      = nullptr;
    TData *m_tmp      = nullptr;
    TData *m_derivFac = nullptr;
    TData *m_jac      = nullptr;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_basis;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>>
        m_dbasis;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>>
        m_weight;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_D;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_Z;
    size_t m_dfSize;
    size_t m_wspsize   = 0;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

template <typename TData, bool DEFORMED>
void IProductWRTDerivBase1DKernel(
    const unsigned int gridSize, const unsigned int blockSize,
    const unsigned int nq0, const unsigned int nCoord,
    const unsigned int nElmts, const unsigned int nSize,
    const unsigned int dfSize, const TData *df, const TData *in, TData *out)
{
    if constexpr (!FLAG_QP)
    {
        IProductWRTDerivBase1DKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
            nq0, nCoord, nElmts, nSize, dfSize, df, in, out);
    }
    else
    {
        IProductWRTDerivBase1DKernel_QP<TData, DEFORMED>
            <<<gridSize, dim3(32)>>>(nq0, nCoord, nElmts, nSize, dfSize, df, in,
                                     out);
    }
}

template <typename TData, bool DEFORMED>
void IProductWRTDerivBase2DKernel(
    const unsigned int gridSize, const unsigned int blockSize,
    LibUtilities::ShapeType shapetype, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nCoord,
    const unsigned int nElmts, const unsigned int nSize,
    const unsigned int dfSize, const TData *Z0, const TData *Z1,
    const TData *df, const TData *in, TData *out)
{
    if (shapetype == LibUtilities::Quad)
    {
        if constexpr (!FLAG_QP)
        {
            IProductWRTDerivBase2DKernel<TData, LibUtilities::Quad, DEFORMED>
                <<<gridSize, blockSize>>>(nq0, nq1, nCoord, nElmts, nSize,
                                          dfSize, nullptr, nullptr, df, in,
                                          out);
        }
        else
        {
            IProductWRTDerivBase2DKernel_QP<TData, LibUtilities::Quad, DEFORMED>
                <<<gridSize, dim3(8, 8)>>>(nq0, nq1, nCoord, nElmts, nSize,
                                           dfSize, nullptr, nullptr, df, in,
                                           out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        if constexpr (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1);
            IProductWRTDerivBase2DKernel<TData, LibUtilities::Tri, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nCoord, nElmts,
                                                   nSize, dfSize, Z0, Z1, df,
                                                   in, out);
        }
        else
        {
            IProductWRTDerivBase2DKernel_QP<TData, LibUtilities::Tri, DEFORMED>
                <<<gridSize, dim3(8, 8)>>>(nq0, nq1, nCoord, nElmts, nSize,
                                           dfSize, Z0, Z1, df, in, out);
        }
    }
}

template <typename TData, bool DEFORMED>
void IProductWRTDerivBase3DKernel(
    const unsigned int gridSize, const unsigned int blockSize,
    LibUtilities::ShapeType shapetype, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nCoord,
    const unsigned int nElmts, const unsigned int nSize,
    const unsigned int dfSize, const TData *Z0, const TData *Z1,
    const TData *Z2, const TData *df, const TData *in, TData *out)
{
    if (shapetype == LibUtilities::Hex)
    {
        if constexpr (!FLAG_QP)
        {
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Hex, DEFORMED>
                <<<gridSize, blockSize>>>(nq0, nq1, nq2, nCoord, nElmts, nSize,
                                          dfSize, nullptr, nullptr, nullptr, df,
                                          in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Hex, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nCoord, nElmts,
                                              nSize, dfSize, nullptr, nullptr,
                                              nullptr, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        if constexpr (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + 2 * nq1 + nq2);
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Tet, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nq2, nCoord,
                                                   nElmts, nSize, dfSize, Z0,
                                                   Z1, Z2, df, in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Tet, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nCoord, nElmts,
                                              nSize, dfSize, Z0, Z1, Z2, df, in,
                                              out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        if constexpr (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq2);
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Prism, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nq2, nCoord,
                                                   nElmts, nSize, dfSize, Z0,
                                                   nullptr, Z2, df, in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Prism,
                                            DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nCoord, nElmts,
                                              nSize, dfSize, Z0, nullptr, Z2,
                                              df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        if constexpr (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Pyr, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nq2, nCoord,
                                                   nElmts, nSize, dfSize, Z0,
                                                   Z1, Z2, df, in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Pyr, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nCoord, nElmts,
                                              nSize, dfSize, Z0, Z1, Z2, df, in,
                                              out);
        }
    }
}

} // namespace Nektar::Operators::detail
