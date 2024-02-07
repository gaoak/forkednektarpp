///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivCUDA.hpp
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

#include "MemoryRegionCUDA.hpp"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorPhysDeriv.hpp"
#include "Operators/PhysDeriv/PhysDerivCUDAKernels.cuh"

#define FLAG_QP false

namespace Nektar::Operators::detail
{

template <typename TData, bool DEFORMED = false>
void PhysDeriv1DKernel(const unsigned int gridSize,
                       const unsigned int blockSize, const unsigned int nq0,
                       const unsigned int nCoord, const unsigned int nElmts,
                       const unsigned int nSize, const unsigned int dfSize,
                       const TData *D0, const TData *df, const TData *in,
                       TData *out);

template <typename TData, bool DEFORMED = false>
void PhysDeriv2DKernel(const unsigned int gridSize,
                       const unsigned int blockSize,
                       LibUtilities::ShapeType shapetype,
                       const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nCoord, const unsigned int nElmts,
                       const unsigned int nSize, const unsigned int dfSize,
                       const TData *D0, const TData *D1, const TData *Z0,
                       const TData *Z1, const TData *df, const TData *in,
                       TData *out);

template <typename TData, bool DEFORMED = false>
void PhysDeriv3DKernel(const unsigned int gridSize,
                       const unsigned int blockSize,
                       LibUtilities::ShapeType shapetype,
                       const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nq2, const unsigned int nElmts,
                       const unsigned int nSize, const unsigned int dfSize,
                       const TData *D0, const TData *D1, const TData *D2,
                       const TData *Z0, const TData *Z1, const TData *Z2,
                       const TData *df, const TData *in, TData *out);

// CUDA implementation
template <typename TData>
class OperatorPhysDerivImpl<TData, ImplCUDA> : public OperatorPhysDeriv<TData>
{
public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        size_t nDim      = this->m_expansionList->GetShapeDimension();
        size_t nCoord    = this->m_expansionList->GetCoordim(0);

        // Initialise derivative factor.
        m_dfSize      = Operator<TData>::GetGeometricFactorSize();
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

        // Initialize points.
        m_Z = GetPointDataCUDA<TData>(expansionList);

        // Initialize derivative matrix.
        m_D = GetDerivativeDataCUDA<TData>(expansionList);
    }

    ~OperatorPhysDerivImpl(void)
    {
        DeallocateDataCUDA<TData>(m_Z);
        DeallocateDataCUDA<TData>(m_D);
        cudaFree(m_derivFac);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Initialize pointers.
        auto const *inptr =
            in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto dfptr   = m_derivFac;
        size_t nSize = out.GetFieldSize();

        // Initialize index.
        size_t expIdx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = in.GetBlocks()[block_idx].num_elements;
            auto nPadElmts    = in.GetBlocks()[block_idx].num_padding_elements;
            auto nqTot        = expPtr->GetTotPoints();
            auto nCoord       = expPtr->GetCoordim();
            auto shape        = expPtr->DetShapeType();
            auto deformed     = expPtr->GetMetricInfo()->GetGtype() ==
                            SpatialDomains::eDeformed;

            // Determine CUDA grid size.
            m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < expPtr->GetShapeDimension(); d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Function call to kernel functions.
            if (expPtr->GetShapeDimension() == 1)
            {
                auto D0  = m_D[basisKeys][0];
                auto nq0 = expPtr->GetNumPoints(0);
                if (deformed)
                {
                    PhysDeriv1DKernel<TData, true>(
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, nSize,
                        m_dfSize, D0, dfptr, inptr, outptr);
                }
                else
                {
                    PhysDeriv1DKernel<TData, false>(
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, nSize,
                        m_dfSize, D0, dfptr, inptr, outptr);
                }
            }
            else if (expPtr->GetShapeDimension() == 2)
            {
                auto D0  = m_D[basisKeys][0];
                auto D1  = m_D[basisKeys][1];
                auto Z0  = m_Z[basisKeys][0];
                auto Z1  = m_Z[basisKeys][1];
                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);
                if (deformed)
                {
                    PhysDeriv2DKernel<TData, true>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nCoord,
                        nElmts, nSize, m_dfSize, D0, D1, Z0, Z1, dfptr, inptr,
                        outptr);
                }
                else
                {
                    PhysDeriv2DKernel<TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nCoord,
                        nElmts, nSize, m_dfSize, D0, D1, Z0, Z1, dfptr, inptr,
                        outptr);
                }
            }
            else if (expPtr->GetShapeDimension() == 3)
            {
                auto D0  = m_D[basisKeys][0];
                auto D1  = m_D[basisKeys][1];
                auto D2  = m_D[basisKeys][2];
                auto Z0  = m_Z[basisKeys][0];
                auto Z1  = m_Z[basisKeys][1];
                auto Z2  = m_Z[basisKeys][2];
                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);
                auto nq2 = expPtr->GetNumPoints(2);
                if (deformed)
                {
                    PhysDeriv3DKernel<TData, true>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nElmts,
                        nSize, m_dfSize, D0, D1, D2, Z0, Z1, Z2, dfptr, inptr,
                        outptr);
                }
                else
                {
                    PhysDeriv3DKernel<TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nElmts,
                        nSize, m_dfSize, D0, D1, D2, Z0, Z1, Z2, dfptr, inptr,
                        outptr);
                }
            }

            // Increment pointer and index for next element type.
            dfptr += deformed ? nqTot * nElmts : nElmts;
            outptr += (nPadElmts + nElmts) * nqTot;
            inptr += (nPadElmts + nElmts) * nqTot;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        MultiRegions::ExpListSharedPtr expansionList)
    {
        return std::make_unique<OperatorPhysDerivImpl<TData, ImplCUDA>>(
            expansionList);
    }

    static std::string className;

private:
    TData *m_derivFac;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_D;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_Z;
    size_t m_dfSize;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

template <typename TData, bool DEFORMED>
void PhysDeriv1DKernel(const unsigned int gridSize,
                       const unsigned int blockSize, const unsigned int nq0,
                       const unsigned int nCoord, const unsigned int nElmts,
                       const unsigned int nSize, const unsigned int dfSize,
                       const TData *D0, const TData *df, const TData *in,
                       TData *out)
{
    // Compute tensorial derivative.
    if (!FLAG_QP)
    {
        unsigned int nshared = sizeof(TData) * (nq0 * nq0);
        PhysDerivTensor1DKernel<TData>
            <<<gridSize, blockSize, nshared>>>(nq0, nElmts, D0, in, out);
    }
    else
    {
        PhysDerivTensor1DKernel_QP<TData>
            <<<gridSize, dim3(32)>>>(nq0, nElmts, D0, in, out);
    }

    // Compute physical derivative.
    if (!FLAG_QP)
    {
        PhysDeriv1DKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
            nq0, nCoord, nElmts, nSize, dfSize, df, out);
    }
    else
    {
        PhysDeriv1DKernel_QP<TData, DEFORMED><<<gridSize, dim3(32)>>>(
            nq0, nCoord, nElmts, nSize, dfSize, df, out);
    }
}

template <typename TData, bool DEFORMED>
void PhysDeriv2DKernel(const unsigned int gridSize,
                       const unsigned int blockSize,
                       LibUtilities::ShapeType shapetype,
                       const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nCoord, const unsigned int nElmts,
                       const unsigned int nSize, const unsigned int dfSize,
                       const TData *D0, const TData *D1, const TData *Z0,
                       const TData *Z1, const TData *df, const TData *in,
                       TData *out)
{
    // Compute tensorial derivative.
    if (!FLAG_QP)
    {
        unsigned int nshared = sizeof(TData) * (nq0 * nq0 + nq1 * nq1);
        PhysDerivTensor2DKernel<TData><<<gridSize, blockSize, nshared>>>(
            nq0, nq1, nElmts, nSize, D0, D1, in, out);
    }
    else
    {
        PhysDerivTensor2DKernel_QP<TData><<<gridSize, dim3(8, 8)>>>(
            nq0, nq1, nElmts, nSize, D0, D1, in, out);
    }

    // Compute physical derivative.
    if (shapetype == LibUtilities::Quad)
    {
        if (!FLAG_QP)
        {
            PhysDeriv2DKernel<TData, LibUtilities::Quad, DEFORMED>
                <<<gridSize, blockSize>>>(nq0, nq1, nCoord, nElmts, nSize,
                                          dfSize, nullptr, nullptr, df, out);
        }
        else
        {
            PhysDeriv2DKernel_QP<TData, LibUtilities::Quad, DEFORMED>
                <<<gridSize, dim3(8, 8)>>>(nq0, nq1, nCoord, nElmts, nSize,
                                           dfSize, nullptr, nullptr, df, out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        if (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1);
            PhysDeriv2DKernel<TData, LibUtilities::Tri, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nq0, nq1, nCoord, nElmts, nSize, dfSize, Z0, Z1, df, out);
        }
        else
        {
            PhysDeriv2DKernel_QP<TData, LibUtilities::Tri, DEFORMED>
                <<<gridSize, dim3(8, 8)>>>(nq0, nq1, nCoord, nElmts, nSize,
                                           dfSize, Z0, Z1, df, out);
        }
    }
}

template <typename TData, bool DEFORMED>
void PhysDeriv3DKernel(const unsigned int gridSize,
                       const unsigned int blockSize,
                       LibUtilities::ShapeType shapetype,
                       const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nq2, const unsigned int nElmts,
                       const unsigned int nSize, const unsigned int dfSize,
                       const TData *D0, const TData *D1, const TData *D2,
                       const TData *Z0, const TData *Z1, const TData *Z2,
                       const TData *df, const TData *in, TData *out)
{
    // Compute tensorial derivative.
    if (!FLAG_QP)
    {
        unsigned int nshared =
            sizeof(TData) * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2);
        PhysDerivTensor3DKernel<TData><<<gridSize, blockSize, nshared>>>(
            nq0, nq1, nq2, nElmts, nSize, D0, D1, D2, in, out);
    }
    else
    {
        PhysDerivTensor3DKernel_QP<TData><<<gridSize, dim3(4, 4, 4)>>>(
            nq0, nq1, nq2, nElmts, nSize, D0, D1, D2, in, out);
    }

    // Compute physical derivative.
    if (shapetype == LibUtilities::Hex)
    {
        if (!FLAG_QP)
        {
            PhysDeriv3DKernel<TData, LibUtilities::Hex, DEFORMED>
                <<<gridSize, blockSize>>>(nq0, nq1, nq2, nElmts, nSize, dfSize,
                                          nullptr, nullptr, nullptr, df, out);
        }
        else
        {
            PhysDeriv3DKernel_QP<TData, LibUtilities::Hex, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nElmts, nSize,
                                              dfSize, nullptr, nullptr, nullptr,
                                              df, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        if (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + 2 * nq1 + nq2);
            PhysDeriv3DKernel<TData, LibUtilities::Tet, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nq2, nElmts, nSize,
                                                   dfSize, Z0, Z1, Z2, df, out);
        }
        else
        {
            PhysDeriv3DKernel_QP<TData, LibUtilities::Tet, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nElmts, nSize,
                                              dfSize, Z0, Z1, Z2, df, out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        if (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq2);
            PhysDeriv3DKernel<TData, LibUtilities::Prism, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nq2, nElmts, nSize,
                                                   dfSize, Z0, nullptr, Z2, df,
                                                   out);
        }
        else
        {
            PhysDeriv3DKernel_QP<TData, LibUtilities::Prism, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nElmts, nSize,
                                              dfSize, Z0, nullptr, Z2, df, out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        if (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            PhysDeriv3DKernel<TData, LibUtilities::Pyr, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nq0, nq1, nq2, nElmts, nSize,
                                                   dfSize, Z0, Z1, Z2, df, out);
        }
        else
        {
            PhysDeriv3DKernel_QP<TData, LibUtilities::Pyr, DEFORMED>
                <<<gridSize, dim3(4, 4, 4)>>>(nq0, nq1, nq2, nElmts, nSize,
                                              dfSize, Z0, Z1, Z2, df, out);
        }
    }
}

} // namespace Nektar::Operators::detail
