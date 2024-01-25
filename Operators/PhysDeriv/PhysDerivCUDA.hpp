# pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorPhysDeriv.hpp"
#include "Operators/PhysDeriv/PhysDerivCUDAKernels.cuh"

namespace Nektar::Operators::detail
{

template <typename TData, bool DEFORMED = false>
void PhysDeriv1DKernel(const size_t gridSize, const size_t blockSize,
                       const size_t nq0, const size_t nCoord,
                       const size_t nElmts, const TData *D0,
                       const size_t dfSize, TData *df, const TData *in,
                       TData *out0, TData *out1, TData *out2);

template <typename TData, bool DEFORMED = false>
void PhysDeriv2DKernel(const size_t gridSize, const size_t blockSize,
                       LibUtilities::ShapeType shapetype, const size_t nq0,
                       const size_t nq1, const size_t nCoord,
                       const size_t nElmts, const TData *D0, const TData *D1,
                       const TData *Z0, const TData *Z1, const size_t dfSize,
                       TData *df, const TData *in, TData *out0, TData *out1,
                       TData *out2);

template <typename TData, bool DEFORMED = false>
void PhysDeriv3DKernel(const size_t gridSize, const size_t blockSize,
                       LibUtilities::ShapeType shapetype, const size_t nq0,
                       const size_t nq1, const size_t nq2, const size_t nCoord,
                       const size_t nElmts, const TData *D0, const TData *D1,
                       const TData *D2, const TData *Z0, const TData *Z1,
                       const TData *Z2, const size_t dfSize, TData *df,
                       const TData *in, TData *out0, TData *out1, TData *out2);

// Matrix-free implementation
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
        auto *outptr0 = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *outptr1 = outptr0 + out.GetFieldSize();
        auto *outptr2 = outptr1 + out.GetFieldSize();
        auto dfptr    = m_derivFac;

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
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, D0,
                        m_dfSize, dfptr, inptr, outptr0, outptr1, outptr2);
                }
                else
                {
                    PhysDeriv1DKernel<TData, false>(
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, D0,
                        m_dfSize, dfptr, inptr, outptr0, outptr1, outptr2);
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
                        nElmts, D0, D1, Z0, Z1, m_dfSize, dfptr, inptr, outptr0,
                        outptr1, outptr2);
                }
                else
                {
                    PhysDeriv2DKernel<TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nCoord,
                        nElmts, D0, D1, Z0, Z1, m_dfSize, dfptr, inptr, outptr0,
                        outptr1, outptr2);
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
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nCoord,
                        nElmts, D0, D1, D2, Z0, Z1, Z2, m_dfSize, dfptr, inptr,
                        outptr0, outptr1, outptr2);
                }
                else
                {
                    PhysDeriv3DKernel<TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nCoord,
                        nElmts, D0, D1, D2, Z0, Z1, Z2, m_dfSize, dfptr, inptr,
                        outptr0, outptr1, outptr2);
                }
            }

            // Increment pointer and index for next element type.
            dfptr += deformed ? nqTot * nElmts : nElmts;
            outptr0 += (nPadElmts + nElmts) * nqTot;
            outptr1 += (nPadElmts + nElmts) * nqTot;
            outptr2 += (nPadElmts + nElmts) * nqTot;
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
    size_t m_blockSize = 32;
    size_t m_gridSize;
};

template <typename TData, bool DEFORMED>
void PhysDeriv1DKernel(const size_t gridSize, const size_t blockSize,
                       const size_t nq0, const size_t nCoord,
                       const size_t nElmts, const TData *D0,
                       const size_t dfSize, TData *df, const TData *in,
                       TData *out0, TData *out1, TData *out2)
{
    // Compute tensorial derivative.
    PhysDerivTensor1DKernel<TData>
        <<<gridSize, blockSize>>>(nq0, nElmts, D0, in, out0);

    // Compute physical derivative.
    PhysDerivSegKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
        nq0, nCoord, nElmts, dfSize, df, out0, out1, out2);
}

template <typename TData, bool DEFORMED>
void PhysDeriv2DKernel(const size_t gridSize, const size_t blockSize,
                       LibUtilities::ShapeType shapetype, const size_t nq0,
                       const size_t nq1, const size_t nCoord,
                       const size_t nElmts, const TData *D0, const TData *D1,
                       const TData *Z0, const TData *Z1, const size_t dfSize,
                       TData *df, const TData *in, TData *out0, TData *out1,
                       TData *out2)
{
    // Compute tensorial derivative.
    PhysDerivTensor2DKernel<TData>
        <<<gridSize, blockSize>>>(nq0, nq1, nElmts, D0, D1, in, out0, out1);

    // Compute physical derivative.
    if (shapetype == LibUtilities::Quad)
    {
        PhysDerivQuadKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
            nq0, nq1, nCoord, nElmts, dfSize, df, out0, out1, out2);
    }
    else if (shapetype == LibUtilities::Tri)
    {
        PhysDerivTriKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
            nq0, nq1, nCoord, nElmts, Z0, Z1, dfSize, df, out0, out1, out2);
    }
}

template <typename TData, bool DEFORMED>
void PhysDeriv3DKernel(const size_t gridSize, const size_t blockSize,
                       LibUtilities::ShapeType shapetype, const size_t nq0,
                       const size_t nq1, const size_t nq2, const size_t nCoord,
                       const size_t nElmts, const TData *D0, const TData *D1,
                       const TData *D2, const TData *Z0, const TData *Z1,
                       const TData *Z2, const size_t dfSize, TData *df,
                       const TData *in, TData *out0, TData *out1, TData *out2)
{
    // Compute tensorial derivative.
    PhysDerivTensor3DKernel<TData><<<gridSize, blockSize>>>(
        nq0, nq1, nq2, nElmts, D0, D1, D2, in, out0, out1, out2);

    // Compute physical derivative.
    if (shapetype == LibUtilities::Hex)
    {
        PhysDerivHexKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
            nq0, nq1, nq2, nCoord, nElmts, dfSize, df, out0, out1, out2);
    }
    else if (shapetype == LibUtilities::Tet)
    {
        PhysDerivTetKernel<TData, DEFORMED>
            <<<gridSize, blockSize>>>(nq0, nq1, nq2, nCoord, nElmts, Z0, Z1, Z2,
                                      dfSize, df, out0, out1, out2);
    }
    else if (shapetype == LibUtilities::Prism)
    {
        PhysDerivPrismKernel<TData, DEFORMED>
            <<<gridSize, blockSize>>>(nq0, nq1, nq2, nCoord, nElmts, Z0, Z1, Z2,
                                      dfSize, df, out0, out1, out2);
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        PhysDerivPyrKernel<TData, DEFORMED>
            <<<gridSize, blockSize>>>(nq0, nq1, nq2, nCoord, nElmts, Z0, Z1, Z2,
                                      dfSize, df, out0, out1, out2);
    }
}

} // namespace Nektar::Operators::detail
