#include "MemoryRegionCUDA.hpp"
#include "Operators/IProductWRTBase/IProductWRTBaseCUDA.hpp"
#include "Operators/IProductWRTDerivBase/IProductWRTDerivBaseCUDAKernels.cuh"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorIProductWRTDerivBase.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, bool DEFORMED = false>
void IProductWRTDerivBase1DKernel(const size_t gridSize, const size_t blockSize,
                                  const size_t nq0, const size_t nCoord,
                                  const size_t nElmts, const size_t dfSize,
                                  TData *df, TData *in0, TData *in1, TData *in2,
                                  TData *out0);

template <typename TData, bool DEFORMED = false>
void IProductWRTDerivBase2DKernel(const size_t gridSize, const size_t blockSize,
                                  LibUtilities::ShapeType shapetype,
                                  const size_t nq0, const size_t nq1,
                                  const size_t nCoord, const size_t nElmts,
                                  const TData *Z0, const TData *Z1,
                                  const size_t dfSize, TData *df, TData *in0,
                                  TData *in1, TData *in2, TData *out0,
                                  TData *out1);

template <typename TData, bool DEFORMED = false>
void IProductWRTDerivBase3DKernel(
    const size_t gridSize, const size_t blockSize,
    LibUtilities::ShapeType shapetype, const size_t nq0, const size_t nq1,
    const size_t nq2, const size_t nCoord, const size_t nElmts, const TData *Z0,
    const TData *Z1, const TData *Z2, const size_t dfSize, TData *df,
    TData *in0, TData *in1, TData *in2, TData *out0, TData *out1, TData *out2);

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
        auto ndata = this->m_expansionList->GetTotPoints();
        cudaMalloc((void **)&m_wsp0, sizeof(TData) * ndata);
        if (nCoord > 1)
        {
            cudaMalloc((void **)&m_wsp1, sizeof(TData) * ndata);
        }
        if (nCoord > 2)
        {
            cudaMalloc((void **)&m_wsp2, sizeof(TData) * ndata);
        }
    }

    ~OperatorIProductWRTDerivBaseImpl(void)
    {
        size_t nCoord = this->m_expansionList->GetCoordim(0);

        DeallocateDataCUDA<TData>(m_basis);
        DeallocateDataCUDA<TData>(m_dbasis);
        DeallocateDataCUDA<TData>(m_weight);
        DeallocateDataCUDA<TData>(m_Z);
        DeallocateDataCUDA<TData>(m_D);
        cudaFree(m_jac);
        cudaFree(m_derivFac);
        cudaFree(m_wsp0);
        if (nCoord > 1)
        {
            cudaFree(m_wsp1);
        }
        if (nCoord > 2)
        {
            cudaFree(m_wsp2);
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               bool APPEND = false) override
    {
        if (!APPEND) // Zero output (temporary solution)
        {
            auto *tmpptr =
                out.template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
            for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
                 ++block_idx)
            {
                for (size_t i = 0; i < out.GetBlocks()[block_idx].block_size;
                     i++)
                {
                    *(tmpptr++) = 0.0;
                }
            }
        }

        // Copy memory to GPU, if necessary and get raw pointers.
        auto *inptr0 = in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *inptr1 = inptr0 + in.GetFieldSize();
        auto *inptr2 = inptr1 + in.GetFieldSize();
        std::vector<TData *> inptr{inptr0, inptr1, inptr2};
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        std::vector<TData *> wspptr{m_wsp0, m_wsp1, m_wsp2};
        auto jacptr = m_jac;
        auto dfptr  = m_derivFac;

        // Initialize index.
        size_t expIdx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

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

            // Deterime CUDA grid parameters.
            m_gridSize = nElmts / m_blockSize;
            m_gridSize += (nElmts % m_blockSize == 0) ? 0 : 1;

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
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, m_dfSize,
                        dfptr, inptr[0], inptr[1], inptr[2], wspptr[0]);
                    IProductWRTBase1DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, nm0, nq0, nElmts, dbasis0, w0,
                        jacptr, wspptr[0], outptr);
                }
                else
                {
                    IProductWRTDerivBase1DKernel<TData, false>(
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, m_dfSize,
                        dfptr, inptr[0], inptr[1], inptr[2], wspptr[0]);
                    IProductWRTBase1DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, nm0, nq0, nElmts, dbasis0, w0,
                        jacptr, wspptr[0], outptr);
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
                if (deformed)
                {
                    IProductWRTDerivBase2DKernel<TData, true>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nCoord,
                        nElmts, Z0, Z1, m_dfSize, dfptr, inptr[0], inptr[1],
                        inptr[2], wspptr[0], wspptr[1]);
                    IProductWRTBase2DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                        nElmts, correct, dbasis0, basis1, w0, w1, jacptr,
                        wspptr[0], outptr);
                    IProductWRTBase2DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                        nElmts, correct, basis0, dbasis1, w0, w1, jacptr,
                        wspptr[1], outptr);
                }
                else
                {
                    IProductWRTDerivBase2DKernel<TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nCoord,
                        nElmts, Z0, Z1, m_dfSize, dfptr, inptr[0], inptr[1],
                        inptr[2], wspptr[0], wspptr[1]);
                    IProductWRTBase2DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                        nElmts, correct, dbasis0, basis1, w0, w1, jacptr,
                        wspptr[0], outptr);
                    IProductWRTBase2DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                        nElmts, correct, basis0, dbasis1, w0, w1, jacptr,
                        wspptr[1], outptr);
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
                if (deformed)
                {
                    IProductWRTDerivBase3DKernel<TData, true>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nCoord,
                        nElmts, Z0, Z1, Z2, m_dfSize, dfptr, inptr[0], inptr[1],
                        inptr[2], wspptr[0], wspptr[1], wspptr[2]);
                    IProductWRTBase3DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, dbasis0, basis1, basis2, w0, w1,
                        w2, jacptr, wspptr[0], outptr);
                    IProductWRTBase3DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, basis0, dbasis1, basis2, w0, w1,
                        w2, jacptr, wspptr[1], outptr);
                    IProductWRTBase3DKernel<TData, false, true, true>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, basis0, basis1, dbasis2, w0, w1,
                        w2, jacptr, wspptr[2], outptr);
                }
                else
                {
                    IProductWRTDerivBase3DKernel<TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nCoord,
                        nElmts, Z0, Z1, Z2, m_dfSize, dfptr, inptr[0], inptr[1],
                        inptr[2], wspptr[0], wspptr[1], wspptr[2]);
                    IProductWRTBase3DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, dbasis0, basis1, basis2, w0, w1,
                        w2, jacptr, wspptr[0], outptr);
                    IProductWRTBase3DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, basis0, dbasis1, basis2, w0, w1,
                        w2, jacptr, wspptr[1], outptr);
                    IProductWRTBase3DKernel<TData, false, true, false>(
                        m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0, nq1,
                        nq2, nElmts, correct, basis0, basis1, dbasis2, w0, w1,
                        w2, jacptr, wspptr[2], outptr);
                }
            }

            // Increment pointer and index for next element type.
            jacptr += deformed ? nqTot * nElmts : nElmts;
            dfptr += deformed ? nqTot * nElmts : nElmts;
            for (size_t d = 0; d < nDim; d++)
            {
                inptr[d] += nqTot * nElmts;
                wspptr[d] += nqTot * nElmts;
            }
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
    TData *m_wsp0, *m_wsp1, *m_wsp2;
    TData *m_derivFac;
    TData *m_jac;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_basis;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>>
        m_dbasis;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>>
        m_weight;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_D;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_Z;
    size_t m_dfSize;
    size_t m_blockSize = 32;
    size_t m_gridSize;
};

template <typename TData, bool DEFORMED>
void IProductWRTDerivBase1DKernel(const size_t gridSize, const size_t blockSize,
                                  const size_t nq0, const size_t nCoord,
                                  const size_t nElmts, const size_t dfSize,
                                  TData *df, TData *in0, TData *in1, TData *in2,
                                  TData *out0)
{
    IProductWRTDerivBaseSegKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
        nq0, nCoord, nElmts, dfSize, df, in0, in1, in2, out0);
}

template <typename TData, bool DEFORMED>
void IProductWRTDerivBase2DKernel(const size_t gridSize, const size_t blockSize,
                                  LibUtilities::ShapeType shapetype,
                                  const size_t nq0, const size_t nq1,
                                  const size_t nCoord, const size_t nElmts,
                                  const TData *Z0, const TData *Z1,
                                  const size_t dfSize, TData *df, TData *in0,
                                  TData *in1, TData *in2, TData *out0,
                                  TData *out1)
{
    if (shapetype == LibUtilities::Quad)
    {
        IProductWRTDerivBaseQuadKernel<TData, DEFORMED>
            <<<gridSize, blockSize>>>(nq0, nq1, nCoord, nElmts, dfSize, df, in0,
                                      in1, in2, out0, out1);
    }
    else if (shapetype == LibUtilities::Tri)
    {
        IProductWRTDerivBaseTriKernel<TData, DEFORMED>
            <<<gridSize, blockSize>>>(nq0, nq1, nCoord, nElmts, Z0, Z1, dfSize,
                                      df, in0, in1, in2, out0, out1);
    }
}

template <typename TData, bool DEFORMED>
void IProductWRTDerivBase3DKernel(
    const size_t gridSize, const size_t blockSize,
    LibUtilities::ShapeType shapetype, const size_t nq0, const size_t nq1,
    const size_t nq2, const size_t nCoord, const size_t nElmts, const TData *Z0,
    const TData *Z1, const TData *Z2, const size_t dfSize, TData *df,
    TData *in0, TData *in1, TData *in2, TData *out0, TData *out1, TData *out2)
{
    if (shapetype == LibUtilities::Hex)
    {
        IProductWRTDerivBaseHexKernel<TData, DEFORMED>
            <<<gridSize, blockSize>>>(nq0, nq1, nq2, nCoord, nElmts, dfSize, df,
                                      in0, in1, in2, out0, out1, out2);
    }
    else if (shapetype == LibUtilities::Tet)
    {
        IProductWRTDerivBaseTetKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
            nq0, nq1, nq2, nCoord, nElmts, Z0, Z1, Z2, dfSize, df, in0, in1,
            in2, out0, out1, out2);
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        IProductWRTDerivBasePyrKernel<TData, DEFORMED><<<gridSize, blockSize>>>(
            nq0, nq1, nq2, nCoord, nElmts, Z0, Z1, Z2, dfSize, df, in0, in1,
            in2, out0, out1, out2);
    }
    else if (shapetype == LibUtilities::Prism)
    {
        IProductWRTDerivBasePrismKernel<TData, DEFORMED>
            <<<gridSize, blockSize>>>(nq0, nq1, nq2, nCoord, nElmts, Z0, Z2,
                                      dfSize, df, in0, in1, in2, out0, out1,
                                      out2);
    }
}

} // namespace Nektar::Operators::detail
