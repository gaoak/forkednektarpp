#pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/IProductWRTBase/IProductWRTBaseCUDAKernels.cuh"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorIProductWRTBase.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, bool SCALE = false, bool APPEND = false,
          bool DEFORMED = false>
void IProductWRTBase1DKernel(const size_t gridSize, const size_t blockSize,
                             const size_t nm0, const size_t nq0,
                             const size_t nElmts, const TData *basis0,
                             const TData *w0, const TData *jac, const TData *in,
                             TData *out, TData scale = 1.0);

template <typename TData, bool SCALE = false, bool APPEND = false,
          bool DEFORMED = false>
void IProductWRTBase2DKernel(const size_t gridSize, const size_t blockSize,
                             LibUtilities::ShapeType shapetype,
                             const size_t nm0, const size_t nm1,
                             const size_t nq0, const size_t nq1,
                             const size_t nElmts, const bool correct,
                             const TData *basis0, const TData *basis1,
                             const TData *w0, const TData *w1, const TData *jac,
                             const TData *in, TData *out, TData scale = 1.0);

template <typename TData, bool SCALE = false, bool APPEND = false,
          bool DEFORMED = false>
void IProductWRTBase3DKernel(const size_t gridSize, const size_t blockSize,
                             LibUtilities::ShapeType shapetype,
                             const size_t nm0, const size_t nm1,
                             const size_t nm2, const size_t nq0,
                             const size_t nq1, const size_t nq2,
                             const size_t nElmts, const bool correct,
                             const TData *basis0, const TData *basis1,
                             const TData *basis2, const TData *w0,
                             const TData *w1, const TData *w2, const TData *jac,
                             const TData *in, TData *out, TData scale = 1.0);

// IProductWRTBase implementation
template <typename TData>
class OperatorIProductWRTBaseImpl<TData, ImplCUDA>
    : public OperatorIProductWRTBase<TData>
{
public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        auto jac       = Operator<TData>::SetJacobian(jacSize);
        cudaMalloc((void **)&m_jac, sizeof(TData) * jacSize);
        cudaMemcpy(m_jac, jac.get(), sizeof(TData) * jacSize,
                   cudaMemcpyHostToDevice);

        // Initialize basis.
        m_basis = GetBasisDataCUDA<TData>(expansionList);

        // Initialize weight.
        m_weight = GetWeightDataCUDA<TData>(expansionList);
    }

    ~OperatorIProductWRTBaseImpl(void)
    {
        DeallocateDataCUDA<TData>(m_basis);
        DeallocateDataCUDA<TData>(m_weight);
        cudaFree(m_jac);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               const TData lambda = 1.0) override
    {
        // Copy memory to GPU, if necessary and get raw pointers.
        auto const *inptr =
            in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Initialize index.
        size_t expIdx = 0;
        auto jacptr   = m_jac;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = in.GetBlocks()[block_idx].num_elements;
            auto nqTot        = expPtr->GetTotPoints();
            auto nmTot        = expPtr->GetNcoeffs();
            auto ptsKeys      = expPtr->GetPointsKeys();
            auto deformed     = expPtr->GetMetricInfo()->GetGtype() ==
                            SpatialDomains::eDeformed;

            // Deterime CUDA grid size.
            m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);

            // Flag for collapsed coordinate correction.
            bool correct = expPtr->GetBasis(0)->GetBasisType() ==
                           LibUtilities::eModified_A;

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < expPtr->GetShapeDimension(); d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Function call to kernel functions.
            if (expPtr->GetShapeDimension() == 1)
            {
                auto basis0 = m_basis[basisKeys][0];
                auto w0     = m_weight[basisKeys][0];
                auto nm0    = expPtr->GetBasisNumModes(0);
                auto nq0    = expPtr->GetNumPoints(0);
                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase1DKernel<TData, false, false, true>(
                            m_gridSize, m_blockSize, nm0, nq0, nElmts, basis0,
                            w0, jacptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase1DKernel<TData, true, false, true>(
                            m_gridSize, m_blockSize, nm0, nq0, nElmts, basis0,
                            w0, jacptr, inptr, outptr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase1DKernel<TData, false, false, false>(
                            m_gridSize, m_blockSize, nm0, nq0, nElmts, basis0,
                            w0, jacptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase1DKernel<TData, true, false, false>(
                            m_gridSize, m_blockSize, nm0, nq0, nElmts, basis0,
                            w0, jacptr, inptr, outptr, lambda);
                    }
                }
            }
            else if (expPtr->GetShapeDimension() == 2)
            {
                auto basis0 = m_basis[basisKeys][0];
                auto basis1 = m_basis[basisKeys][1];
                auto w0     = m_weight[basisKeys][0];
                auto w1     = m_weight[basisKeys][1];
                auto shape  = expPtr->DetShapeType();
                auto nm0    = expPtr->GetBasisNumModes(0);
                auto nm1    = expPtr->GetBasisNumModes(1);
                auto nq0    = expPtr->GetNumPoints(0);
                auto nq1    = expPtr->GetNumPoints(1);
                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase2DKernel<TData, false, false, true>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                            nElmts, correct, basis0, basis1, w0, w1, jacptr,
                            inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase2DKernel<TData, true, false, true>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                            nElmts, correct, basis0, basis1, w0, w1, jacptr,
                            inptr, outptr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase2DKernel<TData, false, false, false>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                            nElmts, correct, basis0, basis1, w0, w1, jacptr,
                            inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase2DKernel<TData, true, false, false>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                            nElmts, correct, basis0, basis1, w0, w1, jacptr,
                            inptr, outptr, lambda);
                    }
                }
            }
            else if (expPtr->GetShapeDimension() == 3)
            {
                auto basis0 = m_basis[basisKeys][0];
                auto basis1 = m_basis[basisKeys][1];
                auto basis2 = m_basis[basisKeys][2];
                auto w0     = m_weight[basisKeys][0];
                auto w1     = m_weight[basisKeys][1];
                auto w2     = m_weight[basisKeys][2];
                auto shape  = expPtr->DetShapeType();
                auto nm0    = expPtr->GetBasisNumModes(0);
                auto nm1    = expPtr->GetBasisNumModes(1);
                auto nm2    = expPtr->GetBasisNumModes(2);
                auto nq0    = expPtr->GetNumPoints(0);
                auto nq1    = expPtr->GetNumPoints(1);
                auto nq2    = expPtr->GetNumPoints(2);
                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase3DKernel<TData, false, false, true>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0,
                            nq1, nq2, nElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase3DKernel<TData, true, false, true>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0,
                            nq1, nq2, nElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacptr, inptr, outptr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase3DKernel<TData, false, false, false>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0,
                            nq1, nq2, nElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase3DKernel<TData, true, false, false>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0,
                            nq1, nq2, nElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacptr, inptr, outptr, lambda);
                    }
                }
            }

            // Increment pointer and index for next element type.
            jacptr += deformed ? nqTot * nElmts : nElmts;
            inptr += nqTot * nElmts;
            outptr += nmTot * nElmts;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        MultiRegions::ExpListSharedPtr expansionList)
    {
        return std::make_unique<OperatorIProductWRTBaseImpl<TData, ImplCUDA>>(
            expansionList);
    }

    static std::string className;

private:
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_basis;
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>>
        m_weight;
    TData *m_jac;
    size_t m_blockSize = 32;
    size_t m_gridSize;
};

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
void IProductWRTBase1DKernel(const size_t gridSize, const size_t blockSize,
                             const size_t nm0, const size_t nq0,
                             const size_t nElmts, const TData *basis0,
                             const TData *w0, const TData *jac, const TData *in,
                             TData *out, TData scale)
{
    IProductWRTBaseSegKernel<TData, SCALE, APPEND, DEFORMED>
        <<<gridSize, blockSize>>>(nm0, nq0, nElmts, basis0, w0, jac, in, out,
                                  scale);
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
void IProductWRTBase2DKernel(const size_t gridSize, const size_t blockSize,
                             LibUtilities::ShapeType shapetype,
                             const size_t nm0, const size_t nm1,
                             const size_t nq0, const size_t nq1,
                             const size_t nElmts, const bool correct,
                             const TData *basis0, const TData *basis1,
                             const TData *w0, const TData *w1, const TData *jac,
                             const TData *in, TData *out, TData scale)
{
    if (shapetype == LibUtilities::Quad)
    {
        IProductWRTBaseQuadKernel<TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, blockSize>>>(nm0, nm1, nq0, nq1, nElmts, basis0,
                                      basis1, w0, w1, jac, in, out, scale);
    }
    else if (shapetype == LibUtilities::Tri)
    {
        size_t nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        IProductWRTBaseTriKernel<TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, blockSize>>>(nm0, nm1, nmTot, nq0, nq1, nElmts,
                                      correct, basis0, basis1, w0, w1, jac, in,
                                      out, scale);
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
void IProductWRTBase3DKernel(
    const size_t gridSize, const size_t blockSize,
    LibUtilities::ShapeType shapetype, const size_t nm0, const size_t nm1,
    const size_t nm2, const size_t nq0, const size_t nq1, const size_t nq2,
    const size_t nElmts, const bool correct, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *w0, const TData *w1,
    const TData *w2, const TData *jac, const TData *in, TData *out, TData scale)
{
    if (shapetype == LibUtilities::Hex)
    {
        IProductWRTBaseHexKernel<TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, blockSize>>>(nm0, nm1, nm2, nq0, nq1, nq2, nElmts,
                                      basis0, basis1, basis2, w0, w1, w2, jac,
                                      in, out, scale);
    }
    else if (shapetype == LibUtilities::Tet)
    {
        size_t nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);
        IProductWRTBaseTetKernel<TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, blockSize>>>(nm0, nm1, nm2, nmTot, nq0, nq1, nq2,
                                      nElmts, correct, basis0, basis1, basis2,
                                      w0, w1, w2, jac, in, out, scale);
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        size_t nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);
        IProductWRTBasePyrKernel<TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, blockSize>>>(nm0, nm1, nm2, nmTot, nq0, nq1, nq2,
                                      nElmts, correct, basis0, basis1, basis2,
                                      w0, w1, w2, jac, in, out, scale);
    }
    else if (shapetype == LibUtilities::Prism)
    {
        size_t nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);
        IProductWRTBasePrismKernel<TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, blockSize>>>(nm0, nm1, nm2, nmTot, nq0, nq1, nq2,
                                      nElmts, correct, basis0, basis1, basis2,
                                      w0, w1, w2, jac, in, out, scale);
    }
}

} // namespace Nektar::Operators::detail
