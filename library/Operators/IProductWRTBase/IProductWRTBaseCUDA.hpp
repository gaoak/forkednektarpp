#pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/IProductWRTBase/IProductWRTBaseCUDAKernels.cuh"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorIProductWRTBase.hpp"

#define FLAG_QP false

namespace Nektar::Operators::detail
{

template <typename TData, bool SCALE = false, bool APPEND = false,
          bool DEFORMED = false>
void IProductWRTBase1DKernel(const unsigned int gridSize,
                             const unsigned int blockSize,
                             const unsigned int nm0, const unsigned int nq0,
                             const unsigned int nElmts, const TData *basis0,
                             const TData *w0, const TData *jac, const TData *in,
                             TData *out, TData scale = 1.0);

template <typename TData, bool SCALE = false, bool APPEND = false,
          bool DEFORMED = false>
void IProductWRTBase2DKernel(const unsigned int gridSize,
                             const unsigned int blockSize,
                             LibUtilities::ShapeType shapetype,
                             const unsigned int nm0, const unsigned int nm1,
                             const unsigned int nq0, const unsigned int nq1,
                             const unsigned int nElmts, const bool correct,
                             const TData *basis0, const TData *basis1,
                             const TData *w0, const TData *w1, const TData *jac,
                             const TData *in, TData *out, TData scale = 1.0);

template <typename TData, bool SCALE = false, bool APPEND = false,
          bool DEFORMED = false>
void IProductWRTBase3DKernel(
    const unsigned int gridSize, const unsigned int blockSize,
    LibUtilities::ShapeType shapetype, const unsigned int nm0,
    const unsigned int nm1, const unsigned int nm2, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nElmts,
    const bool correct, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *w0, const TData *w1, const TData *w2,
    const TData *jac, const TData *in, TData *out, TData scale = 1.0);

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
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
void IProductWRTBase1DKernel(const unsigned int gridSize,
                             const unsigned int blockSize,
                             const unsigned int nm0, const unsigned int nq0,
                             const unsigned int nElmts, const TData *basis0,
                             const TData *w0, const TData *jac, const TData *in,
                             TData *out, TData scale)
{
    if (!FLAG_QP)
    {
        unsigned int nshared = sizeof(TData) * (nm0 * nq0 + nq0);
        IProductWRTBaseSegKernel<TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, blockSize, nshared>>>(nm0, nq0, nElmts, basis0, w0,
                                               jac, in, out, scale);
    }
    else
    {
        unsigned int nshared = sizeof(TData) * (nq0);
        IProductWRTBaseSegKernel<TData, SCALE, APPEND, DEFORMED>
            <<<gridSize, dim3(32), nshared>>>(nm0, nq0, nElmts, basis0, w0, jac,
                                              in, out, scale);
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
void IProductWRTBase2DKernel(const unsigned int gridSize,
                             const unsigned int blockSize,
                             LibUtilities::ShapeType shapetype,
                             const unsigned int nm0, const unsigned int nm1,
                             const unsigned int nq0, const unsigned int nq1,
                             const unsigned int nElmts, const bool correct,
                             const TData *basis0, const TData *basis1,
                             const TData *w0, const TData *w1, const TData *jac,
                             const TData *in, TData *out, TData scale)
{
    if (shapetype == LibUtilities::Quad)
    {
        unsigned int nmTot =
            LibUtilities::StdQuadData::getNumberOfCoefficients(nm0, nm1);
        if (!FLAG_QP)
        {
            unsigned int nshared =
                sizeof(TData) * (nm0 * nq0 + nm1 * nq1 + nq0 + nq1);
            IProductWRTBaseQuadKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(nm0, nm1, nmTot, nq0, nq1,
                                                   nElmts, basis0, basis1, w0,
                                                   w1, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 * nq1 + nm0 * nq1);
            IProductWRTBaseQuadKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(8, 8), nshared>>>(nm0, nm1, nmTot, nq0, nq1,
                                                    nElmts, basis0, basis1, w0,
                                                    w1, jac, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        if (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1);
            IProductWRTBaseTriKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nmTot, nq0, nq1, nElmts, correct, basis0, basis1,
                    w0, w1, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nq0 * nq1 + nm0 * nq1 + 1);
            IProductWRTBaseTriKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(8, 8), nshared>>>(
                    nm0, nm1, nmTot, nq0, nq1, nElmts, correct, basis0, basis1,
                    w0, w1, jac, in, out, scale);
        }
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
void IProductWRTBase3DKernel(
    const unsigned int gridSize, const unsigned int blockSize,
    LibUtilities::ShapeType shapetype, const unsigned int nm0,
    const unsigned int nm1, const unsigned int nm2, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nElmts,
    const bool correct, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *w0, const TData *w1, const TData *w2,
    const TData *jac, const TData *in, TData *out, TData scale)
{
    if (shapetype == LibUtilities::Hex)
    {
        unsigned int nmTot =
            LibUtilities::StdHexData::getNumberOfCoefficients(nm0, nm1, nm2);
        if (!FLAG_QP)
        {
            unsigned int nshared =
                sizeof(TData) *
                (nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 + nq1 + nq2);
            IProductWRTBaseHexKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, basis0, basis1,
                    basis2, w0, w1, w2, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) *
                (nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2);
            IProductWRTBaseHexKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(4, 4, 4), nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, basis0, basis1,
                    basis2, w0, w1, w2, jac, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        unsigned int nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);
        if (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            IProductWRTBaseTetKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) * (nq0 * nq1 * nq2 + nm0 * nq1 * nq2 +
                                 ((2 * nm1 - nm0 + 1) * nm0 / 2) * nq2 + nm2);
            IProductWRTBaseTetKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(4, 4, 4), nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        unsigned int nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);
        if (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            IProductWRTBasePrismKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) *
                (nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + nm1);
            IProductWRTBasePrismKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(4, 4, 4), nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        unsigned int nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);
        if (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 + nq1 + nq2);
            IProductWRTBasePyrKernel<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, blockSize, nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) *
                (nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + 1);
            IProductWRTBasePyrKernel_QP<TData, SCALE, APPEND, DEFORMED>
                <<<gridSize, dim3(4, 4, 4), nshared>>>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
        }
    }
}

} // namespace Nektar::Operators::detail
