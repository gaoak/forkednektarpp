#pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/BwdTrans/BwdTransCUDAKernels.cuh"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorHelper.cuh"

#define FLAG_QP false

namespace Nektar::Operators::detail
{

template <typename TData>
void BwdTrans1DKernel(const unsigned int gridSize, const unsigned int blockSize,
                      const unsigned int nm0, const unsigned int nq0,
                      const unsigned int nElmts, const TData *basis0,
                      const TData *in, TData *out);

template <typename TData>
void BwdTrans2DKernel(const unsigned int gridSize, const unsigned int blockSize,
                      LibUtilities::ShapeType shapetype, const unsigned int nm0,
                      const unsigned int nm1, const unsigned int nq0,
                      const unsigned int nq1, const unsigned int nElmts,
                      const bool correct, const TData *basis0,
                      const TData *basis1, const TData *in, TData *out);

template <typename TData>
void BwdTrans3DKernel(const unsigned int gridSize, const unsigned int blockSize,
                      LibUtilities::ShapeType shapetype, const unsigned int nm0,
                      const unsigned int nm1, const unsigned int nm2,
                      const unsigned int nq0, const unsigned int nq1,
                      const unsigned int nq2, const unsigned int nElmts,
                      const bool correct, const TData *basis0,
                      const TData *basis1, const TData *basis2, const TData *in,
                      TData *out);

// BwdTrans implementation
template <typename TData>
class OperatorBwdTransImpl<TData, ImplCUDA> : public OperatorBwdTrans<TData>
{
public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
        m_basis = GetBasisDataCUDA<TData>(expansionList);
    }

    ~OperatorBwdTransImpl()
    {
        DeallocateDataCUDA<TData>(m_basis);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Copy memory to GPU, if necessary and get raw pointers.
        auto const *inptr =
            in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Initialize index.
        size_t expIdx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = in.GetBlocks()[block_idx].num_elements;
            auto nmTot        = expPtr->GetNcoeffs();
            auto nqTot        = expPtr->GetTotPoints();
            auto shape        = expPtr->DetShapeType();

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
                auto nm0    = expPtr->GetBasisNumModes(0);
                auto nq0    = expPtr->GetNumPoints(0);
                BwdTrans1DKernel(m_gridSize, m_blockSize, nm0, nq0, nElmts,
                                 basis0, inptr, outptr);
            }
            else if (expPtr->GetShapeDimension() == 2)
            {
                auto basis0 = m_basis[basisKeys][0];
                auto basis1 = m_basis[basisKeys][1];
                auto nm0    = expPtr->GetBasisNumModes(0);
                auto nm1    = expPtr->GetBasisNumModes(1);
                auto nq0    = expPtr->GetNumPoints(0);
                auto nq1    = expPtr->GetNumPoints(1);
                BwdTrans2DKernel(m_gridSize, m_blockSize, shape, nm0, nm1, nq0,
                                 nq1, nElmts, correct, basis0, basis1, inptr,
                                 outptr);
            }
            else if (expPtr->GetShapeDimension() == 3)
            {
                auto basis0 = m_basis[basisKeys][0];
                auto basis1 = m_basis[basisKeys][1];
                auto basis2 = m_basis[basisKeys][2];
                auto nm0    = expPtr->GetBasisNumModes(0);
                auto nm1    = expPtr->GetBasisNumModes(1);
                auto nm2    = expPtr->GetBasisNumModes(2);
                auto nq0    = expPtr->GetNumPoints(0);
                auto nq1    = expPtr->GetNumPoints(1);
                auto nq2    = expPtr->GetNumPoints(2);
                BwdTrans3DKernel(m_gridSize, m_blockSize, shape, nm0, nm1, nm2,
                                 nq0, nq1, nq2, nElmts, correct, basis0, basis1,
                                 basis2, inptr, outptr);
            }

            // Increment pointer and index for next element type.
            inptr += in.GetBlocks()[block_idx].block_size;
            outptr += out.GetBlocks()[block_idx].block_size;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        MultiRegions::ExpListSharedPtr expansionList)
    {
        return std::make_unique<OperatorBwdTransImpl<TData, ImplCUDA>>(
            expansionList);
    }

    static std::string className;

private:
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>> m_basis;
    size_t m_gridSize  = 32;
    size_t m_blockSize = 32;
};

template <typename TData>
void BwdTrans1DKernel(const unsigned int gridSize, const unsigned int blockSize,
                      const unsigned int nm0, const unsigned int nq0,
                      const unsigned int nElmts, const TData *basis0,
                      const TData *in, TData *out)
{
    if (!FLAG_QP)
    {
        unsigned int nshared = sizeof(TData) * (nq0 * nm0);
        BwdTransSegKernel<TData><<<gridSize, blockSize, nshared>>>(
            nm0, nq0, nElmts, basis0, in, out);
    }
    else
    {
        unsigned int nshared = sizeof(TData) * (nm0);
        BwdTransSegKernel_QP<TData><<<gridSize, dim3(32), nshared>>>(
            nm0, nq0, nElmts, basis0, in, out);
    }
}

template <typename TData>
void BwdTrans2DKernel(const unsigned int gridSize, const unsigned int blockSize,
                      LibUtilities::ShapeType shapetype, const unsigned int nm0,
                      const unsigned int nm1, const unsigned int nq0,
                      const unsigned int nq1, const unsigned int nElmts,
                      const bool correct, const TData *basis0,
                      const TData *basis1, const TData *in, TData *out)
{
    if (shapetype == LibUtilities::Quad)
    {
        unsigned int nmTot =
            LibUtilities::StdQuadData::getNumberOfCoefficients(nm0, nm1);
        if (!FLAG_QP)
        {
            unsigned int nshared = sizeof(TData) * (nq0 * nm0 + nq1 * nm1);
            BwdTransQuadKernel<TData><<<gridSize, blockSize, nshared>>>(
                nm0, nm1, nmTot, nq0, nq1, nElmts, basis0, basis1, in, out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nmTot + nq0 * nm1);
            BwdTransQuadKernel_QP<TData><<<gridSize, dim3(8, 8), nshared>>>(
                nm0, nm1, nmTot, nq0, nq1, nElmts, basis0, basis1, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        if (!FLAG_QP)
        {
            BwdTransTriKernel<TData>
                <<<gridSize, blockSize>>>(nm0, nm1, nmTot, nq0, nq1, nElmts,
                                          correct, basis0, basis1, in, out);
        }
        else
        {
            unsigned int nshared = sizeof(TData) * (nmTot + nm0 * nq1);
            BwdTransTriKernel_QP<TData><<<nElmts, dim3(8, 8), nshared>>>(
                nm0, nm1, nmTot, nq0, nq1, nElmts, correct, basis0, basis1, in,
                out);
        }
    }
}

template <typename TData>
void BwdTrans3DKernel(const unsigned int gridSize, const unsigned int blockSize,
                      LibUtilities::ShapeType shapetype, const unsigned int nm0,
                      const unsigned int nm1, const unsigned int nm2,
                      const unsigned int nq0, const unsigned int nq1,
                      const unsigned int nq2, const unsigned int nElmts,
                      const bool correct, const TData *basis0,
                      const TData *basis1, const TData *basis2, const TData *in,
                      TData *out)
{
    if (shapetype == LibUtilities::Hex)
    {
        unsigned int nmTot =
            LibUtilities::StdHexData::getNumberOfCoefficients(nm0, nm1, nm2);
        if (!FLAG_QP)
        {
            unsigned int nshared =
                sizeof(TData) * (nq0 * nm0 + nq1 * nm1 + nq2 * nm2);
            BwdTransHexKernel<TData><<<gridSize, blockSize, nshared>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, basis0, basis1,
                basis2, in, out);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) * (nmTot + (nq0 * nm1 * nm2) + (nq0 * nq1 * nm2));
            BwdTransHexKernel_QP<TData><<<gridSize, dim3(4, 4, 4), nshared>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, basis0, basis1,
                basis2, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        unsigned int nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);
        if (!FLAG_QP)
        {
            BwdTransTetKernel<TData><<<gridSize, blockSize>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct, basis0,
                basis1, basis2, in, out);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) * (nmTot + ((2 * nm1 - nm0 + 1) * nm0 / 2 * nq2) +
                                 (nm0 * nq1 * nq2));
            BwdTransTetKernel_QP<TData><<<gridSize, dim3(4, 4, 4), nshared>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct, basis0,
                basis1, basis2, in, out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        unsigned int nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);
        if (!FLAG_QP)
        {
            BwdTransPrismKernel<TData><<<gridSize, blockSize>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct, basis0,
                basis1, basis2, in, out);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) * (nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2));
            BwdTransPrismKernel_QP<TData><<<gridSize, dim3(4, 4, 4), nshared>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct, basis0,
                basis1, basis2, in, out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        unsigned int nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);
        if (!FLAG_QP)
        {
            BwdTransPyrKernel<TData><<<gridSize, blockSize>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct, basis0,
                basis1, basis2, in, out);
        }
        else
        {
            unsigned int nshared =
                sizeof(TData) * (nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2));
            BwdTransPyrKernel_QP<TData><<<gridSize, dim3(4, 4, 4), nshared>>>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct, basis0,
                basis1, basis2, in, out);
        }
    }
}

} // namespace Nektar::Operators::detail
