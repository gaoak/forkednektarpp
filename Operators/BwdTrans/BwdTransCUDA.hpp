#include "MemoryRegionCUDA.hpp"
#include "Operators/BwdTrans/BwdTransCUDAKernels.cuh"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorHelper.cuh"

namespace Nektar::Operators::detail
{

template <typename TData>
void BwdTrans1DKernel(const size_t gridSize, const size_t blockSize,
                      const size_t nm0, const size_t nq0, const size_t nElmts,
                      const TData *basis0, const TData *in, TData *out);

template <typename TData>
void BwdTrans1DKernel_QP(const size_t nm0, const size_t nq0,
                         const size_t nElmts, const TData *basis0,
                         const TData *in, TData *out);

template <typename TData>
void BwdTrans2DKernel(const size_t gridSize, const size_t blockSize,
                      LibUtilities::ShapeType shapetype, const size_t nm0,
                      const size_t nm1, const size_t nq0, const size_t nq1,
                      const size_t nElmts, const bool correct,
                      const TData *basis0, const TData *basis1, const TData *in,
                      TData *out);

template <typename TData>
void BwdTrans2DKernel_QP(LibUtilities::ShapeType shapetype, const size_t nm0,
                         const size_t nm1, const size_t nq0, const size_t nq1,
                         const size_t nElmts, const bool correct,
                         const TData *basis0, const TData *basis1,
                         const TData *in, TData *out);

template <typename TData>
void BwdTrans3DKernel(const size_t gridSize, const size_t blockSize,
                      LibUtilities::ShapeType shapetype, const size_t nm0,
                      const size_t nm1, const size_t nm2, const size_t nq0,
                      const size_t nq1, const size_t nq2, const size_t nElmts,
                      const bool correct, const TData *basis0,
                      const TData *basis1, const TData *basis2, const TData *in,
                      TData *out);

template <typename TData>
void BwdTrans3DKernel_QP(LibUtilities::ShapeType shapetype, const size_t nm0,
                         const size_t nm1, const size_t nm2, const size_t nq0,
                         const size_t nq1, const size_t nq2,
                         const size_t nElmts, const bool correct,
                         const TData *basis0, const TData *basis1,
                         const TData *basis2, const TData *in, TData *out);

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

            // Deterime CUDA grid parameters.
            m_gridSize = nElmts / m_blockSize;
            m_gridSize += (nElmts % m_blockSize == 0) ? 0 : 1;

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
    size_t m_blockSize = 32;
    size_t m_gridSize;
};

template <typename TData>
void BwdTrans1DKernel(const size_t gridSize, const size_t blockSize,
                      const size_t nm0, const size_t nq0, const size_t nElmts,
                      const TData *basis0, const TData *in, TData *out)
{
    BwdTransSegKernel<<<gridSize, blockSize>>>(nm0, nq0, nElmts, basis0, in,
                                               out);
}

template <typename TData>
void BwdTrans1DKernel_QP(const size_t nm0, const size_t nq0,
                         const size_t nElmts, const TData *basis0,
                         const TData *in, TData *out)
{
    BwdTransSegKernel_QP<<<nElmts, nq0>>>(nm0, nq0, basis0, in, out);
}

template <typename TData>
void BwdTrans2DKernel(const size_t gridSize, const size_t blockSize,
                      LibUtilities::ShapeType shapetype, const size_t nm0,
                      const size_t nm1, const size_t nq0, const size_t nq1,
                      const size_t nElmts, const bool correct,
                      const TData *basis0, const TData *basis1, const TData *in,
                      TData *out)
{
    if (shapetype == LibUtilities::Quad)
    {
        BwdTransQuadKernel<<<gridSize, blockSize>>>(nm0, nm1, nq0, nq1, nElmts,
                                                    basis0, basis1, in, out);
    }
    else if (shapetype == LibUtilities::Tri)
    {
        size_t nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        BwdTransTriKernel<<<gridSize, blockSize>>>(nm0, nm1, nmTot, nq0, nq1,
                                                   nElmts, correct, basis0,
                                                   basis1, in, out);
    }
}

template <typename TData>
void BwdTrans2DKernel_QP(LibUtilities::ShapeType shapetype, const size_t nm0,
                         const size_t nm1, const size_t nq0, const size_t nq1,
                         const size_t nElmts, const bool correct,
                         const TData *basis0, const TData *basis1,
                         const TData *in, TData *out)
{
    if (shapetype == LibUtilities::Quad)
    {
        TData *wsp = 0;
        cudaMalloc((void **)&wsp, sizeof(TData) * nq0 * nm1);
        BwdTransQuadKernel_QP<<<nElmts, dim3(nq0, nq1)>>>(
            nm0, nm1, nq0, nq1, basis0, basis1, in, wsp, out);
        cudaFree(wsp);
    }
    else if (shapetype == LibUtilities::Tri)
    {
        size_t nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        TData *wsp = 0;
        cudaMalloc((void **)&wsp, sizeof(TData) * nm0 * nq1);
        BwdTransTriKernel_QP<<<nElmts, dim3(nq0, nq1)>>>(
            nm0, nm1, nmTot, nq0, nq1, correct, basis0, basis1, in, wsp, out);
        cudaFree(wsp);
    }
}

template <typename TData>
void BwdTrans3DKernel(const size_t gridSize, const size_t blockSize,
                      LibUtilities::ShapeType shapetype, const size_t nm0,
                      const size_t nm1, const size_t nm2, const size_t nq0,
                      const size_t nq1, const size_t nq2, const size_t nElmts,
                      const bool correct, const TData *basis0,
                      const TData *basis1, const TData *basis2, const TData *in,
                      TData *out)
{
    if (shapetype == LibUtilities::Hex)
    {
        BwdTransHexKernel<<<gridSize, blockSize>>>(nm0, nm1, nm2, nq0, nq1, nq2,
                                                   nElmts, basis0, basis1,
                                                   basis2, in, out);
    }
    else if (shapetype == LibUtilities::Tet)
    {
        size_t nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);
        BwdTransTetKernel<<<gridSize, blockSize>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct, basis0,
            basis1, basis2, in, out);
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        size_t nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);
        BwdTransPyrKernel<<<gridSize, blockSize>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct, basis0,
            basis1, basis2, in, out);
    }
    else if (shapetype == LibUtilities::Prism)
    {
        size_t nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);
        BwdTransPrismKernel<<<gridSize, blockSize>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nElmts, correct, basis0,
            basis1, basis2, in, out);
    }
}

template <typename TData>
void BwdTrans3DKernel_QP(LibUtilities::ShapeType shapetype, const size_t nm0,
                         const size_t nm1, const size_t nm2, const size_t nq0,
                         const size_t nq1, const size_t nq2,
                         const size_t nElmts, const bool correct,
                         const TData *basis0, const TData *basis1,
                         const TData *basis2, const TData *in, TData *out)
{
    if (shapetype == LibUtilities::Hex)
    {
        TData *wsp0 = 0;
        TData *wsp1 = 0;
        cudaMalloc((void **)&wsp0, sizeof(TData) * nq0 * nm1 * nm2);
        cudaMalloc((void **)&wsp1, sizeof(TData) * nm0 * nq1 * nq2);
        BwdTransHexKernel_QP<<<nElmts, dim3(nq0, nq1, nq2)>>>(
            nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2, in, wsp0,
            wsp1, out);
        cudaFree(wsp0);
        cudaFree(wsp1);
    }
    if (shapetype == LibUtilities::Tet)
    {
        size_t nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);
        TData *fpq = 0;
        TData *fp  = 0;
        cudaMalloc((void **)&fpq,
                   sizeof(TData) * (2 * nm1 - nm0 + 1) * nm0 / 2 * nq2);
        cudaMalloc((void **)&fp, sizeof(TData) * nm0 * nq1 * nq2);
        BwdTransTetKernel_QP<<<nElmts, dim3(nq0, nq1, nq2)>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, correct, basis0, basis1,
            basis2, in, fpq, fp, out);
        cudaFree(fpq);
        cudaFree(fp);
    }
    if (shapetype == LibUtilities::Pyr)
    {
        size_t nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);
        TData *fpq = 0;
        TData *fp  = 0;
        cudaMalloc((void **)&fpq, sizeof(TData) * nm0 * nm1 * nq2);
        cudaMalloc((void **)&fp, sizeof(TData) * nm0 * nq1 * nq2);
        BwdTransPyrKernel_QP<<<nElmts, dim3(nq0, nq1, nq2)>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, correct, basis0, basis1,
            basis2, in, fpq, fp, out);
        cudaFree(fpq);
        cudaFree(fp);
    }
    if (shapetype == LibUtilities::Prism)
    {
        size_t nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);
        TData *fpq = 0;
        TData *fp  = 0;
        cudaMalloc((void **)&fpq, sizeof(TData) * nm0 * nm1 * nq2);
        cudaMalloc((void **)&fp, sizeof(TData) * nm0 * nq1 * nq2);
        BwdTransPrismKernel_QP<<<nElmts, dim3(nq0, nq1, nq2)>>>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, correct, basis0, basis1,
            basis2, in, fpq, fp, out);
        cudaFree(fpq);
        cudaFree(fp);
    }
}

} // namespace Nektar::Operators::detail
