#include "MemoryRegionCUDA.hpp"
#include "Operators/Identity/IdentityCUDAKernels.cuh"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorIdentity.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
void IdentityKernel(const size_t gridSize, const size_t blockSize,
                    const size_t numPts, const size_t nElmts, const TData *in,
                    TData *out);

// Identity matrix implementation
template <typename TData, FieldState TFieldState>
class OperatorIdentityImpl<TData, TFieldState, ImplCUDA>
    : public OperatorIdentity<TData, TFieldState>
{
public:
    OperatorIdentityImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIdentity<TData, TFieldState>(std::move(expansionList))
    {
    }

    void apply(Field<TData, TFieldState> &in,
               Field<TData, TFieldState> &out) override
    {
        // Copy memory to GPU, if necessary and get raw pointers.
        auto const *inptr =
            in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Initialise index
        size_t expIdx = 0;

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = in.GetBlocks()[block_idx].num_elements;
            auto numPts       = (TFieldState == FieldState::Coeff)
                                    ? expPtr->GetNcoeffs()
                                    : expPtr->GetTotPoints();

            // Deterime CUDA grid parameters.
            m_gridSize = nElmts / m_blockSize;
            m_gridSize += (nElmts % m_blockSize == 0) ? 0 : 1;

            IdentityKernel(m_gridSize, m_blockSize, numPts, nElmts, inptr,
                           outptr);

            // Increment pointer and index for next element type.
            inptr += numPts * nElmts;
            outptr += numPts * nElmts;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIdentityImpl<TData, TFieldState, ImplCUDA>>(
            std::move(expansionList));
    }

    static std::string className;

private:
    size_t m_gridSize;
    size_t m_blockSize = 32;
};

template <typename TData>
void IdentityKernel(const size_t gridSize, const size_t blockSize,
                    const size_t numPts, const size_t nElmts, const TData *in,
                    TData *out)
{
    IdentityKernel<TData><<<gridSize, blockSize>>>(numPts, nElmts, in, out);
}

} // namespace Nektar::Operators::detail
