#include "Operators/OperatorBwdTrans.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorBwdTransImpl<TData, ImplStdMat> : public OperatorBwdTrans<TData>
{
public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Initialize pointers.
        auto const *inptr = in.GetStorage().GetCPUPtr();
        auto *outptr      = out.GetStorage().GetCPUPtr();

        size_t expIdx = 0;
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = in.GetBlocks()[block_idx].num_elements;
            auto nmTot        = expPtr->GetNcoeffs();
            auto nqTot        = expPtr->GetTotPoints();

            // Get BwdTrans matrix.
            Nektar::StdRegions::StdMatrixKey key(
                StdRegions::eBwdTrans, expPtr->DetShapeType(), *expPtr);
            auto const matPtr = expPtr->GetStdMatrix(key);

            // Perform matrix-matrix multiply.
            Blas::Dgemm('N', 'N', nqTot, nElmts, nmTot, 1.0,
                        matPtr->GetRawPtr(), nqTot, inptr, nmTot, 0.0, outptr,
                        nqTot);

            // Increment pointer and index for next element type.
            inptr += in.GetBlocks()[block_idx].block_size;
            outptr += out.GetBlocks()[block_idx].block_size;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorBwdTransImpl<TData, ImplStdMat>>(
            expansionList);
    }

    static std::string className;
};

} // namespace Nektar::Operators::detail
