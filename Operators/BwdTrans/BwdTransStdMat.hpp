#include "Operators/OperatorBwdTrans.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

// sum-factorisation implementation
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
        auto const *inptr = in.GetStorage().GetCPUPtr();
        auto *outptr      = out.GetStorage().GetCPUPtr();

        size_t exp_idx = 0;
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto const expPtr = this->m_expansionList->GetExp(exp_idx);

            Nektar::StdRegions::StdMatrixKey key(
                StdRegions::eBwdTrans, expPtr->DetShapeType(), *expPtr);
            auto const matPtr = expPtr->GetStdMatrix(key);

            auto const &block = in.GetBlocks()[block_idx];

            Blas::Dgemm('N', 'N', matPtr->GetRows(), block.num_elements,
                        matPtr->GetColumns(), 1.0, matPtr->GetRawPtr(),
                        matPtr->GetRows(), inptr, block.num_pts, 0.0, outptr,
                        expPtr->GetTotPoints());

            inptr += block.block_size;
            outptr += expPtr->GetTotPoints() * block.num_elements;
            exp_idx += block.num_elements;
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
