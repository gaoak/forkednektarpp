#include "Operators/OperatorIProductWRTBase.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorIProductWRTBaseImpl<TData, ImplStdMat>
    : public OperatorIProductWRTBase<TData>
{
public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(std::move(expansionList))
    {
        // Initialise jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        m_jac          = Operator<TData>::SetJacobian(jacSize);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               const TData lambda = 1.0) override
    {
        auto const *inptr = in.GetStorage().GetCPUPtr();
        auto *outptr      = out.GetStorage().GetCPUPtr();

        size_t expIdx = 0;
        size_t jacIdx = 0;
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = in.GetBlocks()[block_idx].num_elements;
            auto nqTot        = expPtr->GetTotPoints();
            auto nmTot        = expPtr->GetNcoeffs();

            Nektar::StdRegions::StdMatrixKey key(
                StdRegions::eIProductWRTBase, expPtr->DetShapeType(), *expPtr);

            // This is the B^{T} matrix
            auto const matPtr = expPtr->GetStdMatrix(key);

            Array<OneD, TData> wsp(nqTot * nElmts, 0.0);
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                for (size_t i = 0; i < nElmts * nqTot; ++i)
                {
                    wsp[i] = m_jac[jacIdx++] * inptr[i];
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        wsp[e * nqTot + i] =
                            m_jac[jacIdx] * inptr[e * nqTot + i];
                    }
                    jacIdx++;
                }
            }

            Blas::Dgemm('N', 'N', matPtr->GetRows(), nElmts,
                        matPtr->GetColumns(), lambda, matPtr->GetRawPtr(),
                        matPtr->GetRows(), wsp.get(), nqTot, 0.0, outptr,
                        nmTot);

            inptr += in.GetBlocks()[block_idx].block_size;
            outptr += out.GetBlocks()[block_idx].block_size;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorIProductWRTBaseImpl<TData, ImplStdMat>>(
            std::move(expansionList));
    }

    static std::string className;

private:
    Array<OneD, TData> m_jac;
};

} // namespace Nektar::Operators::detail
