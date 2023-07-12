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
        size_t jacSize   = 0;
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        // Calculate the jacobian array size
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            // Determine shape and type of the element
            auto const expPtr = this->m_expansionList->GetExp(e);
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                jacSize += expPtr->GetTotPoints();
            }
            else
            {
                jacSize++;
            }
        }

        // Allocate memory for the jacobian
        m_jac = {jacSize, 0.0};

        // Initialise jacobian.
        size_t index = 0;
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            auto expPtr = this->m_expansionList->GetExp(e);
            auto &auxJac =
                expPtr->GetMetricInfo()->GetJac(expPtr->GetPointsKeys());
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                size_t nqe = expPtr->GetTotPoints();
                for (size_t i = 0; i < nqe; ++i)
                {
                    m_jac[index++] = auxJac[i];
                }
            }
            else
            {
                m_jac[index++] = auxJac[0];
            }
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
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

            Array<OneD, NekDouble> wsp(nqTot * nElmts, 0.0);
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
                        matPtr->GetColumns(), 1.0, matPtr->GetRawPtr(),
                        matPtr->GetRows(), wsp.get(), nqTot, 0.0, outptr,
                        nmTot);

            inptr += nqTot * nElmts;
            outptr += nmTot * nElmts;
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
    Array<OneD, NekDouble> m_jac;
};

} // namespace Nektar::Operators::detail
