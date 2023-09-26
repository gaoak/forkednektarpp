#include "Operators/OperatorIProductWRTDerivBase.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorIProductWRTDerivBaseImpl<TData, ImplStdMat>
    : public OperatorIProductWRTDerivBase<TData>
{
public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(std::move(expansionList))
    {
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        size_t nDim      = this->m_expansionList->GetShapeDimension();

        // Initialise jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        m_jac          = Operator<TData>::SetJacobian(jacSize);
        m_derivFac     = Operator<TData>::SetDerivativeFactor(jacSize);

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the elements of expansionList.
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            auto const expPtr = this->m_expansionList->GetExp(e);

            // Fetch basiskeys of current element.
            for (size_t d = 0; d < nDim; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Copy data to m_matPtr, if necessary.
            if (m_matPtr.find(basisKeys) == m_matPtr.end())
            {
                size_t nqTot = expPtr->GetTotPoints();
                size_t nmTot = expPtr->GetNcoeffs();
                auto &matPtr = m_matPtr[basisKeys];
                matPtr       = Array<OneD, Array<OneD, TData>>(nDim);
                Array<OneD, NekDouble> tmp(nqTot), t;
                for (size_t d = 0; d < nDim; ++d)
                {
                    // Get IProductWRTDerivBase matrix.
                    matPtr[d] = Array<OneD, TData>(nqTot * nmTot);
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        expPtr->GetStdExp()->IProductWRTDerivBase(
                            d, tmp, t = matPtr[d] + i * nmTot);
                    }
                }
            }
        }

        // Initialize workspace memory.
        auto ndata = this->m_expansionList->GetTotPoints();
        m_wsp      = Array<OneD, Array<OneD, NekDouble>>(3);
        for (size_t i = 0; i < nDim; ++i)
        {
            m_wsp[i] = Array<OneD, NekDouble>(ndata);
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               bool APPEND = false) override
    {
        // Copy memory to GPU, if necessary and get raw pointers.
        auto *inptr0 = in.GetStorage().GetCPUPtr();
        auto *inptr1 = inptr0 + in.GetFieldSize();
        auto *inptr2 = inptr1 + in.GetFieldSize();
        std::vector<TData *> inptr{inptr0, inptr1, inptr2};
        auto *outptr = out.GetStorage().GetCPUPtr();
        std::vector<TData *> wspptr{m_wsp[0].get(), m_wsp[1].get(),
                                    m_wsp[2].get()};

        // Initialize index.
        size_t expIdx = 0;
        size_t jacIdx = 0;
        size_t dfIdx  = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = out.GetBlocks()[block_idx].num_elements;
            auto nDim         = expPtr->GetShapeDimension();
            auto nCoord       = expPtr->GetCoordim();
            auto nqTot        = expPtr->GetTotPoints();
            auto nmTot        = expPtr->GetNcoeffs();
            auto deformed     = expPtr->GetMetricInfo()->GetGtype() ==
                            SpatialDomains::eDeformed;

            // calculate dx/dxi in[0] + dy/dxi in[1] + dz/dxi in[2]
            if (deformed)
            {
                for (size_t d = 0; d < nDim; ++d)
                {
                    Vmath::Vmul(nqTot * nElmts, m_derivFac[d].get() + dfIdx, 1,
                                inptr[0], 1, wspptr[d], 1);
                    for (size_t i = 1; i < nCoord; ++i)
                    {
                        Vmath::Vvtvp(nqTot * nElmts,
                                     m_derivFac[d + i * nDim].get() + dfIdx, 1,
                                     inptr[i], 1, wspptr[d], 1, wspptr[d], 1);
                    }
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t d = 0; d < nDim; ++d)
                    {
                        Vmath::Smul(nqTot, m_derivFac[d][dfIdx + e],
                                    inptr[0] + e * nqTot, 1,
                                    wspptr[d] + e * nqTot, 1);
                        for (size_t i = 1; i < nCoord; ++i)
                        {
                            Vmath::Svtvp(
                                nqTot, m_derivFac[d + i * nDim][dfIdx + e],
                                inptr[i] + e * nqTot, 1, wspptr[d] + e * nqTot,
                                1, wspptr[d] + e * nqTot, 1);
                        }
                    }
                }
            }

            // Multiply by jacobian.
            if (deformed)
            {
                for (size_t d = 0; d < nDim; ++d)
                {
                    Vmath::Vmul(nqTot * nElmts, m_jac.get() + jacIdx, 1,
                                wspptr[d], 1, wspptr[d], 1);
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t d = 0; d < nDim; ++d)
                    {
                        Vmath::Smul(nqTot, m_jac[jacIdx + e],
                                    wspptr[d] + e * nqTot, 1,
                                    wspptr[d] + e * nqTot, 1);
                    }
                }
            }

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < nDim; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Fetch matrix.
            auto matPtr = m_matPtr[basisKeys];

            // Matrix products.
            for (size_t d = 0; d < nDim; d++)
            {
                TData alpha = (d == 0 && !APPEND) ? 0.0 : 1.0;
                Blas::Dgemm('N', 'N', nmTot, nElmts, nqTot, 1.0,
                            matPtr[d].get(), nmTot, wspptr[d], nqTot, alpha,
                            outptr, nmTot);

                // Increment pointer and index for next element type.
                inptr[d] += in.GetBlocks()[block_idx].block_size;
                wspptr[d] += nqTot * nElmts;
            }
            jacIdx += deformed ? nqTot * nElmts : nElmts;
            dfIdx += deformed ? nqTot * nElmts : nElmts;
            outptr += out.GetBlocks()[block_idx].block_size;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTDerivBaseImpl<TData, ImplStdMat>>(
            std::move(expansionList));
    }

    static std::string className;

private:
    Array<OneD, TData> m_jac;
    Array<OneD, Array<OneD, TData>> m_derivFac;
    std::map<std::vector<LibUtilities::BasisKey>,
             Array<OneD, Array<OneD, TData>>>
        m_matPtr;
    Array<OneD, Array<OneD, TData>> m_wsp;
};

} // namespace Nektar::Operators::detail
