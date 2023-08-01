#include "Operators/OperatorPhysDeriv.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorPhysDerivImpl<TData, ImplStdMat> : public OperatorPhysDeriv<TData>
{
public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        size_t nDim      = this->m_expansionList->GetShapeDimension();

        // Initialise derivative factor.
        size_t dfSize = Operator<TData>::GetGeometricFactorSize();
        m_derivFac    = Operator<TData>::SetDerivativeFactor(dfSize);

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
                auto &matPtr = m_matPtr[basisKeys];
                matPtr       = Array<OneD, Array<OneD, TData>>(nDim);
                Array<OneD, NekDouble> tmp(nqTot), t;
                for (size_t d = 0; d < nDim; ++d)
                {
                    // Get deriv matrix.
                    matPtr[d] = Array<OneD, TData>(nqTot * nqTot);
                    for (int i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        expPtr->GetStdExp()->PhysDeriv(
                            d, tmp, t = matPtr[d] + i * nqTot);
                    }
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out0,
               Field<TData, FieldState::Phys> &out1,
               Field<TData, FieldState::Phys> &out2) override
    {
        // Initialize pointers.
        auto const *inptr = in.GetStorage().GetCPUPtr();
        std::vector<TData *> outptr{out0.GetStorage().GetCPUPtr(),
                                    out1.GetStorage().GetCPUPtr(),
                                    out2.GetStorage().GetCPUPtr()};

        // Initialize index.
        size_t expIdx  = 0;
        size_t dfindex = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = in.GetBlocks()[block_idx].num_elements;
            auto nDim         = expPtr->GetShapeDimension();
            auto nCoord       = expPtr->GetCoordim();
            auto nqTot        = expPtr->GetTotPoints();
            auto ptsKeys      = expPtr->GetPointsKeys();
            auto deformed     = expPtr->GetMetricInfo()->GetGtype() ==
                            SpatialDomains::eDeformed;
            Array<OneD, Array<OneD, TData>> deriv(nDim);

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < expPtr->GetShapeDimension(); d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Get derivative matrix.
            auto &matPtr = m_matPtr[basisKeys];
            for (size_t d = 0; d < nDim; ++d)
            {
                // Perform matrix-matrix multiply.
                deriv[d] = Array<OneD, TData>(nqTot * nElmts);
                Blas::Dgemm('N', 'N', nqTot, nElmts, nqTot, 1.0,
                            matPtr[d].get(), nqTot, inptr, nqTot, 0.0,
                            deriv[d].get(), nqTot);
            }

            if (deformed)
            {
                for (size_t i = 0; i < nCoord; i++)
                {
                    Vmath::Vmul(nqTot * nElmts,
                                m_derivFac[i * nDim].get() + dfindex, 1,
                                deriv[0].get(), 1, outptr[i], 1);
                    for (size_t d = 1; d < nDim; d++)
                    {
                        Vmath::Vvtvp(nqTot * nElmts,
                                     m_derivFac[i * nDim + d].get() + dfindex,
                                     1, deriv[d].get(), 1, outptr[i], 1,
                                     outptr[i], 1);
                    }
                    outptr[i] += nqTot * nElmts;
                }
                dfindex += nqTot * nElmts;
            }
            else
            {
                for (size_t i = 0; i < nCoord; i++)
                {
                    for (size_t e = 0; e < nElmts; ++e)
                    {
                        Vmath::Smul(nqTot, m_derivFac[i * nDim][dfindex + e],
                                    deriv[0].get() + e * nqTot, 1, outptr[i],
                                    1);
                        for (size_t d = 1; d < nDim; d++)
                        {
                            Vmath::Svtvp(nqTot,
                                         m_derivFac[i * nDim + d][dfindex + e],
                                         deriv[d].get() + e * nqTot, 1,
                                         outptr[i], 1, outptr[i], 1);
                        }
                        outptr[i] += nqTot;
                    }
                }
                dfindex += nElmts;
            }
            inptr += nqTot * nElmts;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorPhysDerivImpl<TData, ImplStdMat>>(
            expansionList);
    }

    static std::string className;

private:
    Array<OneD, Array<OneD, TData>> m_derivFac;
    std::map<std::vector<LibUtilities::BasisKey>,
             Array<OneD, Array<OneD, TData>>>
        m_matPtr;
};

} // namespace Nektar::Operators::detail
