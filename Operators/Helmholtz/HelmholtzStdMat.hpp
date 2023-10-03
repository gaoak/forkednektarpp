#include "Operators/BwdTrans/BwdTransStdMat.hpp"
#include "Operators/IProductWRTBase/IProductWRTBaseStdMat.hpp"
#include "Operators/IProductWRTDerivBase/IProductWRTDerivBaseStdMat.hpp"
#include "Operators/OperatorHelmholtz.hpp"
#include "Operators/PhysDeriv/PhysDerivStdMat.hpp"

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorHelmholtzImpl<TData, ImplStdMat> : public OperatorHelmholtz<TData>
{
public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtz<TData>(expansionList),
          m_bwd(Field<TData, FieldState::Phys>::create(
              GetBlockAttributes(FieldState::Phys, expansionList))),
          m_deriv(Field<TData, FieldState::Phys>::create(
              GetBlockAttributes(FieldState::Phys, expansionList),
              expansionList->GetCoordim(0))),
          m_derivcoeff(Field<TData, FieldState::Phys>::create(
              GetBlockAttributes(FieldState::Phys, expansionList),
              expansionList->GetCoordim(0)))
    {
        auto nCoord = this->m_expansionList->GetCoordim(0);

        m_BwdTransOp  = BwdTrans<>::create(this->m_expansionList, "StdMat");
        m_PhysDerivOp = PhysDeriv<>::create(this->m_expansionList, "StdMat");
        m_IProductWRTBaseOp =
            IProductWRTBase<>::create(this->m_expansionList, "StdMat");
        m_IProductWRTDerivBaseOp =
            IProductWRTDerivBase<>::create(this->m_expansionList, "StdMat");

        m_diffCoeff = Array<OneD, TData>(nCoord * nCoord, 0.0);
        for (size_t d = 0; d < nCoord; d++)
        {
            m_diffCoeff[d * nCoord + d] = 1.0; // temporary solution
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Step 1: BwdTrans
        m_BwdTransOp->apply(in, m_bwd);

        // Step 2: PhysDeriv
        m_PhysDerivOp->apply(m_bwd, m_deriv);

        // Step 3: Inner product for mass matrix operation
        m_IProductWRTBaseOp->apply(m_bwd, out, this->m_lambda);

        // Step 4: Multiply by diffusion coefficient
        DiffusionCoeff(m_deriv, m_derivcoeff);

        // Step 5: Inner product
        m_IProductWRTDerivBaseOp->apply(m_derivcoeff, out, true);
    }

    void DiffusionCoeff(Field<TData, FieldState::Phys> &deriv,
                        Field<TData, FieldState::Phys> &derivcoeff)
    {
        // Initialize pointers.
        auto *derivptr0      = deriv.GetStorage().GetCPUPtr();
        auto *derivptr1      = derivptr0 + deriv.GetFieldSize();
        auto *derivptr2      = derivptr1 + deriv.GetFieldSize();
        auto *derivcoeffptr0 = derivcoeff.GetStorage().GetCPUPtr();
        auto *derivcoeffptr1 = derivcoeffptr0 + derivcoeff.GetFieldSize();
        auto *derivcoeffptr2 = derivcoeffptr1 + derivcoeff.GetFieldSize();
        std::vector<TData *> derivptr{derivptr0, derivptr1, derivptr2};
        std::vector<TData *> derivcoeffptr{derivcoeffptr0, derivcoeffptr1,
                                           derivcoeffptr2};

        // Initialize index.
        size_t expIdx = 0;

        for (size_t block_idx = 0; block_idx < deriv.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = deriv.GetBlocks()[block_idx].num_elements;
            auto nCoord       = expPtr->GetCoordim();
            auto nqTot        = expPtr->GetTotPoints();

            // Multiply by diffusion coefficient.
            for (size_t d = 0; d < nCoord; d++)
            {
                Vmath::Smul(nqTot * nElmts, m_diffCoeff[d * nCoord],
                            derivptr[0], 1, derivcoeffptr[d], 1);
                for (size_t l = 1; l < nCoord; l++)
                {
                    Vmath::Svtvp(nqTot * nElmts, m_diffCoeff[d * nCoord + l],
                                 derivptr[l], 1, derivcoeffptr[d], 1,
                                 derivcoeffptr[d], 1);
                }
            }

            // Increment pointer and index for next element type.
            for (size_t d = 0; d < nCoord; d++)
            {
                derivptr[d] += nqTot * nElmts;
                derivcoeffptr[d] += nqTot * nElmts;
            }
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorHelmholtzImpl<TData, ImplStdMat>>(
            expansionList);
    }

    static std::string className;

private:
    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorPhysDeriv<TData>> m_PhysDerivOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    std::shared_ptr<OperatorIProductWRTDerivBase<TData>>
        m_IProductWRTDerivBaseOp;
    Field<TData, FieldState::Phys> m_bwd;
    Field<TData, FieldState::Phys> m_deriv;
    Field<TData, FieldState::Phys> m_derivcoeff;
    Array<OneD, TData> m_diffCoeff;
};

} // namespace Nektar::Operators::detail
