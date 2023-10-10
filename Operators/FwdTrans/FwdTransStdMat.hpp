#pragma once

#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorFwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorIdentity.hpp"
#include "Operators/OperatorMass.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorFwdTransImpl<TData, ImplStdMat> : public OperatorFwdTrans<TData>
{
public:
    OperatorFwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorFwdTrans<TData>(std::move(expansionList)),
          m_field(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
        m_ConjGradOp = ConjGrad<TData>::create(this->m_expansionList);
        m_PreconOp =
            Identity<FieldState::Coeff, TData>::create(this->m_expansionList);
        m_MassOp = Mass<TData>::create(this->m_expansionList);
        m_IProductWRTBaseOp =
            IProductWRTBase<TData>::create(this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        auto blocks =
            GetBlockAttributes(FieldState::Coeff, this->m_expansionList);
        m_field = Field<TData, FieldState::Coeff>::create(blocks);

        // transform physical points f to coefficients f_hat
        m_IProductWRTBaseOp->apply(in, m_field);

        // set up and apply conjugate gradient
        // to solve for coefficients u_hat from f_hat
        m_ConjGradOp->setLHS(m_MassOp);
        m_ConjGradOp->setPrecon(m_PreconOp);
        m_ConjGradOp->apply(m_field, out);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorFwdTransImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorIdentity<TData, FieldState::Coeff>> m_PreconOp;
    std::shared_ptr<OperatorConjGrad<TData>> m_ConjGradOp;
    std::shared_ptr<OperatorMass<TData>> m_MassOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    Field<TData, FieldState::Coeff> m_field;
};

} // namespace Nektar::Operators::detail
