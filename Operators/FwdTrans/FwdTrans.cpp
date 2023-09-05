#include "FwdTrans.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

template <typename TData>
void OperatorFwdTransImpl<TData>::apply(Field<TData, FieldState::Phys> &in, Field<TData, FieldState::Coeff> &out)
{
    auto blocks = GetBlockAttributes(FieldState::Coeff, this->m_expansionList);
    m_field = Field<TData, FieldState::Coeff>::create(blocks);

    // transform physical points f to coefficients f_hat
    m_IProductWRTBaseOp->apply(in, m_field);

    // set up and apply conjugate gradient
    // to solve for coefficients u_hat from f_hat
    m_ConjGradOp->setLHS(m_MassOp);
    m_ConjGradOp->setPrecon(m_PreconOp);
    m_ConjGradOp->apply(m_field, out);
}

// ****************************************************************************************************************

// Register implementation with Operator Factory
template <>
std::string OperatorFwdTransImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "FwdTrans",
        OperatorFwdTransImpl<double>::instantiate, 
        ""
    );
}