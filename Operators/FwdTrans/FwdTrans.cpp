#include "FwdTrans.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

template <typename TData>
void OperatorFwdTransImpl<TData>::apply(Field<TData, FieldState::Phys> &in, Field<TData, FieldState::Coeff> &out)
{
    // create temporary field for coefficient points
    auto blocks = GetBlockAttributes(FieldState::Coeff, this->m_expansionList);
    this->m_field = Field<TData, FieldState::Coeff>::create(blocks);

    // transform physical points f to coefficients f_hat
    this->m_IProductWRTBaseOp->apply(in, this->m_field);

    // set up and apply conjugate gradient
    // to solve for coefficients u_hat from f_hat
    this->m_ConjGradOp->setLHS(this->m_MassOp);
    this->m_ConjGradOp->setPrecon(this->m_PreconOp);
    this->m_ConjGradOp->apply(this->m_field, out);
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