#include "Mass.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

template <typename TData>
void OperatorMassImpl<TData>::apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
{
    // create temporary field for physical points
    auto blocks = GetBlockAttributes(FieldState::Phys, this->m_expansionList);
    this->m_field = Field<TData, FieldState::Phys>::create(blocks);

    // transform coefficients into physical points
    this->m_BwdTransOp->apply(in, this->m_field);

    // take inner product of physical points
    this->m_IProductWRTBaseOp->apply(this->m_field, out);
}

// ****************************************************************************************************************

// Register implementation with Operator Factory
template <>
std::string OperatorMassImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "Mass",
        OperatorMassImpl<double>::instantiate, 
        ""
    );
}