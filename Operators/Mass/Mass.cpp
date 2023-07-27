#include "Mass.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
void OperatorMassImpl<TData>::apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
{
    std::cout << "Applying mass operator\n";

    // create temporary field for physical points
    auto tmp = Field<TData, FieldState::Phys>::create(in.GetBlocks());

    // transform coefficients into physical points
    this->m_BwdTransOp->apply(in, tmp);
    
    // take inner product of physical points
    this->m_IProductWRTBaseOp->apply(tmp, out);

    std::cout << "Finished mass operator\n";
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