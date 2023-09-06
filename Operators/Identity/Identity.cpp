#include "Identity.hpp"

namespace Nektar::Operators::detail
{
    
// Register implementation with Operator Factory
template <>
std::string OperatorIdentityImpl<double, FieldState::Coeff>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IdentityCoeff",
        OperatorIdentityImpl<double, FieldState::Coeff>::instantiate, 
        ""
    );

template <>
std::string OperatorIdentityImpl<double, FieldState::Phys>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IdentityPhys",
        OperatorIdentityImpl<double, FieldState::Phys>::instantiate, 
        ""
    );
}