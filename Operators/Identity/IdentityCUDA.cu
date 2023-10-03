#include "IdentityCUDA.hpp"

namespace Nektar::Operators::detail
{
template <>
std::string OperatorIdentityImpl<double, FieldState::Coeff,
                                 ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IdentityCoeffCUDA",
        OperatorIdentityImpl<double, FieldState::Coeff, ImplCUDA>::instantiate,
        "...");
}

namespace Nektar::Operators::detail
{
template <>
std::string OperatorIdentityImpl<double, FieldState::Phys,
                                 ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IdentityPhysCUDA",
        OperatorIdentityImpl<double, FieldState::Phys, ImplCUDA>::instantiate,
        "...");
}