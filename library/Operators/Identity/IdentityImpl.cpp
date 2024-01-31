#include "IdentityStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorIdentityImpl<double, FieldState::Coeff,
                                 ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IdentityCoeffStdMat",
        OperatorIdentityImpl<double, FieldState::Coeff,
                             ImplStdMat>::instantiate,
        "...");

template <>
std::string OperatorIdentityImpl<double, FieldState::Phys,
                                 ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IdentityPhysStdMat",
        OperatorIdentityImpl<double, FieldState::Phys, ImplStdMat>::instantiate,
        "...");
} // namespace Nektar::Operators::detail
