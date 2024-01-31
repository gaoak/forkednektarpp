#include "OperatorIdentity.hpp"

namespace Nektar::Operators
{
// define static variables for Identity operator descriptor for the default fp
// type config For coeff -> coeff
template <>
const std::string Identity<FieldState::Coeff, default_fp_type>::key =
    "IdentityCoeff";

template <>
const std::string Identity<FieldState::Coeff, default_fp_type>::default_impl =
    "StdMat";

// For phys -> phys
template <>
const std::string Identity<FieldState::Phys, default_fp_type>::key =
    "IdentityPhys";

template <>
const std::string Identity<FieldState::Phys, default_fp_type>::default_impl =
    "StdMat";
} // namespace Nektar::Operators
