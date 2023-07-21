#include "OperatorIdentity.hpp"

namespace Nektar::Operators
{    
    // define static variables for Identity operator descriptor for the default fp type config
    // For coeff -> coeff
    template <>
    const std::string Identity<default_fp_type, FieldState::Coeff>::key = "IdentityCoeff";

    template <>
    const std::string Identity<default_fp_type, FieldState::Coeff>::default_impl = "";

    // For phys -> phys
    template <>
    const std::string Identity<default_fp_type, FieldState::Phys>::key = "IdentityPhys";

    template <>
    const std::string Identity<default_fp_type, FieldState::Phys>::default_impl = "";
}