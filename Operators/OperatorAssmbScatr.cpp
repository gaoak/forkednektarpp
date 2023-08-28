#include "OperatorAssmbScatr.hpp"

namespace Nektar::Operators
{

// define static variables for Assembly+Scatter (for coeff field)
template <>
const std::string AssmbScatr<default_fp_type, FieldState::Coeff>::key = "AssmbScatr";

template <>
const std::string AssmbScatr<default_fp_type, FieldState::Coeff>::default_impl = "";

}