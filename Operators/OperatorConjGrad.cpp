#include "OperatorConjGrad.hpp"

namespace Nektar::Operators
{

// define static variables for ConjGrad (coeff - coeff)
template <>
const std::string ConjGrad<default_fp_type, FieldState::Coeff>::key = "ConjGrad";

template <>
const std::string ConjGrad<default_fp_type, FieldState::Coeff>::default_impl = "";

}