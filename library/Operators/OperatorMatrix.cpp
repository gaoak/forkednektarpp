#include "OperatorMatrix.hpp"

namespace Nektar::Operators
{
// define static variables for Matrix operator descriptor for the default fp
// type config For coeff -> coeff
template <>
const std::string Matrix<FieldState::Coeff, default_fp_type>::key =
    "MatrixCoeff";

template <>
const std::string Matrix<FieldState::Coeff, default_fp_type>::default_impl = "";

// For phys -> phys
template <>
const std::string Matrix<FieldState::Phys, default_fp_type>::key = "MatrixPhys";

template <>
const std::string Matrix<FieldState::Phys, default_fp_type>::default_impl = "";
} // namespace Nektar::Operators
