#include "OperatorMatrix.hpp"

namespace Nektar::Operators
{
// define static variables for Matrix operator descriptor for the default fp
// type config For coeff -> coeff
template <>
const std::string Matrix<default_fp_type, FieldState::Coeff>::key =
    "MatrixCoeff";

template <>
const std::string Matrix<default_fp_type, FieldState::Coeff>::default_impl = "";

// For phys -> phys
template <>
const std::string Matrix<default_fp_type, FieldState::Phys>::key = "MatrixPhys";

template <>
const std::string Matrix<default_fp_type, FieldState::Phys>::default_impl = "";
} // namespace Nektar::Operators