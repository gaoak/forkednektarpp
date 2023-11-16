#include "MatrixStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorMatrixImpl<double, FieldState::Coeff,
                               ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "MatrixCoeff",
        OperatorMatrixImpl<double, FieldState::Coeff, ImplStdMat>::instantiate,
        "...");

template <>
std::string OperatorMatrixImpl<double, FieldState::Phys,
                               ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "MatrixPhys",
        OperatorMatrixImpl<double, FieldState::Phys, ImplStdMat>::instantiate,
        "...");
} // namespace Nektar::Operators::detail
