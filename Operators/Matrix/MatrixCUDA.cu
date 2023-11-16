#include "MatrixCUDA.hpp"

namespace Nektar::Operators::detail
{
template <>
std::string OperatorMatrixImpl<double, FieldState::Coeff, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "MatrixCoeffCUDA",
        OperatorMatrixImpl<double, FieldState::Coeff, ImplCUDA>::instantiate,
        "...");
}

namespace Nektar::Operators::detail
{
template <>
std::string OperatorMatrixImpl<double, FieldState::Phys, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "MatrixPhysCUDA",
        OperatorMatrixImpl<double, FieldState::Phys, ImplCUDA>::instantiate,
        "...");
}