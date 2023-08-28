#include "Matrix.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorMatrixImpl<double, FieldState::Coeff>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "MatrixCoeff",
        OperatorMatrixImpl<double, FieldState::Coeff>::instantiate, 
        ""
    );

template <>
std::string OperatorMatrixImpl<double, FieldState::Phys>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "MatrixPhys",
        OperatorMatrixImpl<double, FieldState::Phys>::instantiate, 
        ""
    );
}