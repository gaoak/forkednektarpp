#include "HelmSolveStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorHelmSolveImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "HelmSolve", OperatorHelmSolveImpl<double, ImplStdMat>::instantiate,
        "");

} // namespace Nektar::Operators::detail
