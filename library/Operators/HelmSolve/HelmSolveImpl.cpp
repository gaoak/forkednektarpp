#include "HelmSolveStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorHelmSolveImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "HelmSolveStdMat",
        OperatorHelmSolveImpl<double, ImplStdMat>::instantiate, "...");

} // namespace Nektar::Operators::detail
