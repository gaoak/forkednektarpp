#include "HelmSolve.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorHelmSolveImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "HelmSolve", OperatorHelmSolveImpl<double>::instantiate, "");

} // namespace Nektar::Operators::detail