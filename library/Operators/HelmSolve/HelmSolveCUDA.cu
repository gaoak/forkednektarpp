#include "HelmSolveCUDA.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorHelmSolveImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "HelmSolveCUDA",
        OperatorHelmSolveImpl<double, ImplCUDA>::instantiate, "...");

} // namespace Nektar::Operators::detail
