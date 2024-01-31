#include "FwdTransCUDA.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorFwdTransImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "FwdTransCUDA", OperatorFwdTransImpl<double, ImplCUDA>::instantiate,
        "...");
} // namespace Nektar::Operators::detail
