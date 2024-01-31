#include "DiagPreconCUDA.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorDiagPreconImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "DiagPreconCUDA", OperatorDiagPreconImpl<double, ImplCUDA>::instantiate,
        "");

} // namespace Nektar::Operators::detail
