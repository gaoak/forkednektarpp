#include "AssmbScatrCUDA.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorAssmbScatrImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "AssmbScatrCUDA", OperatorAssmbScatrImpl<double, ImplCUDA>::instantiate,
        "");

} // namespace Nektar::Operators::detail
