#include "NullPreconCUDA.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorNullPreconImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "NullPreconCUDA", OperatorNullPreconImpl<double, ImplCUDA>::instantiate,
        "");

} // namespace Nektar::Operators::detail
