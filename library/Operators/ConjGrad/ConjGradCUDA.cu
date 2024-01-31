#include "ConjGradCUDA.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorConjGradImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "ConjGradCUDA", OperatorConjGradImpl<double, ImplCUDA>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
