#include "ConjGradStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorConjGradImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "ConjGrad", OperatorConjGradImpl<double, ImplStdMat>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
