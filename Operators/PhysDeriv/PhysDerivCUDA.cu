#include "PhysDerivCUDA.hpp"

namespace Nektar::Operators::detail
{
template <>
std::string OperatorPhysDerivImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "PhysDerivCUDA", OperatorPhysDerivImpl<double, ImplCUDA>::instantiate,
        "...");
}
