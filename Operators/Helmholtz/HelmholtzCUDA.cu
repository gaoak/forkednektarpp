#include "HelmholtzCUDA.hpp"

namespace Nektar::Operators::detail
{
template <>
std::string OperatorHelmholtzImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "HelmholtzCUDA", OperatorHelmholtzImpl<double, ImplCUDA>::instantiate,
        "...");
}
