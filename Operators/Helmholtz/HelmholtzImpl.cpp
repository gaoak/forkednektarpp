#include "HelmholtzStdMat.hpp"

namespace Nektar::Operators::detail
{

// Add different Helmholtz implementations to the factory.
template <>
std::string OperatorHelmholtzImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "HelmholtzStdMat",
        OperatorHelmholtzImpl<double, ImplStdMat>::instantiate, "...");

} // namespace Nektar::Operators::detail
