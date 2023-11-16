#include "DiagPreconStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorDiagPreconImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "DiagPrecon", OperatorDiagPreconImpl<double, ImplStdMat>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
