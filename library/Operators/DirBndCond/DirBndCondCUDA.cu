#include "DirBndCondCUDA.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorDirBndCondImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "DirBndCondCUDA", OperatorDirBndCondImpl<double, ImplCUDA>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
