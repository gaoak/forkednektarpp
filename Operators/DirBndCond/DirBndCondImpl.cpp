#include "DirBndCondStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorDirBndCondImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "DirBndCond", OperatorDirBndCondImpl<double, ImplStdMat>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
