#include "RobBndCondStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorRobBndCondImpl<double, ImplStdMat>::className =
    GetOperatorFactory<default_fp_type>().RegisterCreatorFunction(
        "RobBndCond", OperatorRobBndCondImpl<double, ImplStdMat>::instantiate,
        "");

} // namespace Nektar::Operators::detail
