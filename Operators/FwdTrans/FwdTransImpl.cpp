#include "FwdTransStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorFwdTransImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "FwdTrans", OperatorFwdTransImpl<double, ImplStdMat>::instantiate, "");
} // namespace Nektar::Operators::detail
