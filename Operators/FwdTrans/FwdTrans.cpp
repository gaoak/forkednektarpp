#include "FwdTrans.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorFwdTransImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "FwdTrans", OperatorFwdTransImpl<double>::instantiate, "");
} // namespace Nektar::Operators::detail