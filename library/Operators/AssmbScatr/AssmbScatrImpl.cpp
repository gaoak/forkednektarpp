#include "AssmbScatrStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorAssmbScatrImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "AssmbScatr", OperatorAssmbScatrImpl<double, ImplStdMat>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
