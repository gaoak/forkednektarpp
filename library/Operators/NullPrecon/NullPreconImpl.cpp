#include "NullPreconStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorNullPreconImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "NullPrecon", OperatorNullPreconImpl<double, ImplStdMat>::instantiate,
        "");

} // namespace Nektar::Operators::detail
