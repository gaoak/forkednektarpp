#include "BwdTransCUDA.hpp"

namespace Nektar::Operators::detail
{
template <>
std::string OperatorBwdTransImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "BwdTransCUDA", OperatorBwdTransImpl<double, ImplCUDA>::instantiate,
        "...");
}
