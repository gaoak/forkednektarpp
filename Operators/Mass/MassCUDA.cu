#include "MassCUDA.hpp"

namespace Nektar::Operators::detail
{
template <>
std::string OperatorMassImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "MassCUDA", OperatorMassImpl<double, ImplCUDA>::instantiate, "...");
}
