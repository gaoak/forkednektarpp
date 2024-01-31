#include "AddTraceIntegralStdMat.hpp"

namespace Nektar::Operators::detail
{

// Add different AddTraceIntegral implementations to the factory.
template <>
std::string OperatorAddTraceIntegralImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "AddTraceIntegralStdMat",
        OperatorAddTraceIntegralImpl<double, ImplStdMat>::instantiate, "...");

} // namespace Nektar::Operators::detail
