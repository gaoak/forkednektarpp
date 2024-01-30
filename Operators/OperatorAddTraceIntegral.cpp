#include "OperatorAddTraceIntegral.hpp"

namespace Nektar::Operators
{

template <> const std::string AddTraceIntegral<>::key = "AddTraceIntegral";
template <> const std::string AddTraceIntegral<>::default_impl = "StdMat";

} // namespace Nektar::Operators
