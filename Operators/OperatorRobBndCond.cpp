#include "OperatorRobBndCond.hpp"

namespace Nektar::Operators
{
// define static variables for RobBndCond operator descriptor for the default fp
// type config
template <> const std::string RobBndCond<default_fp_type>::key = "RobBndCond";

template <> const std::string RobBndCond<default_fp_type>::default_impl = "";
} // namespace Nektar::Operators
