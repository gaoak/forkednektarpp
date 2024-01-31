#include "OperatorMass.hpp"

namespace Nektar::Operators
{
// define static variables for Mass operator descriptor for the default fp type
// config
template <> const std::string Mass<default_fp_type>::key = "Mass";

template <> const std::string Mass<default_fp_type>::default_impl = "StdMat";

} // namespace Nektar::Operators
