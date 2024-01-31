#include "OperatorDiagPrecon.hpp"

namespace Nektar::Operators
{
// define static variables for DiagPrecon operator descriptor for the default fp
// type config
template <> const std::string DiagPrecon<default_fp_type>::key = "DiagPrecon";

template <> const std::string DiagPrecon<default_fp_type>::default_impl = "";
} // namespace Nektar::Operators