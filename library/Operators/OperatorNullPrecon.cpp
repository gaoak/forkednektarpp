#include "OperatorNullPrecon.hpp"

namespace Nektar::Operators
{
// define static variables for NullPrecon operator descriptor for the default fp
// type config
template <> const std::string NullPrecon<default_fp_type>::key = "NullPrecon";

template <> const std::string NullPrecon<default_fp_type>::default_impl = "";
} // namespace Nektar::Operators
