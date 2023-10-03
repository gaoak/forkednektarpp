#include "OperatorHelmSolve.hpp"

namespace Nektar::Operators
{
// define static variables for HelmSolve operator descriptor for the default fp
// type config
template <> const std::string HelmSolve<default_fp_type>::key = "HelmSolve";

template <> const std::string HelmSolve<default_fp_type>::default_impl = "";
} // namespace Nektar::Operators