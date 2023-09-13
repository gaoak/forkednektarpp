#include "OperatorHelmholtz.hpp"

namespace Nektar::Operators
{

template <> const std::string Helmholtz<>::key          = "Helmholtz";
template <> const std::string Helmholtz<>::default_impl = "MatFree";

} // namespace Nektar::Operators
