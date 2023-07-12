#include "OperatorIProductWRTBase.hpp"

namespace Nektar::Operators
{

template <> const std::string IProductWRTBase<>::key = "IProductWRTBase";
template <> const std::string IProductWRTBase<>::default_impl = "StdMat";

} // namespace Nektar::Operators