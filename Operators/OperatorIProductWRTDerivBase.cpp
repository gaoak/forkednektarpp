#include "OperatorIProductWRTDerivBase.hpp"

namespace Nektar::Operators
{

template <>
const std::string IProductWRTDerivBase<>::key = "IProductWRTDerivBase";
template <> const std::string IProductWRTDerivBase<>::default_impl = "StdMat";

} // namespace Nektar::Operators
