#include "OperatorBwdTrans.hpp"

namespace Nektar::Operators
{

template <> const std::string BwdTrans<>::key          = "BwdTrans";
template <> const std::string BwdTrans<>::default_impl = "MatFree";

} // namespace Nektar::Operators
