#include "OperatorPhysDeriv.hpp"

namespace Nektar::Operators
{

template <> const std::string PhysDeriv<>::key          = "PhysDeriv";
template <> const std::string PhysDeriv<>::default_impl = "StdMat";

} // namespace Nektar::Operators
