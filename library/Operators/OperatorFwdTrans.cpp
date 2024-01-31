#include "OperatorFwdTrans.hpp"

namespace Nektar::Operators
{

// define static variables for FwdTrans operator descriptor for the default fp
// type config
template <> const std::string FwdTrans<default_fp_type>::key = "FwdTrans";

template <>
const std::string FwdTrans<default_fp_type>::default_impl = "StdMat";

} // namespace Nektar::Operators
