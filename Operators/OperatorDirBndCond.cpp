#include "OperatorDirBndCond.hpp"

namespace Nektar::Operators
{    
    // define static variables for DirBndCond operator descriptor for the default fp type config
    template <>
    const std::string DirBndCond<default_fp_type>::key = "DirBndCond";

    template <>
    const std::string DirBndCond<default_fp_type>::default_impl = "";
}