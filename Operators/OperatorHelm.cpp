#include "OperatorHelm.hpp"

namespace Nektar::Operators
{    
    // define static variables for Helm operator descriptor for the default fp type config
    template <>
    const std::string Helm<default_fp_type>::key = "Helm";

    template <>
    const std::string Helm<default_fp_type>::default_impl = "";
}