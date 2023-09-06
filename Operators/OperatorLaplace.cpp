#include "OperatorLaplace.hpp"

namespace Nektar::Operators
{    
    // define static variables for Laplace operator descriptor for the default fp type config
    template <>
    const std::string Laplace<default_fp_type>::key = "Laplace";

    template <>
    const std::string Laplace<default_fp_type>::default_impl = "";
}