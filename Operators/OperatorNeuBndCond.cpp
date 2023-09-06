#include "OperatorNeuBndCond.hpp"

namespace Nektar::Operators
{    
    // define static variables for NeuBndCond operator descriptor for the default fp type config
    template <>
    const std::string NeuBndCond<default_fp_type>::key = "NeuBndCond";

    template <>
    const std::string NeuBndCond<default_fp_type>::default_impl = "";
}