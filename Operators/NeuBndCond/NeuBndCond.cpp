#include "NeuBndCond.hpp"

namespace Nektar::Operators::detail
{
    
// Register implementation with Operator Factory
template <>
std::string OperatorNeuBndCondImpl<default_fp_type>::className =
    GetOperatorFactory<default_fp_type>().RegisterCreatorFunction(
        "NeuBndCond",
        OperatorNeuBndCondImpl<default_fp_type>::instantiate, 
        ""
    );

}