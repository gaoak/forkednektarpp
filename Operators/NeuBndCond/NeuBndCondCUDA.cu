#include "NeuBndCondCUDA.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorNeuBndCondImpl<double, ImplCUDA>::className =
    GetOperatorFactory<default_fp_type>().RegisterCreatorFunction(
        "NeuBndCondCUDA", OperatorNeuBndCondImpl<double, ImplCUDA>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
