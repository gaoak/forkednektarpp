#include "NeuBndCondStdMat.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorNeuBndCondImpl<double, ImplStdMat>::className =
    GetOperatorFactory<default_fp_type>().RegisterCreatorFunction(
        "NeuBndCond", OperatorNeuBndCondImpl<double, ImplStdMat>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
