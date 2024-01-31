#include "MassCUDA.hpp"

namespace Nektar::Operators::detail
{

// Add different Mass implementations to the factory.
template <>
std::string OperatorMassImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "MassCUDA", OperatorMassImpl<double, ImplCUDA>::instantiate, "...");

} // namespace Nektar::Operators::detail
