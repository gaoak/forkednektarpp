#include "MassStdMat.hpp"

namespace Nektar::Operators::detail
{

// Add different Mass implementations to the factory.
template <>
std::string OperatorMassImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "MassStdMat", OperatorMassImpl<double, ImplStdMat>::instantiate, "...");

} // namespace Nektar::Operators::detail
