#include "PhysDerivStdMat.hpp"

namespace Nektar::Operators::detail
{

// Add different PhysDeriv implementations to the factory.
template <>
std::string OperatorPhysDerivImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "PhysDerivStdMat",
        OperatorPhysDerivImpl<double, ImplStdMat>::instantiate, "...");

} // namespace Nektar::Operators::detail
