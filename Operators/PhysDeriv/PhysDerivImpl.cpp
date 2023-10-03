#include "PhysDerivMatFree.hpp"
#include "PhysDerivStdMat.hpp"

namespace Nektar::Operators::detail
{

// Add different PhysDeriv implementations to the factory.
template <>
std::string OperatorPhysDerivImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "PhysDerivStdMat",
        OperatorPhysDerivImpl<double, ImplStdMat>::instantiate, "...");

template <>
std::string OperatorPhysDerivImpl<double, ImplMatFree>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "PhysDerivMatFree",
        OperatorPhysDerivImpl<double, ImplMatFree>::instantiate, "...");

} // namespace Nektar::Operators::detail
