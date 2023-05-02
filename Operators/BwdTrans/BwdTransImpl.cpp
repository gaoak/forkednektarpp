#include "BwdTransStdMat.hpp"
#include "BwdTransMatFree.hpp"
#include "BwdTransSumFac.hpp"

namespace Nektar::Operators::detail
{

// Add different BwdTrans implementations to the factory.
template <>
std::string OperatorBwdTransImpl<double, ImplMatFree>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "BwdTransMatFree",
        OperatorBwdTransImpl<double, ImplMatFree>::instantiate, "...");

template <>
std::string OperatorBwdTransImpl<double, ImplSumFac>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "BwdTransSumFac", OperatorBwdTransImpl<double, ImplSumFac>::instantiate,
        "...");

template <>
std::string OperatorBwdTransImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "BwdTransStdMat", OperatorBwdTransImpl<double, ImplStdMat>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
