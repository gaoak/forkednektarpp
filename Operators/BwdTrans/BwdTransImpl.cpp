#include "BwdTransSumFac.hpp"
#include "BwdTransMatFree.hpp"

namespace Nektar::Operators
{

template <>
std::string OperatorBwdTransImpl<double, ImplMatFree>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "OperatorBwdTransMatFree",
        OperatorBwdTransImpl<double, ImplMatFree>::instantiate, "...");

template <>
std::string OperatorBwdTransImpl<double, ImplSumFac>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "OperatorBwdTransSumFac",
        OperatorBwdTransImpl<double, ImplSumFac>::instantiate, "...");

}