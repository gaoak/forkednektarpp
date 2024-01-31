#include "IProductWRTBaseMatFree.hpp"
#include "IProductWRTBaseStdMat.hpp"
#include "IProductWRTBaseSumFac.hpp"

namespace Nektar::Operators::detail
{

// Add different IProductWRTBase implementations to the factory.
template <>
std::string OperatorIProductWRTBaseImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IProductWRTBaseStdMat",
        OperatorIProductWRTBaseImpl<double, ImplStdMat>::instantiate, "...");

template <>
std::string OperatorIProductWRTBaseImpl<double, ImplMatFree>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IProductWRTBaseMatFree",
        OperatorIProductWRTBaseImpl<double, ImplMatFree>::instantiate, "...");

template <>
std::string OperatorIProductWRTBaseImpl<double, ImplSumFac>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IProductWRTBaseSumFac",
        OperatorIProductWRTBaseImpl<double, ImplSumFac>::instantiate, "...");

} // namespace Nektar::Operators::detail
