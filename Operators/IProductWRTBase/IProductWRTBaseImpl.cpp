#include "IProductWRTBaseStdMat.hpp"
#include "IProductWRTBaseMatFree.hpp"

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

} // namespace Nektar::Operators::detail
