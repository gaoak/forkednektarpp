#include "IProductWRTBaseCUDA.hpp"

namespace Nektar::Operators::detail
{

// Add different IProductWRTBase implementations to the factory.
template <>
std::string OperatorIProductWRTBaseImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IProductWRTBaseCUDA",
        OperatorIProductWRTBaseImpl<double, ImplCUDA>::instantiate, "...");

} // namespace Nektar::Operators::detail
