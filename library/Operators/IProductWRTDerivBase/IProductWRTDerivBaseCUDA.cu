#include "IProductWRTDerivBaseCUDA.hpp"

namespace Nektar::Operators::detail
{

// Add different IProductWRTDerivBase implementations to the factory.
template <>
std::string OperatorIProductWRTDerivBaseImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IProductWRTDerivBaseCUDA",
        OperatorIProductWRTDerivBaseImpl<double, ImplCUDA>::instantiate, "...");

} // namespace Nektar::Operators::detail
