#include "IProductWRTDerivBaseStdMat.hpp"

namespace Nektar::Operators::detail
{

// Add different IProductWRTDerivBase implementations to the factory.
template <>
std::string OperatorIProductWRTDerivBaseImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IProductWRTDerivBaseStdMat",
        OperatorIProductWRTDerivBaseImpl<double, ImplStdMat>::instantiate,
        "...");

} // namespace Nektar::Operators::detail
