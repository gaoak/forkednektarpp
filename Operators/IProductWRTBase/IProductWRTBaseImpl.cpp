#include "IProductWRTBaseStdMat.hpp"

namespace Nektar::Operators::detail
{

// Add different IProductWRTBase implementations to the factory.
template <>
std::string OperatorIProductWRTBaseImpl<double, ImplStdMat>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IProductWRTBaseStdMat",
        OperatorIProductWRTBaseImpl<double, ImplStdMat>::instantiate, "...");

} // namespace Nektar::Operators::detail
