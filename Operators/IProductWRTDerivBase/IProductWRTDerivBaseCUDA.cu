#include "IProductWRTDerivBaseCUDA.hpp"

namespace Nektar::Operators::detail
{
template <>
std::string OperatorIProductWRTDerivBaseImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IProductWRTDerivBaseCUDA",
        OperatorIProductWRTDerivBaseImpl<double, ImplCUDA>::instantiate, "...");
}
