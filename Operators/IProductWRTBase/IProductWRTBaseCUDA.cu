#include "IProductWRTBaseCUDA.hpp"

namespace Nektar::Operators::detail
{
template <>
std::string OperatorIProductWRTBaseImpl<double, ImplCUDA>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IProductWRTBaseCUDA",
        OperatorIProductWRTBaseImpl<double, ImplCUDA>::instantiate, "...");
}
