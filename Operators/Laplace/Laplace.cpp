#include "Laplace.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorLaplaceImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "Laplace",
        OperatorLaplaceImpl<double>::instantiate, 
        ""
    );

}