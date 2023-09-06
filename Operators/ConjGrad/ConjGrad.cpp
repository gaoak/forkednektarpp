#include "ConjGrad.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorConjGradImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "ConjGrad",
        OperatorConjGradImpl<double>::instantiate, 
        ""
    );

}