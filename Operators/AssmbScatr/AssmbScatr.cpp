#include "AssmbScatr.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorAssmbScatrImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "AssmbScatr",
        OperatorAssmbScatrImpl<double>::instantiate, 
        ""
    );

}