#include "Mass.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorMassImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "Mass",
        OperatorMassImpl<double>::instantiate, 
        ""
    );
}