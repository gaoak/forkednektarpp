#include "DiagPrecon.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorDiagPreconImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "DiagPrecon",
        OperatorDiagPreconImpl<double>::instantiate, 
        ""
    );

}