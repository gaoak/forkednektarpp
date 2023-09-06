#include "DirBndCond.hpp"

namespace Nektar::Operators::detail
{

// Register implementation with Operator Factory
template <>
std::string OperatorDirBndCondImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "DirBndCond",
        OperatorDirBndCondImpl<double>::instantiate, 
        ""
    );

}