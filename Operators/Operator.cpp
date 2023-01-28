#include "Operators/Operator.hpp"

namespace Nektar::Operators
{
    
template< typename TData>
OperatorFactory<TData> &GetOperatorFactory()
{
    static OperatorFactory<TData> instance;
    return instance;
}

}