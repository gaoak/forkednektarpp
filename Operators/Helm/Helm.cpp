#include "Helm.hpp"

#include <tuple>
#include <set>

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

#include <Operators/OperatorPhysDeriv.hpp>
#include <Operators/OperatorBwdTrans.hpp>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

template <typename TData>
void OperatorHelmImpl<TData>::apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
{
    /* IMPLEMENTATION OF HELMHOLTZ OPERATOR */

    // Do backwards transform to go from coeff to phys

    // Do phys deriv 

    // Add lambda * in

}

// ****************************************************************************************************************

// Register implementation with Operator Factory
template <>
std::string OperatorHelmImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "Helm",
        OperatorHelmImpl<double>::instantiate, 
        ""
    );

}