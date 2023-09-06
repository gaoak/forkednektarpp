#pragma once

#include "Operators/OperatorLaplace.hpp"

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
class OperatorLaplaceImpl : public OperatorLaplace<TData>
{
public:
    OperatorLaplaceImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLaplace<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
    {
        /* IMPLEMENTATION OF LAPLACE OPERATOR */


    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorLaplaceImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}