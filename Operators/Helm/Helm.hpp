#pragma once

#include "Operators/OperatorHelm.hpp"

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
class OperatorHelmImpl : public OperatorHelm<TData>
{
public:
    OperatorHelmImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelm<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
    {
        /* IMPLEMENTATION OF HELMHOLTZ OPERATOR */

        // [ (dphi_p, dphi_q) + lambda*(phi_p, phi_q) ] uhat_q

        // (dphi_p, dphi_q) uhat_q --> out field

        // (phi_p, phi_q) uhat_q (mass matrix) --> tmp field

        // tmp field *= lambda

        // tmp field + out field

    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorHelmImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}