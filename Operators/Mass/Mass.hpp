#pragma once

#include "Operators/OperatorMass.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorMassImpl : public OperatorMass<TData>
{
public:
    OperatorMassImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMass<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out);

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorMassImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}