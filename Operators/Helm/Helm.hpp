#pragma once

#include "Operators/OperatorHelm.hpp"

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

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out);

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