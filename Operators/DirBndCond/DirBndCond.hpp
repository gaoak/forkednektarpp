#pragma once

#include "Operators/OperatorDirBndCond.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorDirBndCondImpl : public OperatorDirBndCond<TData>
{
public:
    OperatorDirBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDirBndCond<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &inout);

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDirBndCondImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}