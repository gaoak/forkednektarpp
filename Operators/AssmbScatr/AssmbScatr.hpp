#pragma once

#include "Operators/OperatorAssmbScatr.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename TData, FieldState TFieldState>
class OperatorAssmbScatrImpl : public OperatorAssmbScatr<TData, TFieldState>
{
public:
    OperatorAssmbScatrImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAssmbScatr<TData, TFieldState>(std::move(expansionList))
    {
    }

    void apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out);

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorAssmbScatrImpl<TData, TFieldState>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}