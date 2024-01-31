#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// Robin boundary condition operator base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorRobBndCond : public Operator<TData>
{

public:
    virtual ~OperatorRobBndCond() = default;

    OperatorRobBndCond(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &in,
                       Field<TData, FieldState::Coeff> &out,
                       const bool &negflag = false) = 0;
};

// Descriptor / traits class for RobBndCond to be used by Operator create
// function
template <typename TData = default_fp_type> struct RobBndCond
{
    using class_name = OperatorRobBndCond<TData>;
    OPERATORS_EXPORT static const std::string key;
    OPERATORS_EXPORT static const std::string default_impl;

    RobBndCond() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<RobBndCond<TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// Template for implementation of RobBndCond operator
template <typename TData, typename Op> class OperatorRobBndCondImpl;
} // namespace detail

} // namespace Nektar::Operators
