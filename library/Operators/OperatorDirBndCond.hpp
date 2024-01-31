#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// Dirichlet boundary condition operator base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorDirBndCond : public Operator<TData>
{

public:
    virtual ~OperatorDirBndCond() = default;

    OperatorDirBndCond(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &out) = 0;

    virtual void operator()(Field<TData, FieldState::Coeff> &out)
    {
        apply(out);
    }
};

// Descriptor / traits class for DirBndCond to be used by Operator create
// function
template <typename TData = default_fp_type> struct DirBndCond
{
    using class_name = OperatorDirBndCond<TData>;
    OPERATORS_EXPORT static const std::string key;
    OPERATORS_EXPORT static const std::string default_impl;

    DirBndCond() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<DirBndCond<TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// Template for implementation of DirBndCond operator
template <typename TData, typename Op> class OperatorDirBndCondImpl;
} // namespace detail

} // namespace Nektar::Operators
