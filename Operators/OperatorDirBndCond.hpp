#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// Matrix operator base class
template <typename TData> class OperatorDirBndCond : public Operator<TData>
{

public:
    OperatorDirBndCond(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &inout) = 0;
};

// Descriptor / traits class for DirBndCond to be used by Operator create
// function
template <typename TData> struct DirBndCond
{
    using class_name = OperatorDirBndCond<TData>;
    static const std::string key;
    static const std::string default_impl;

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
// declare class for implementation of DirBndCond operator
template <typename TData, typename Op> class OperatorDirBndCondImpl;
} // namespace detail

} // namespace Nektar::Operators
