#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// AssmbScatr base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorAssmbScatr : public Operator<TData>
{
public:
    virtual ~OperatorAssmbScatr() = default;

    OperatorAssmbScatr(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &in,
                       Field<TData, FieldState::Coeff> &out,
                       const bool &zeroDir = false) = 0;

    virtual void operator()(Field<TData, FieldState::Coeff> &in,
                            Field<TData, FieldState::Coeff> &out,
                            const bool &zeroDir)
    {
        apply(in, out, zeroDir);
    }
};

// Descriptor / traits class for Assembly+scatter to be used by Operator create
// function
template <typename TData = default_fp_type> struct AssmbScatr
{
    using class_name = OperatorAssmbScatr<TData>;
    static const std::string key;
    static const std::string default_impl;

    AssmbScatr() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<AssmbScatr<TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// Template for implementation of assembly+scatter operator
template <typename TData, typename Op> class OperatorAssmbScatrImpl;
} // namespace detail

} // namespace Nektar::Operators
