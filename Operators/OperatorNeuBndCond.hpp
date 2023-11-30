#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// Neuman boundary condition operator base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorNeuBndCond : public Operator<TData>
{
public:
    virtual ~OperatorNeuBndCond() = default;

    OperatorNeuBndCond(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &inout) = 0;

    virtual void operator()(Field<TData, FieldState::Coeff> &inout)
    {
        apply(inout);
    }
};

// Descriptor / traits class for NeuBndCond to be used by Operator create
// function
template <typename TData = default_fp_type> struct NeuBndCond
{
    using class_name = OperatorNeuBndCond<TData>;
    static const std::string key;
    static const std::string default_impl;

    NeuBndCond() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<NeuBndCond<TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// Template for implementation of NeuBndCond operator
template <typename TData, typename Op> class OperatorNeuBndCondImpl;
} // namespace detail

} // namespace Nektar::Operators
