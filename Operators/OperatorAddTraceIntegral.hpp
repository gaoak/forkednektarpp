#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// AddTraceIntegral base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorAddTraceIntegral : public Operator<TData>
{
public:
    virtual ~OperatorAddTraceIntegral() = default;

    OperatorAddTraceIntegral(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Coeff> &out) = 0;
    virtual void operator()(Field<TData, FieldState::Phys> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        apply(in, out);
    }
};

// Descriptor / traits class for AddTraceIntegral
template <typename TData = default_fp_type> struct AddTraceIntegral
{
    using class_name = OperatorAddTraceIntegral<TData>;
    using FieldIn    = Field<TData, FieldState::Phys>;
    using FieldOut   = Field<TData, FieldState::Coeff>;
    static const std::string key;
    static const std::string default_impl;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<AddTraceIntegral>(expansionList,
                                                                  pKey);
    }
};

namespace detail
{
// Template for AddTraceIntegral implementations
template <typename TData, typename Op> class OperatorAddTraceIntegralImpl;
} // namespace detail

} // namespace Nektar::Operators
