#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// IProductWRTBase base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorIProductWRTBase : public Operator<TData>
{
public:
    virtual ~OperatorIProductWRTBase() = default;

    OperatorIProductWRTBase(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Coeff> &out,
                       const TData lambda = 1.0) = 0;

    virtual void operator()(Field<TData, FieldState::Phys> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        apply(in, out);
    }
};

// Descriptor / traits class for IProductWRTBase
template <typename TData = default_fp_type> struct IProductWRTBase
{
    using class_name = OperatorIProductWRTBase<TData>;
    using FieldIn    = Field<TData, FieldState::Phys>;
    using FieldOut   = Field<TData, FieldState::Coeff>;
    OPERATORS_EXPORT static const std::string key;
    OPERATORS_EXPORT static const std::string default_impl;

    IProductWRTBase() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<IProductWRTBase<TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// Template for IProductWRTBase implementations
template <typename TData, typename Op> class OperatorIProductWRTBaseImpl;
} // namespace detail

} // namespace Nektar::Operators
