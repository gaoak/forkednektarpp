#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// IProductWRTDerivBase base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorIProductWRTDerivBase : public Operator<TData>
{
public:
    virtual ~OperatorIProductWRTDerivBase() = default;

    OperatorIProductWRTDerivBase(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Coeff> &out,
                       bool APPEND = false) = 0;

    virtual void operator()(Field<TData, FieldState::Phys> &in,
                            Field<TData, FieldState::Coeff> &out,
                            [[maybe_unused]] bool APPEND = false)
    {
        apply(in, out);
    }
};

// Descriptor / traits class for IProductWRTDerivBase
template <typename TData = default_fp_type> struct IProductWRTDerivBase
{
    using class_name = OperatorIProductWRTDerivBase<TData>;
    using FieldIn    = Field<TData, FieldState::Phys>;
    using FieldOut   = Field<TData, FieldState::Coeff>;
    static const std::string key;
    static const std::string default_impl;

    IProductWRTDerivBase() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<IProductWRTDerivBase<TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// Template for IProductWRTDerivBase implementations
template <typename TData, typename Op> class OperatorIProductWRTDerivBaseImpl;
} // namespace detail

} // namespace Nektar::Operators
