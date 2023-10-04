#pragma once

#include "Field.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// BwdTrans base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorBwdTrans
    : public OperatorLinear<TData, FieldState::Coeff, FieldState::Phys>
{
public:
    virtual ~OperatorBwdTrans() = default;

    OperatorBwdTrans(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, FieldState::Coeff, FieldState::Phys>(
              expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &in,
                       Field<TData, FieldState::Phys> &out) override
    {
    }

    virtual void operator()(Field<TData, FieldState::Coeff> &in,
                            Field<TData, FieldState::Phys> &out)
    {
        apply(in, out);
    }
};

// Descriptor / traits class for BwdTrans
template <typename TData = default_fp_type> struct BwdTrans
{
    using class_name = OperatorBwdTrans<TData>;
    using FieldIn    = Field<TData, FieldState::Coeff>;
    using FieldOut   = Field<TData, FieldState::Phys>;
    static const std::string key;
    static const std::string default_impl;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<BwdTrans>(expansionList, pKey);
    }
};

namespace detail
{
// Template for BwdTrans implementations
template <typename TData, typename Op> class OperatorBwdTransImpl;
} // namespace detail

} // namespace Nektar::Operators
