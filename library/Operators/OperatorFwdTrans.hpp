#pragma once

#include "Field.hpp"
#include "OperatorLinear.hpp"
#include "OperatorPrecon.hpp"

namespace Nektar::Operators
{

// FwdTrans base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorFwdTrans : public Operator<TData>
{
public:
    virtual ~OperatorFwdTrans() = default;

    OperatorFwdTrans(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    // apply must be implemented in implementation class
    virtual void apply(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Coeff> &out) = 0;

    virtual void operator()(Field<TData, FieldState::Phys> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        apply(in, out);
    }

    virtual void setPrecon(
        const std::shared_ptr<OperatorPrecon<TData>> &precon) = 0;
};

// Descriptor / traits class for FwdTrans
template <typename TData = default_fp_type> struct FwdTrans
{
    using class_name = OperatorFwdTrans<TData>;
    OPERATORS_EXPORT static const std::string key;
    OPERATORS_EXPORT static const std::string default_impl;

    FwdTrans() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<FwdTrans<TData>>(expansionList,
                                                                 pKey);
    }
};

namespace detail
{
// Template for implementation of FwdTrans matrix operator
template <typename TData, typename Op> class OperatorFwdTransImpl;
} // namespace detail

} // namespace Nektar::Operators
