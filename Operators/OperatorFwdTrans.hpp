#pragma once

#include "Field.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

template <typename TData> class OperatorFwdTrans : public Operator<TData>
{
public:
    OperatorFwdTrans(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual ~OperatorFwdTrans() = default;

    // apply must be implemented in implementation class
    virtual void apply(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Coeff> &out) = 0;
};

// Descriptor / traits class for FwdTrans to be used by Operator create function
template <typename TData> struct FwdTrans
{
    using class_name = OperatorFwdTrans<TData>;
    static const std::string key;
    static const std::string default_impl;

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
// declare class for implementation of FwdTrans matrix operator
template <typename TData> class OperatorFwdTransImpl;
} // namespace detail

} // namespace Nektar::Operators
