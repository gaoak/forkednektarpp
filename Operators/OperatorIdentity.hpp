#pragma once

#include <vector>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// Identity base class
template <typename TData, FieldState TFieldState>
class OperatorIdentity : public OperatorLinear<TData, TFieldState, TFieldState>
{
public:
    virtual ~OperatorIdentity() = default;

    OperatorIdentity(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, TFieldState, TFieldState>(std::move(expansionList))
    {
    }

    virtual void apply(Field<TData, TFieldState> &in,
                       Field<TData, TFieldState> &out) = 0;

    virtual void operator()(Field<TData, TFieldState> &in,
                            Field<TData, TFieldState> &out)
    {
        apply(in, out);
    }
};

// Descriptor / traits class for Identity
template <typename TData         = default_fp_type,
          FieldState TFieldState = FieldState::Coeff>
struct Identity
{
    using class_name = OperatorIdentity<TData, TFieldState>;
    using FieldIn    = Field<TData, TFieldState>;
    using FieldOut   = Field<TData, TFieldState>;
    static const std::string key;
    static const std::string default_impl;

    Identity() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<Identity<TData, TFieldState>>(
            expansionList, pKey);
    }
};

namespace detail
{
// declare class for implementation of Identity operator
template <typename TData, FieldState TFieldState, typename Op>
class OperatorIdentityImpl;
} // namespace detail

} // namespace Nektar::Operators
