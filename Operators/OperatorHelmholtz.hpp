#pragma once

#include "Field.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// Helmholtz base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorHelmholtz
    : public OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>
{
public:
    virtual ~OperatorHelmholtz() = default;

    OperatorHelmholtz(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>(
              expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &in,
                       Field<TData, FieldState::Coeff> &out) = 0;

    virtual void operator()(Field<TData, FieldState::Coeff> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        apply(in, out);
    }

    void SetLambda(TData lambda)
    {
        m_lambda = lambda;
    }

    TData m_lambda = 1.0;
};

// Descriptor / traits class for Helmholtz
template <typename TData = default_fp_type> struct Helmholtz
{
    using class_name = OperatorHelmholtz<TData>;
    using FieldIn    = Field<TData, FieldState::Coeff>;
    using FieldOut   = Field<TData, FieldState::Coeff>;
    static const std::string key;
    static const std::string default_impl;

    Helmholtz() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<Helmholtz<TData>>(expansionList,
                                                                  pKey);
    }
};

namespace detail
{
// Template for Helmholtz implementations
template <typename TData, typename Op> class OperatorHelmholtzImpl;
} // namespace detail

} // namespace Nektar::Operators
