#pragma once

#include "Field.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// ConjGrad base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorConjGrad : public Operator<TData>
{
public:
    virtual ~OperatorConjGrad() = default;

    OperatorConjGrad(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &in,
                       Field<TData, FieldState::Coeff> &out) = 0;

    virtual void operator()(Field<TData, FieldState::Coeff> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        apply(in, out);
    }

    void setLHS(const std::shared_ptr<OperatorLinear<TData, FieldState::Coeff,
                                                     FieldState::Coeff>> &ptr)
    {
        m_LHS = ptr;
    }

    void setPrecon(
        const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &ptr)
    {
        m_precon = ptr;
    }

protected:
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>>
        m_LHS;
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>>
        m_precon;
};

// Descriptor / traits class for ConjGrad to be used by Operator create function
template <typename TData = default_fp_type> struct ConjGrad
{
    using class_name = OperatorConjGrad<TData>;
    OPERATORS_EXPORT static const std::string key;
    OPERATORS_EXPORT static const std::string default_impl;

    ConjGrad() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<ConjGrad<TData>>(expansionList,
                                                                 pKey);
    }
};

namespace detail
{
// Template for implementation of CG operator
template <typename TData, typename Op> class OperatorConjGradImpl;
} // namespace detail

} // namespace Nektar::Operators
