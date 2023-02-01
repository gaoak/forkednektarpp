#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// BwdTrans base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorBwdTrans : public Operator<TData>
{
public:
    virtual void apply(Field<TData, FieldState::Coeff> &in,
                       Field<TData, FieldState::Phys> &out) = 0;
    virtual void operator()(Field<TData, FieldState::Coeff> &in,
                       Field<TData, FieldState::Phys> &out)
    {
        apply(in, out);
    }
};

// Descriptor / traits class for BwdTrans
template<typename TData = default_fp_type>
struct BwdTrans {
    using class_name = OperatorBwdTrans<TData>;
    static const std::string key;
    static const std::string default_impl;

    static std::unique_ptr<class_name> create(std::string pKey = "")
    {
        return Operator<TData>::template create<BwdTrans>(pKey);
    }
};

namespace detail {
// Template for BwdTrans implementations
template <typename TData, typename Op>
class OperatorBwdTransImpl;
}


}
