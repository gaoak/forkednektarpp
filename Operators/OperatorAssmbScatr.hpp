#pragma once

#include "Field.hpp"
#include "Operator.hpp"

using namespace Nektar;

namespace Nektar::Operators
{

template <typename TData, FieldState TFieldState>
class OperatorAssmbScatr : public Operator<TData>
{
public:
    OperatorAssmbScatr(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {   
    }

    virtual void apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out) = 0;
};

// Descriptor / traits class for Assembly+scatter to be used by Operator create function
template <typename TData, FieldState TFieldState>
struct AssmbScatr
{
    using class_name = OperatorAssmbScatr<TData, TFieldState>;
    static const std::string key;
    static const std::string default_impl;

    AssmbScatr() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<AssmbScatr<TData, TFieldState>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of assembly+scatter operator
    template <typename TData, FieldState TFieldState> 
    class OperatorAssmbScatrImpl;
}

}