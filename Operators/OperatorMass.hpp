#pragma once

#include <memory>
#include <algorithm>

#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

template <typename TData>
class OperatorMass : public OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>
{
public:
    OperatorMass(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>(expansionList)
    {
    }

    // apply must be implemented in implementation class
    virtual void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out) = 0;

};

// Descriptor / traits class for Mass to be used by Operator create function
template <typename TData>
struct Mass
{
    using class_name = OperatorMass<TData>;
    static const std::string key;
    static const std::string default_impl;

    Mass() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<Mass<TData>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of Mass matrix operator
    template <typename TData> 
    class OperatorMassImpl;
}

}