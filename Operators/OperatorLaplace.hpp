#pragma once

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// Laplace operator base class
template <typename TData>
class OperatorLaplace : public Operator<TData>
{

public:
    OperatorLaplace(const MultiRegions::ExpListSharedPtr &expansionList) : Operator<TData>(expansionList)
    {    
    }

    virtual void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out) = 0;
};

// Descriptor / traits class for Laplace to be used by Operator create function
template <typename TData>
struct Laplace
{
    using class_name = OperatorLaplace<TData>;
    static const std::string key;
    static const std::string default_impl;

    Laplace() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<Laplace<TData>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of Laplace operator
    template <typename TData> 
    class OperatorLaplaceImpl;
}

}