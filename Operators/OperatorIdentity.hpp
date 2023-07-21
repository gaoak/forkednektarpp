#pragma once

#include <vector>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// Matrix operator base class
template <typename TData, FieldState TFieldState>
class OperatorIdentity : public OperatorLinear<TData, TFieldState, TFieldState>
{

public:
    OperatorIdentity(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, TFieldState, TFieldState>(expansionList)
    {    
    }

    virtual void apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out) = 0;

};

// Descriptor / traits class for Identity to be used by Operator create function
template <typename TData, FieldState TFieldState>
struct Identity
{
    using class_name = OperatorIdentity<TData, TFieldState>;
    static const std::string key;
    static const std::string default_impl;

    Identity() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<Identity<TData, TFieldState>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of Identity operator
    template <typename TData, FieldState TFieldState> 
    class OperatorIdentityImpl;
}

}