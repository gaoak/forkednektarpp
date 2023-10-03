#pragma once

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// Matrix operator base class
template <typename TData>
class OperatorNeuBndCond : public Operator<TData>
{

public:
    OperatorNeuBndCond(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {    
    }

    virtual void apply(Field<TData, FieldState::Coeff> &inout) = 0;

};

// Descriptor / traits class for NeuBndCond to be used by Operator create function
template <typename TData>
struct NeuBndCond
{
    using class_name = OperatorNeuBndCond<TData>;
    static const std::string key;
    static const std::string default_impl;

    NeuBndCond() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<NeuBndCond<TData>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of NeuBndCond operator
    template <typename TData> 
    class OperatorNeuBndCondImpl;
}

}
