#pragma once

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// Matrix operator base class
template <typename TData>
class OperatorHelm : public OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>
{

public:
    OperatorHelm(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>(expansionList)
    {    
    }
};

// Descriptor / traits class for Helm to be used by Operator create function
template <typename TData>
struct Helm
{
    using class_name = OperatorHelm<TData>;
    static const std::string key;
    static const std::string default_impl;

    Helm() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<Helm<TData>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of Helm operator
    template <typename TData> 
    class OperatorHelmImpl;
}

}