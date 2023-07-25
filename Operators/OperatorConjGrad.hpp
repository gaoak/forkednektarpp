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
class OperatorConjGrad : public Operator<TData>
{
protected:
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> m_LHS;
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> m_precon;

public:
    OperatorConjGrad(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(std::move(expansionList))
    {
    }

    // pure virtual - must be implemented in implementation class
    virtual void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out) = 0;

    void setLHS(std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> ptr)
    {
        m_LHS = ptr;
    }

    void setPrecon(std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> ptr)
    {
        m_precon = ptr;
    }    
};

// Descriptor / traits class for ConjGrad to be used by Operator create function
template <typename TData>
struct ConjGrad
{
    using class_name = OperatorConjGrad<TData>;
    static const std::string key;
    static const std::string default_impl;

    ConjGrad() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<ConjGrad<TData>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of CG operator
    template <typename TData> 
    class OperatorConjGradImpl;
}

}