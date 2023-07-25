#pragma once

#include <memory>
#include <algorithm>

#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

template <typename TData, FieldState TFieldState>
class OperatorConjGrad : public Operator<TData>
{
protected:
    std::shared_ptr<OperatorLinear<TData, TFieldState>> m_LHS;
    std::shared_ptr<OperatorLinear<TData, TFieldState>> m_precon;

public:
    OperatorConjGrad(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(std::move(expansionList))
    {
    }

    // pure virtual - must be implemented in implementation class
    virtual void apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out) = 0;

    void setLHS(std::shared_ptr<OperatorLinear<TData, TFieldState>> ptr)
    {
        p_LHS = ptr;
    }

    void setPrecon(std::shared_ptr<OperatorLinear<TData, TFieldState>> ptr)
    {
        p_precon = ptr;
    }    
};

// Descriptor / traits class for ConjGrad to be used by Operator create function
template <typename TData, FieldState TFieldState>
struct ConjGrad
{
    using class_name = OperatorConjGrad<TData, TFieldState>;
    static const std::string key;
    static const std::string default_impl;

    ConjGrad() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<ConjGrad<TData, TFieldState>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of CG operator
    template <typename TData, FieldState TFieldState> 
    class OperatorConjGradImpl;
}

}