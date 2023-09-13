#pragma once

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"
#include "OperatorLinear.hpp"
#include "OperatorPrecon.hpp"

namespace Nektar::Operators
{

// Matrix operator base class
template <typename TData>
class OperatorHelmSolve : public Operator<TData>
{

public:
    OperatorHelmSolve(const MultiRegions::ExpListSharedPtr &expansionList) : Operator<TData>(expansionList)
    {    
    }

    virtual void apply(Field<TData, FieldState::Phys> &in, Field<TData, FieldState::Coeff> &out) = 0;

    virtual void setLambda(const TData &lambda) = 0;

    virtual void setPrecon(const std::shared_ptr<OperatorPrecon<TData>> &precon) = 0;
};

// Descriptor / traits class for HelmSolve to be used by Operator create function
template <typename TData>
struct HelmSolve
{
    using class_name = OperatorHelmSolve<TData>;
    static const std::string key;
    static const std::string default_impl;

    HelmSolve() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<HelmSolve<TData>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of HelmSolve operator
    template <typename TData> 
    class OperatorHelmSolveImpl;
}

}