#pragma once

#include "Field.hpp"
#include "OperatorLinear.hpp"
#include "OperatorPrecon.hpp"

namespace Nektar::Operators
{

// HelmSolve operator base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorHelmSolve : public Operator<TData>
{

public:
    virtual ~OperatorHelmSolve() = default;

    OperatorHelmSolve(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Coeff> &out) = 0;

    virtual void setLambda(const TData &lambda) = 0;

    virtual void setPrecon(
        const std::shared_ptr<OperatorPrecon<TData>> &precon) = 0;
};

// Descriptor / traits class for HelmSolve to be used by Operator create
// function
template <typename TData = default_fp_type> struct HelmSolve
{
    using class_name = OperatorHelmSolve<TData>;
    static const std::string key;
    static const std::string default_impl;

    HelmSolve() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<HelmSolve<TData>>(expansionList,
                                                                  pKey);
    }
};

namespace detail
{
// Template for implementation of HelmSolve operator
template <typename TData, typename Op> class OperatorHelmSolveImpl;
} // namespace detail

} // namespace Nektar::Operators
