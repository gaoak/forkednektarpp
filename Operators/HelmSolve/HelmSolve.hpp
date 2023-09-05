#pragma once

#include "Operators/OperatorHelmSolve.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorHelmSolveImpl : public OperatorHelmSolve<TData>
{
public:
    OperatorHelmSolveImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmSolve<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Phys> &in, Field<TData, FieldState::Coeff> &out)
    {
        /* IMPLEMENTATION OF HELMHOLTZ SOLVE OPERATOR */

        // IProductWRT of RHS

        // Handle Dirichlet BCs

        // Solve for u_hat using Conjugate Gradient

    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorHelmSolveImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}