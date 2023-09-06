#pragma once

#include "Operators/OperatorHelmSolve.hpp"

#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorDirBndCond.hpp"
#include "Operators/OperatorNeuBndCond.hpp"
#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorLinear.hpp"
#include "Operators/OperatorHelm.hpp"

using namespace Nektar;
using namespace Nektar::Operators;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorHelmSolveImpl : public OperatorHelmSolve<TData>
{
public:
    OperatorHelmSolveImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmSolve<TData>(std::move(expansionList)),
        m_rhs(Field<TData, FieldState::Coeff>::create(GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
        m_IProdOp   = IProductWRTBase<TData>::create(this->m_expansionList);
        m_DirBCOp   = DirBndCond<TData>::create(this->m_expansionList);
        m_NeuBCOp   = NeuBndCond<TData>::create(this->m_expansionList);
        m_CGOp      = ConjGrad<TData>::create(this->m_expansionList);
        m_HelmOp    = Helm<TData>::create(this->m_expansionList);
        m_CGOp->setLHS(m_HelmOp);
    }

    void apply(Field<TData, FieldState::Phys> &in, Field<TData, FieldState::Coeff> &out)
    {
        // IProductWRT of RHS
        m_IProdOp->apply(in, m_rhs);

        // Handle Neumann BCs
        m_NeuBCOp->apply(out);

        // Handle Dirichlet BCs
        m_DirBCOp->apply(out);

        // Set precon
        m_CGOp->setPrecon(m_precon);

        // Solve for u_hat using Conjugate Gradient
        m_CGOp->apply(m_rhs, out);
    }

    void setLambda(const TData &lambda)
    {
        m_HelmOp->setLambda(lambda);
    }

    void setPrecon(const std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &precon)
    {
        m_precon = precon;
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

protected:
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProdOp;
    std::shared_ptr<OperatorNeuBndCond<TData>> m_NeuBCOp;
    std::shared_ptr<OperatorDirBndCond<TData>> m_DirBCOp;
    std::shared_ptr<OperatorConjGrad<TData>> m_CGOp;
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> m_precon;
    std::shared_ptr<OperatorHelm<TData>> m_HelmOp;
    Field<TData, FieldState::Coeff> m_rhs;
};

}