#pragma once

#include "Operators/OperatorHelmSolve.hpp"

#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorDirBndCond.hpp"
#include "Operators/OperatorHelmholtz.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorLinear.hpp"
#include "Operators/OperatorNeuBndCond.hpp"
#include "Operators/OperatorPrecon.hpp"
#include "Operators/OperatorRobBndCond.hpp"

using namespace Nektar;
using namespace Nektar::Operators;
using namespace Nektar::MultiRegions;

using vec_t = tinysimd::simd<double>;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorHelmSolveImpl<TData, ImplStdMat> : public OperatorHelmSolve<TData>
{
public:
    OperatorHelmSolveImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmSolve<TData>(expansionList),
          m_rhs(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment)),
          m_dir(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment))
    {
        m_IProdOp = IProductWRTBase<TData>::create(this->m_expansionList);
        m_DirBCOp = DirBndCond<TData>::create(this->m_expansionList);
        m_NeuBCOp = NeuBndCond<TData>::create(this->m_expansionList);
        m_RobBCOp = RobBndCond<TData>::create(this->m_expansionList);
        m_HelmOp  = Helmholtz<TData>::create(this->m_expansionList);
        m_CGOp    = ConjGrad<TData>::create(this->m_expansionList);
        m_CGOp->setLHS(m_HelmOp);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        size_t nloc  = out.GetStorage().size();
        auto *outptr = out.GetStorage().GetCPUPtr();
        auto *rhsptr = m_rhs.GetStorage().GetCPUPtr();
        auto *dirptr = m_dir.GetStorage().GetCPUPtr();

        // IProductWRT of RHS
        m_IProdOp->apply(in, m_rhs);
        std::transform(rhsptr, rhsptr + nloc, rhsptr, std::negate<TData>());

        // Handle Neumann BCs on RHS
        m_NeuBCOp->apply(m_rhs);

        // Handle Dirichlet BCs
        m_DirBCOp->apply(m_dir);
        m_HelmOp->apply(m_dir, out); // use out as temporary storage
        std::transform(
            rhsptr, rhsptr + nloc, outptr, rhsptr,
            [](const TData &rhs, const TData &dir) { return rhs - dir; });

        // Handle Robin BCs
        m_RobBCOp->apply(m_dir, m_rhs, true);

        // Solve for u_hat using Conjugate Gradient
        m_CGOp->apply(m_rhs, out);

        // Add Dirichlet BCs
        std::transform(
            outptr, outptr + nloc, dirptr, outptr,
            [](const TData &x, const TData &dir) { return x + dir; });
    }

    void setLambda(const TData &lambda)
    {
        m_HelmOp->setLambda(lambda);
    }

    void setPrecon(
        const std::shared_ptr<OperatorPrecon<TData>> &precon) override
    {
        precon->configure(m_HelmOp);

        m_CGOp->setPrecon(precon);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorHelmSolveImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProdOp;
    std::shared_ptr<OperatorDirBndCond<TData>> m_DirBCOp;
    std::shared_ptr<OperatorNeuBndCond<TData>> m_NeuBCOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_RobBCOp;
    std::shared_ptr<OperatorHelmholtz<TData>> m_HelmOp;
    std::shared_ptr<OperatorConjGrad<TData>> m_CGOp;
    Field<TData, FieldState::Coeff> m_rhs;
    Field<TData, FieldState::Coeff> m_dir;
};

} // namespace Nektar::Operators::detail
