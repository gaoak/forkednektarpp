#pragma once

#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorDirBndCond.hpp"
#include "Operators/OperatorFwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorMass.hpp"
#include "Operators/OperatorPrecon.hpp"
#include "Operators/OperatorRobBndCond.hpp"

using vec_t = tinysimd::simd<double>;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorFwdTransImpl<TData, ImplStdMat> : public OperatorFwdTrans<TData>
{
public:
    OperatorFwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorFwdTrans<TData>(expansionList),
          m_rhs(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment)),
          m_tmp(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment))
    {
        m_MassOp  = Mass<TData>::create(this->m_expansionList);
        m_DirBCOp = DirBndCond<TData>::create(this->m_expansionList);
        m_RobBCOp = RobBndCond<TData>::create(this->m_expansionList);
        m_IProdOp = IProductWRTBase<TData>::create(this->m_expansionList);
        m_CGOp    = ConjGrad<TData>::create(this->m_expansionList);
        m_CGOp->setLHS(m_MassOp);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        size_t nloc  = out.GetStorage().size();
        auto *outptr = out.GetStorage().GetCPUPtr();
        auto *rhsptr = m_rhs.GetStorage().GetCPUPtr();
        auto *tmpptr = m_tmp.GetStorage().GetCPUPtr();

        // IProductWRT of RHS
        m_IProdOp->apply(in, m_rhs);

        // Handle Dirichlet BCs
        m_DirBCOp->apply(out);
        m_MassOp->apply(out, m_tmp);
        std::transform(
            rhsptr, rhsptr + nloc, tmpptr, rhsptr,
            [](const TData &rhs, const TData &dir) { return rhs - dir; });

        // Handle Robin BCs
        m_RobBCOp->apply(out, m_rhs, true);

        // Solve for u_hat using Conjugate Gradient
        m_CGOp->apply(m_rhs, m_tmp);

        // Add Dirichlet BCs
        std::transform(
            outptr, outptr + nloc, tmpptr, outptr,
            [](const TData &x, const TData &diff) { return x + diff; });
    }

    void setPrecon(
        const std::shared_ptr<OperatorPrecon<TData>> &precon) override
    {
        precon->configure(m_MassOp);

        m_CGOp->setPrecon(precon);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorFwdTransImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProdOp;
    std::shared_ptr<OperatorDirBndCond<TData>> m_DirBCOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_RobBCOp;
    std::shared_ptr<OperatorMass<TData>> m_MassOp;
    std::shared_ptr<OperatorConjGrad<TData>> m_CGOp;
    Field<TData, FieldState::Coeff> m_rhs;
    Field<TData, FieldState::Coeff> m_tmp;
};

} // namespace Nektar::Operators::detail
