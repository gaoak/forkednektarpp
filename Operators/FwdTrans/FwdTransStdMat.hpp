#pragma once

#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorDirBndCond.hpp"
#include "Operators/OperatorFwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorMass.hpp"
#include "Operators/OperatorPrecon.hpp"
#include "Operators/OperatorRobBndCond.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorFwdTransImpl<TData, ImplStdMat> : public OperatorFwdTrans<TData>
{
public:
    OperatorFwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorFwdTrans<TData>(expansionList),
          m_rhs(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_dir(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList)))
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
        auto *dirptr = m_dir.GetStorage().GetCPUPtr();

        // IProductWRT of RHS
        m_IProdOp->apply(in, m_rhs);

        // Handle Dirichlet BCs
        m_DirBCOp->apply(m_dir);
        m_MassOp->apply(m_dir, out); // use out as temporary storage
        std::transform(rhsptr, rhsptr + nloc, outptr, rhsptr,
                       [](const TData &rhs, const TData &dir)
                       { return rhs - dir; });

        // Handle Robin BCs
        m_RobBCOp->apply(m_dir, m_rhs, true);

        // Solve for u_hat using Conjugate Gradient
        m_CGOp->apply(m_rhs, out);

        // Add Dirichlet BCs
        std::transform(outptr, outptr + nloc, dirptr, outptr,
                       [](const TData &x, const TData &dir)
                       { return x + dir; });
    }

    void setPrecon(
        const std::shared_ptr<OperatorPrecon<TData>> &precon) override
    {
        m_CGOp->setPrecon(precon);

        precon->configure(m_MassOp);
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
    Field<TData, FieldState::Coeff> m_dir;
};

} // namespace Nektar::Operators::detail
