#include "Operators/OperatorConjGrad.hpp"

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <memory>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ContField.h>

#include "Operators/OperatorAssmbScatr.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorConjGradImpl<TData, ImplStdMat> : public OperatorConjGrad<TData>
{
public:
    OperatorConjGradImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorConjGrad<TData>(expansionList),
          m_w_A(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_s_A(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_p_A(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_r_A(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_q_A(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_wk(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto assmbMap = contfield->GetLocalToGlobalMap();
        m_tol         = assmbMap->GetIterativeTolerance();
        m_maxIter     = assmbMap->GetMaxIterations();
        m_assmbScatr  = AssmbScatr<TData>::create(this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // get number of local coeffs (=size of in/out fields)
        size_t nloc = in.GetStorage().size();

        // store pointers to temporary fields
        auto *p_in  = in.GetStorage().GetCPUPtr();
        auto *p_out = out.GetStorage().GetCPUPtr();
        auto *p_w_A = m_w_A.GetStorage().GetCPUPtr();
        auto *p_s_A = m_s_A.GetStorage().GetCPUPtr();
        auto *p_p_A = m_p_A.GetStorage().GetCPUPtr();
        auto *p_r_A = m_r_A.GetStorage().GetCPUPtr();
        auto *p_q_A = m_q_A.GetStorage().GetCPUPtr();
        auto *p_wk  = m_wk.GetStorage().GetCPUPtr();

        // set the fields to zero
        std::fill(p_out, p_out + nloc, 0.);
        std::fill(p_w_A, p_w_A + nloc, 0.);
        std::fill(p_s_A, p_s_A + nloc, 0.);
        std::fill(p_p_A, p_p_A + nloc, 0.);
        std::fill(p_q_A, p_q_A + nloc, 0.);
        std::fill(p_wk, p_wk + nloc, 0.);

        size_t k;
        size_t totalIterations;
        TData rhsMagnitude;
        TData alpha;
        TData beta;
        TData rho;
        TData rho_new;
        TData mu;
        TData eps;
        TData min_resid;
        std::array<TData, 3> vExchange{0.0, 0.0, 0.0};

        // copy RHS into initial residual
        std::copy(p_in, p_in + nloc, p_r_A);

        // initial residual
        m_assmbScatr->apply(m_r_A, m_wk);
        vExchange[2] = std::inner_product(p_wk, p_wk + nloc, p_r_A, 0.);
        // m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        eps = vExchange[2];

        // calculate rhs magnitude
        m_assmbScatr->apply(in, m_wk);
        rhsMagnitude = std::inner_product(p_in, p_in + nloc, p_wk, 0.);
        // m_Comm->AllReduce(rhsMagnitude, Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        totalIterations = 0;

        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        m_precon->apply(m_r_A, m_w_A);
        m_LHS->apply(m_w_A, m_s_A);

        k = 0;

        vExchange[0] = std::inner_product(p_r_A, p_r_A + nloc, p_w_A, 0.);
        vExchange[1] = std::inner_product(p_s_A, p_s_A + nloc, p_w_A, 0.);
        // m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        rho             = vExchange[0];
        mu              = vExchange[1];
        min_resid       = rhsMagnitude;
        beta            = 0.;
        alpha           = rho / mu;
        totalIterations = 1;

        while (true)
        {
            if (k >= m_maxIter)
            {
                std::cout << "Exceeded max iterations\n";
                return;
            }

            // Compute new search direction p_k, q_k
            std::transform(p_p_A, p_p_A + nloc, p_w_A, p_p_A,
                           [&beta](const TData &pElem, const TData &wElem)
                           { return beta * pElem + wElem; });
            std::transform(p_q_A, p_q_A + nloc, p_s_A, p_q_A,
                           [&beta](const TData &qElem, const TData &sElem)
                           { return beta * qElem + sElem; });

            // Update solution x_{k+1}
            std::transform(p_p_A, p_p_A + nloc, p_out, p_out,
                           [&alpha](const TData &pElem, const TData &xElem)
                           { return alpha * pElem + xElem; });

            // Update residual vector r_{k+1}
            std::transform(p_q_A, p_q_A + nloc, p_r_A, p_r_A,
                           [&alpha](const TData &qElem, const TData &rElem)
                           { return -alpha * qElem + rElem; });

            // Apply preconditioner
            m_precon->apply(m_r_A, m_w_A);

            // Perform the method-specific matrix-vector multiply operation.
            m_LHS->apply(m_w_A, m_s_A);

            // <r_{k+1}, w_{k+1}>
            vExchange[0] = std::inner_product(p_r_A, p_r_A + nloc, p_w_A, 0.);
            // <s_{k+1}, w_{k+1}>
            vExchange[1] = std::inner_product(p_s_A, p_s_A + nloc, p_w_A, 0.);
            // <r_{k+1}, r_{k+1}>
            m_assmbScatr->apply(m_r_A, m_wk); // Assembly (communication)
            vExchange[2] = std::inner_product(p_wk, p_wk + nloc, p_r_A, 0.);

            // Perform inner-product exchanges
            // m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

            // (communication)
            rho_new = vExchange[0];
            mu      = vExchange[1];
            eps     = vExchange[2];

            std::cout << "Iteration " << k << " -- eps = " << eps << "\n";

            totalIterations++;

            // Test if norm is within tolerance
            if (eps < m_tol * m_tol * rhsMagnitude)
            {
                break;
            }
            min_resid = std::min(min_resid, eps);

            // Compute search direction and solution coefficients
            beta  = rho_new / rho;
            alpha = rho_new / (mu - rho_new * beta / alpha);
            rho   = rho_new;
            k++;
        }
    }

    void setLHS(const std::shared_ptr<OperatorLinear<TData, FieldState::Coeff,
                                                     FieldState::Coeff>> &ptr)
    {
        m_LHS = ptr;
    }

    void setPrecon(
        const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &ptr)
    {
        m_precon = ptr;
    }

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorConjGradImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatr;
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>>
        m_LHS;
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>>
        m_precon;
    Field<TData, FieldState::Coeff> m_w_A;
    Field<TData, FieldState::Coeff> m_s_A;
    Field<TData, FieldState::Coeff> m_p_A;
    Field<TData, FieldState::Coeff> m_r_A;
    Field<TData, FieldState::Coeff> m_q_A;
    Field<TData, FieldState::Coeff> m_wk;
    TData m_tol;
    size_t m_maxIter;
};

} // namespace Nektar::Operators::detail
