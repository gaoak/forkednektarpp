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
class OperatorConjGradImpl : public OperatorConjGrad<TData>
{
public:
    OperatorConjGradImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorConjGrad<TData>(std::move(expansionList))
    {
        m_assmbScatr = AssmbScatr<TData>::create(this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // these values should be referenced from Nektar
        TData tol      = 1.e-6;  // ** CHANGE THIS **
        size_t maxIter = 100000; // ** CHANGE THIS **

        // get number of local coeffs (=size of in/out fields)
        size_t nloc = in.GetStorage().size();

        // get block attributes (must be same as out.GetBlocks())
        auto blocks_coeff = in.GetBlocks();

        // create temporary fields
        auto w_A = Field<TData, FieldState::Coeff>::create(blocks_coeff);
        auto s_A = Field<TData, FieldState::Coeff>::create(blocks_coeff);
        auto p_A = Field<TData, FieldState::Coeff>::create(blocks_coeff);
        auto r_A = Field<TData, FieldState::Coeff>::create(blocks_coeff);
        auto q_A = Field<TData, FieldState::Coeff>::create(blocks_coeff);
        auto wk  = Field<TData, FieldState::Coeff>::create(blocks_coeff);

        // store pointers to temporary fields
        auto *p_in  = in.GetStorage().GetCPUPtr();
        auto *p_out = out.GetStorage().GetCPUPtr();
        auto *p_w_A = w_A.GetStorage().GetCPUPtr();
        auto *p_s_A = s_A.GetStorage().GetCPUPtr();
        auto *p_p_A = p_A.GetStorage().GetCPUPtr();
        auto *p_r_A = r_A.GetStorage().GetCPUPtr();
        auto *p_q_A = q_A.GetStorage().GetCPUPtr();
        auto *p_wk  = wk.GetStorage().GetCPUPtr();

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
        std::array<TData, 3> vExchange;

        // copy RHS into initial residual
        std::copy(p_in, p_in + nloc, p_r_A);

        // initial residual
        m_assmbScatr->apply(r_A, wk);

        vExchange[2] = std::inner_product(p_wk, p_wk + nloc, p_wk, 0.);
        // m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        // calculate rhs magnitude
        rhsMagnitude = 0.;
        std::for_each(p_in, p_in + nloc,
                      [&rhsMagnitude](const TData &x)
                      { rhsMagnitude += x * x; });
        rhsMagnitude = std::sqrt(rhsMagnitude);

        eps = vExchange[2];
        if (eps < tol * tol * rhsMagnitude)
            return;

        totalIterations = 0;

        m_precon->apply(r_A, w_A);
        m_LHS->apply(w_A, s_A);

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
            if (k >= maxIter)
            {
                std::cout << "Exceeded max iterations\n";
                return;
            }

            // Compute new search direction p_k, q_k
            // Vmath::Svtvp(nLocal, beta, p_A, 1, w_A, 1, p_A, 1);
            // Vmath::Svtvp(nLocal, beta, q_A, 1, s_A, 1, q_A, 1);
            std::transform(p_p_A, p_p_A + nloc, p_w_A, p_p_A,
                           [&](const TData &pElem, const TData &wElem)
                           { return beta * pElem + wElem; });
            std::transform(p_q_A, p_q_A + nloc, p_s_A, p_q_A,
                           [&](const TData &qElem, const TData &sElem)
                           { return beta * qElem + sElem; });

            // Update solution x_{k+1}
            // Vmath::Svtvp(nLocal, alpha, p_A, 1, pOutput, 1, pOutput, 1);
            std::transform(p_p_A, p_p_A + nloc, p_out, p_out,
                           [&](const TData &pElem, const TData &xElem)
                           { return alpha * pElem + xElem; });

            // Update residual vector r_{k+1}
            // Vmath::Svtvp(nLocal, -alpha, q_A, 1, r_A, 1, r_A, 1);
            std::transform(p_q_A, p_q_A + nloc, p_r_A, p_r_A,
                           [&](const TData &qElem, const TData &rElem)
                           { return -alpha * qElem + rElem; });

            // Apply preconditioner
            // m_operator.DoNekSysPrecon(r_A, w_A, true);
            m_precon->apply(r_A, w_A);

            // Perform the method-specific matrix-vector multiply operation.
            // m_operator.DoNekSysLhsEval(w_A, s_A);
            m_LHS->apply(w_A, s_A);

            // <r_{k+1}, w_{k+1}>
            // vExchange[0] = Vmath::Dot(nLocal, r_A, w_A);
            vExchange[0] = std::inner_product(p_r_A, p_r_A + nloc, p_w_A, 0.);

            // <s_{k+1}, w_{k+1}>
            // vExchange[1] = Vmath::Dot(nLocal, s_A, w_A);
            vExchange[1] = std::inner_product(p_s_A, p_s_A + nloc, p_w_A, 0.);

            // <r_{k+1}, r_{k+1}>
            // m_operator.assembleScatter(r_A, wk, true);
            // vExchange[2] = Vmath::Dot(nLocal, wk, r_A);

            m_assmbScatr->apply(r_A, wk); // Assembly (communication)

            vExchange[2] = std::inner_product(p_wk, p_wk + nloc, p_wk, 0.);

            // Perform inner-product exchanges
            // m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

            // (communication)
            rho_new = vExchange[0];
            mu      = vExchange[1];
            eps     = vExchange[2];

            std::cout << "Iteration " << k << " -- eps = " << eps << "\n";

            totalIterations++;

            // Test if norm is within tolerance
            if (eps < tol * tol * rhsMagnitude)
                break;
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
        return std::make_unique<OperatorConjGradImpl<TData>>(expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatr;
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>>
        m_LHS;
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>>
        m_precon;
};

} // namespace Nektar::Operators::detail
