#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <memory>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ContField.h>

#include "Operators/OperatorAssmbScatr.hpp"
#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorRobBndCond.hpp"

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
          m_r_A(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_wk(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        contfield->GetSession()->LoadParameter("NekLinSysMaxIterations",
                                               m_maxIter, 5000);
        contfield->GetSession()->LoadParameter("IterativeSolverTolerance",
                                               m_tol, 1.0E-09);
        m_rowComm    = contfield->GetSession()->GetComm()->GetRowComm();
        m_nloc       = contfield->GetLocalToGlobalMap()->GetNumLocalCoeffs();
        m_assmbScatr = AssmbScatr<TData>::create(this->m_expansionList);
        m_robBndCond = RobBndCond<TData>::create(this->m_expansionList);
        m_p_A        = std::make_unique<TData[]>(m_nloc);
        m_q_A        = std::make_unique<TData[]>(m_nloc);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        if (m_rowComm->GetSize() > 1 && m_rowComm->GetRank() == 0)
        {
            std::cout << "Solve ConjGrad with " << m_rowComm->GetSize()
                      << " processors" << std::endl;
        }

        // Store pointers to temporary fields
        Array<OneD, TData> inArr(m_nloc, 0.0);
        Array<OneD, TData> outArr(m_nloc, 0.0);
        auto *p_in  = inArr.get();
        auto *p_out = outArr.get();
        auto *p_w_A = m_w_A.GetStorage().GetCPUPtr();
        auto *p_s_A = m_s_A.GetStorage().GetCPUPtr();
        auto *p_r_A = m_r_A.GetStorage().GetCPUPtr();
        auto *p_wk  = m_wk.GetStorage().GetCPUPtr();
        auto *p_p_A = m_p_A.get();
        auto *p_q_A = m_q_A.get();

        // Set the fields to zero
        std::fill(p_w_A, p_w_A + m_nloc, 0.0);
        std::fill(p_s_A, p_s_A + m_nloc, 0.0);
        std::fill(p_wk, p_wk + m_nloc, 0.0);
        std::fill(p_p_A, p_p_A + m_nloc, 0.0);
        std::fill(p_q_A, p_q_A + m_nloc, 0.0);

        // Convergence parameters
        size_t totalIterations = 0;
        TData rhsMagnitude;
        TData alpha;
        TData beta;
        TData rho;
        TData rho_new;
        TData mu;
        TData eps;
        Array<OneD, TData> vExchange(3, 0.0);

        // Copy data from input field
        auto *inarrptr = inArr.data();
        auto *inptr    = in.GetStorage().GetCPUPtr();

        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = in.GetBlocks()[block_idx].block_size;
            auto nElmts = in.GetBlocks()[block_idx].num_elements;
            auto nmTot  = in.GetBlocks()[block_idx].num_pts;

            std::copy(inptr, inptr + nElmts * nmTot, inarrptr);

            inarrptr += nElmts * nmTot;
            inptr += nSize;
        }

        // Copy RHS into initial residual
        std::copy(p_in, p_in + m_nloc, p_r_A);

        // Assembly (communication)
        m_assmbScatr->apply(m_r_A, m_wk, true);
        vExchange[2] = std::inner_product(p_wk, p_wk + m_nloc, p_r_A, 0.0);

        // Perform inner-product exchanges
        m_rowComm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        eps = vExchange[2];

        // Calculate rhs magnitude
        m_assmbScatr->apply(m_r_A, m_wk);
        rhsMagnitude = std::inner_product(p_in, p_in + m_nloc, p_wk, 0.0);
        m_rowComm->AllReduce(rhsMagnitude, Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // If input residual is less than tolerance skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Apply preconditioner
        this->m_precon->apply(m_r_A, m_w_A);

        // Perform the method-specific matrix-vector multiply operation.
        this->m_LHS->apply(m_w_A, m_s_A);

        // Apply Robin BCs
        m_robBndCond->apply(m_w_A, m_s_A);

        vExchange[0] = std::inner_product(p_r_A, p_r_A + m_nloc, p_w_A, 0.0);
        vExchange[1] = std::inner_product(p_s_A, p_s_A + m_nloc, p_w_A, 0.0);

        m_rowComm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        rho             = vExchange[0];
        mu              = vExchange[1];
        beta            = 0.0;
        alpha           = rho / mu;
        totalIterations = 1;
        while (true)
        {
            if (totalIterations > m_maxIter)
            {
                std::cout << "Exceeded max iterations\n";
                return;
            }

            // Compute new search direction p_k
            std::transform(p_p_A, p_p_A + m_nloc, p_w_A, p_p_A,
                           [&beta](const TData &pElem, const TData &wElem) {
                               return beta * pElem + wElem;
                           });

            // Compute new search direction q_k
            std::transform(p_q_A, p_q_A + m_nloc, p_s_A, p_q_A,
                           [&beta](const TData &qElem, const TData &sElem) {
                               return beta * qElem + sElem;
                           });

            // Update solution x_{k+1}
            std::transform(p_p_A, p_p_A + m_nloc, p_out, p_out,
                           [&alpha](const TData &pElem, const TData &xElem) {
                               return alpha * pElem + xElem;
                           });

            // Update residual vector r_{k+1}
            std::transform(p_q_A, p_q_A + m_nloc, p_r_A, p_r_A,
                           [&alpha](const TData &qElem, const TData &rElem) {
                               return -alpha * qElem + rElem;
                           });

            // Apply preconditioner
            this->m_precon->apply(m_r_A, m_w_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_LHS->apply(m_w_A, m_s_A);

            // Apply Robin BCs
            m_robBndCond->apply(m_w_A, m_s_A);

            // <r_{k+1}, w_{k+1}>
            vExchange[0] =
                std::inner_product(p_r_A, p_r_A + m_nloc, p_w_A, 0.0);

            // <s_{k+1}, w_{k+1}>
            vExchange[1] =
                std::inner_product(p_s_A, p_s_A + m_nloc, p_w_A, 0.0);

            // <r_{k+1}, r_{k+1}>
            m_assmbScatr->apply(m_r_A, m_wk, true);
            vExchange[2] = std::inner_product(p_wk, p_wk + m_nloc, p_r_A, 0.0);

            // Perform inner-product exchanges
            m_rowComm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

            rho_new = vExchange[0];
            mu      = vExchange[1];
            eps     = vExchange[2];

            std::cout << "Iteration " << totalIterations << " -- eps = " << eps
                      << "\n";

            totalIterations++;

            // Test if norm is within tolerance
            if (eps < m_tol * m_tol * rhsMagnitude)
            {
                break;
            }

            // Compute search direction and solution coefficients
            beta  = rho_new / rho;
            alpha = rho_new / (mu - rho_new * beta / alpha);
            rho   = rho_new;
        }

        // Copy data to output field
        auto *outarrptr = outArr.data();
        auto *outptr    = out.GetStorage().GetCPUPtr();

        for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = out.GetBlocks()[block_idx].block_size;
            auto nElmts = out.GetBlocks()[block_idx].num_elements;
            auto nmTot  = out.GetBlocks()[block_idx].num_pts;

            std::copy(outarrptr, outarrptr + nElmts * nmTot, outptr);

            outarrptr += nElmts * nmTot;
            outptr += nSize;
        }
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
    LibUtilities::CommSharedPtr m_rowComm;
    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatr;
    std::shared_ptr<OperatorRobBndCond<TData>> m_robBndCond;
    Field<TData, FieldState::Coeff> m_w_A;
    Field<TData, FieldState::Coeff> m_s_A;
    Field<TData, FieldState::Coeff> m_r_A;
    Field<TData, FieldState::Coeff> m_wk;
    std::unique_ptr<TData[]> m_q_A;
    std::unique_ptr<TData[]> m_p_A;
    TData m_tol;
    size_t m_nloc;
    size_t m_maxIter;
};

} // namespace Nektar::Operators::detail
