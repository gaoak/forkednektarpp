///////////////////////////////////////////////////////////////////////////////
//
// File: MINRESOpImpl.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
#include "Operators/GlobalLinSysOps/LinearSolvers/MINRES/MINRESOp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class MINRESOpImpl : public MINRESOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    MINRESOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : MINRESOp<TData>(expansionList, components),
          m_q_A(Field<TData, FieldState::Coeff>(
              "MINRESOp q_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_w_A(Field<TData, FieldState::Coeff>(
              "MINRESOp w_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_p0_A(Field<TData, FieldState::Coeff>(
              "MINRESOp p0_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_p1_A(Field<TData, FieldState::Coeff>(
              "MINRESOp p1_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_v0_A(Field<TData, FieldState::Coeff>(
              "MINRESOp r0_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_v1_A(Field<TData, FieldState::Coeff>(
              "MINRESOp r1_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1))
    {
        auto session = expansionList->GetSession();

        // Set operators.
        m_math         = Math(ExecSpace::name);
        m_assmbScatrOp = std::make_unique<AssmbScatrOpImpl<ExecSpace, TData>>(
            this->m_expansionList, components);
        m_assmbScatrZeroDirOp =
            std::make_unique<AssmbScatrZeroDirOpImpl<ExecSpace, TData>>(
                this->m_expansionList, components);
        m_robBndCondOp = RobBndCondOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
        m_rowComm = session->GetComm()->GetRowComm();
        m_root    = m_rowComm->GetRank() == 0;

        // Set parameters.
        session->LoadParameter("NekLinSysMaxIterations", m_maxIter, 5000);
        session->LoadParameter("IterativeSolverTolerance", m_tol, 1.0E-09);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<MINRESOpImpl<ExecSpace, TData>>(expansionList,
                                                                components);
    }

protected:
    LibUtilities::CommSharedPtr m_rowComm = nullptr;
    std::unique_ptr<AssmbScatrOpImpl<ExecSpace, TData>> m_assmbScatrOp;
    std::unique_ptr<AssmbScatrZeroDirOpImpl<ExecSpace, TData>>
        m_assmbScatrZeroDirOp;
    bool m_root;

    Math m_math;

    std::shared_ptr<RobBndCondOp<TData>> m_robBndCondOp;

    Field<TData, FieldState::Coeff> m_q_A;
    Field<TData, FieldState::Coeff> m_w_A;
    Field<TData, FieldState::Coeff> m_p0_A;
    Field<TData, FieldState::Coeff> m_p1_A;
    Field<TData, FieldState::Coeff> m_v0_A;
    Field<TData, FieldState::Coeff> m_v1_A;

    TData m_tol            = 0.0;
    unsigned int m_maxIter = 0;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Based on the preconditioned MINRES algorithm (Algorithm 12, p. 58) in
        // "Preconditioning iterative methods for PDE constrained optimization",
        // T. Rees, 2010. and on the MINRES algorithm (Fig. 6.9, p.86) in
        // "Iterative Krylov Methods for Large Linear Systems", by Henk A. van
        // der Vorst, 2003.

        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, alpha, eta;
        TData alpha1, alpha2, alpha3, delta;
        TData gamma0, gamma1, sigma0, sigma1, beta0, beta1;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);
        m_p1_A.template Initialize<MemSpace>(0);

        // Calculate inital rhs magnitude.
        m_v0_A.template Copy<MemSpace>(in);
        m_assmbScatrOp->Apply(m_v0_A);
        rhsMagnitude = m_math.ddot(in, m_v0_A);
        m_rowComm->AllReduce(rhsMagnitude, Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // Iteration 0
        // Copy RHS into initial vector.
        m_v0_A.template Copy<MemSpace>(in);

        // Apply preconditioner
        m_assmbScatrZeroDirOp->Apply(m_v0_A, m_w_A);
        this->m_precon->Apply(m_w_A, m_w_A);
        beta1 = m_math.ddot(m_v0_A, m_w_A);
        m_rowComm->AllReduce(beta1, Nektar::LibUtilities::ReduceSum);
        beta1 = std::sqrt(beta1);

        // If the input residual is less than tolerance then skip solve.
        if (beta1 < m_tol * std::sqrt(rhsMagnitude))
        {
            return;
        }

        // Iteration >= 1
        eta    = beta1;
        gamma1 = gamma0 = 1.0;
        sigma1 = sigma0 = 0.0;
        while (true)
        {
            if (totalIterations > m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            mul<ExecSpace>(1.0 / beta1, m_v0_A, m_v0_A);
            mul<ExecSpace>(1.0 / beta1, m_w_A, m_w_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_w_A, m_q_A);
            m_robBndCondOp->Apply(m_w_A, m_q_A);

            // <w_{k+1}, q_{k+1}>
            alpha = m_math.ddot(m_w_A, m_q_A);
            m_rowComm->AllReduce(alpha, LibUtilities::ReduceSum);

            // Update search vector.
            if (totalIterations > 0)
            {
                mul<ExecSpace>(-beta1, m_v1_A, m_v1_A);
                daxpy<ExecSpace>(-alpha, m_v0_A, m_v1_A, m_v1_A);
            }
            else
            {
                mul<ExecSpace>(-alpha, m_v0_A, m_v1_A);
            }
            add<ExecSpace>(m_v1_A, m_q_A, m_v1_A);

            // Apply preconditioner.
            m_assmbScatrZeroDirOp->Apply(m_v1_A, m_q_A);
            this->m_precon->Apply(m_q_A, m_q_A);

            // Update coefficients.
            beta0 = beta1;
            beta1 = m_math.ddot(m_v1_A, m_q_A);
            m_rowComm->AllReduce(beta1, LibUtilities::ReduceSum);
            beta1 = std::sqrt(beta1);

            delta  = gamma1 * alpha - gamma0 * sigma1 * beta0;
            alpha1 = std::sqrt(delta * delta + beta1 * beta1);
            alpha2 = sigma1 * alpha + gamma0 * gamma1 * beta0;
            alpha3 = sigma0 * beta0;

            gamma0 = gamma1;
            gamma1 = delta / alpha1;
            sigma0 = sigma1;
            sigma1 = beta1 / alpha1;

            // Update solution.
            if (totalIterations == 0)
            {
                // m_p1_A, m_p0_A = 0
                mul<ExecSpace>(1.0 / alpha1, m_w_A, m_p0_A);
            }
            else if (totalIterations == 1)
            {
                // m_p0_A = 0
                mul<ExecSpace>(1.0 / alpha1, m_w_A, m_p0_A);
                daxpy<ExecSpace>(-alpha2 / alpha1, m_p1_A, m_p0_A, m_p0_A);
            }
            else
            {
                mul<ExecSpace>(-alpha3 / alpha1, m_p0_A, m_p0_A);
                daxpy<ExecSpace>(-alpha2 / alpha1, m_p1_A, m_p0_A, m_p0_A);
                daxpy<ExecSpace>(1.0 / alpha1, m_w_A, m_p0_A, m_p0_A);
            }
            daxpy<ExecSpace>(gamma1 * eta, m_p0_A, out, out);

            eta *= -sigma1;

            ++totalIterations;

            // Test if norm is within tolerance.
            if (std::abs(eta) < m_tol * std::sqrt(rhsMagnitude))
            {
                if (m_root)
                {
                    std::cout << "iterations: " << totalIterations
                              << " eta: " << std::abs(eta)
                              << " rhs_mag: " << rhsMagnitude << std::endl;
                }
                break;
            }

            // Swap storage for next iteration.
            std::swap(m_w_A, m_q_A);
            std::swap(m_v1_A, m_v0_A);
            std::swap(m_p0_A, m_p1_A);
        }
    }
};

} // namespace Nektar::Operators::detail
