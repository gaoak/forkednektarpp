///////////////////////////////////////////////////////////////////////////////
//
// File: IDRSOpImpl.hpp
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

#include "Operators/GlobalLinSysOps/LinearSolvers/IDRS/IDRSOp.hpp"

#include <iomanip>
#include <random>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class IDRSOpImpl : public IDRSOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IDRSOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
               const std::vector<std::string> &components)
        : IDRSOp<TData>(expansionList, components),
          m_v(Field<TData, FieldState::Coeff>(
              "IDRSOp v",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_w(Field<TData, FieldState::Coeff>(
              "IDRSOp t",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r(Field<TData, FieldState::Coeff>(
              "IDRSOp r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1))
    {
        this->template SetLinearSolver<ExecSpace>();

        auto session = this->m_expansionList->GetSession();

        // Set parameters.
        this->m_leftPreconditioner =
            session->DefinesParameter("LinSysLeftPrecon")
                ? session->GetParameter("LinSysLeftPrecon")
                : false;
        this->m_rightPreconditioner =
            session->DefinesParameter("LinSysRightPrecon")
                ? session->GetParameter("LinSysRightPrecon")
                : true;
        m_stage = session->DefinesParameter("IDRstage")
                      ? session->GetParameter("IDRstage")
                      : 4;

        // Set-up storage.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (unsigned int stage = 0; stage < m_stage; stage++)
        {
            m_U.push_back(Field<TData, FieldState::Coeff>(
                "IDRSOp U" + std::to_string(stage),
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));
            m_G.push_back(Field<TData, FieldState::Coeff>(
                "IDRSOp G" + std::to_string(stage),
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));
            m_P.push_back(Field<TData, FieldState::Coeff>(
                "IDRSOp P" + std::to_string(stage),
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));

            // Initialzie m_P with normalized random number.
            for (unsigned blk = 0; blk < m_P.back().GetBlocks().size(); ++blk)
            {
                auto &block = m_P.back().GetBlocks()[blk];
                auto ptr =
                    block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                std::generate(ptr,
                              ptr + block.GetNumComponents() * block.CompSize(),
                              [&]() { return dis(gen); });
            }
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<IDRSOpImpl<ExecSpace, TData>>(expansionList,
                                                              components);
    }

protected:
    std::vector<Field<TData, FieldState::Coeff>> m_P;
    std::vector<Field<TData, FieldState::Coeff>> m_U;
    std::vector<Field<TData, FieldState::Coeff>> m_G;
    Field<TData, FieldState::Coeff> m_v;
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_r;

    unsigned int m_stage = 0;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Implement IDR(s) iterative method as described in:
        //
        // Reference:
        // Van Gijzen, Martin B., and Peter Sonneveld. "Algorithm 913: An
        // elegant IDR (s) variant that efficiently exploits biorthogonality
        // properties." ACM Transactions on Mathematical Software (TOMS) 38, no.
        // 1 (2011): 1-19.

        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, eps;
        TData omega, omega0, omega1, rho, alpha, beta, kappa = 0.7;
        std::vector<TData> Phi(m_stage), gamma(m_stage);
        std::vector<std::vector<TData>> Mu(m_stage);

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrOp->Apply(m_r);
        rhsMagnitude = this->m_math.ddot(in, m_r);
        this->m_rowComm->AllReduce(rhsMagnitude,
                                   Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrZeroDirOp->Apply(m_r);

        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_r);
        }

        eps = this->m_math.ddot(in, m_r);
        this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        omega = 1.0;
        for (unsigned int k = 0; k < m_stage; k++)
        {
            Mu[k]    = std::vector<TData>(m_stage);
            Mu[k][k] = 1.0;
            m_G[k].template Initialize<MemSpace>(0.0);
            m_U[k].template Initialize<MemSpace>(0.0);
        }
        while (true)
        {
            // Compute Phi.
            for (unsigned int k = 0; k < m_stage; k++)
            {
                Phi[k] = this->m_math.ddot(m_P[k], m_r);
            }
            this->m_rowComm->AllReduce(Phi, Nektar::LibUtilities::ReduceSum);

            // Inner iteration.
            for (unsigned int k = 0; k < m_stage; k++)
            {
                if (totalIterations > this->m_maxIter)
                {
                    std::stringstream msg;
                    msg << "Exceeded max iterations: " << totalIterations;
                    WARNINGL0(false, msg.str());

                    return;
                }

                // Directly solve the small linear system.
                if (m_stage - k == 1)
                {
                    gamma[k] = Phi[k] / Mu[k][k];
                }
                else
                {
                    std::vector<std::vector<TData>> Mu_k(m_stage - k);
                    std::vector<TData> Phi_k(m_stage - k);
                    for (unsigned int i = 0; i < m_stage - k; i++)
                    {
                        Mu_k[i] = std::vector<TData>(m_stage - k);
                        std::copy(Mu[i + k].begin() + k, Mu[i + k].end(),
                                  Mu_k[i].data());
                    }
                    std::copy(Phi.begin() + k, Phi.end(), Phi_k.begin());
                    this->DirectSolve(Mu_k, Phi_k);
                    std::copy(Phi_k.begin(), Phi_k.end(), gamma.begin() + k);
                }

                // Compute m_v.
                if (totalIterations == 0)
                {
                    m_v.template Copy<MemSpace>(m_r);
                }
                else
                {
                    daxpy<ExecSpace>(-gamma[k], m_G[k], m_r, m_v);
                    for (unsigned int i = k + 1; i < m_stage; i++)
                    {
                        daxpy<ExecSpace>(-gamma[i], m_G[i], m_v, m_v);
                    }
                }

                // Apply preconditioner.
                if (this->m_rightPreconditioner)
                {
                    this->m_precon->Apply(m_v, m_v);
                }

                // Compute new U.
                if (totalIterations == 0)
                {
                    mul<ExecSpace>(omega, m_v, m_U[k]);
                }
                else
                {
                    mul<ExecSpace>(gamma[k], m_U[k], m_U[k]);
                    for (unsigned int i = k + 1; i < m_stage; i++)
                    {
                        daxpy<ExecSpace>(gamma[i], m_U[i], m_U[k], m_U[k]);
                    }
                    daxpy<ExecSpace>(omega, m_v, m_U[k], m_U[k]);
                }

                // Perform the method-specific matrix-vector multiply operation.
                this->m_lhs->Apply(m_U[k], m_G[k]);
                this->m_robBndCondOp->Apply(m_U[k], m_G[k]);
                this->m_assmbScatrZeroDirOp->Apply(m_G[k]);
                if (this->m_leftPreconditioner)
                {
                    this->m_precon->Apply(m_G[k], m_G[k]);
                }

                // Bi-Orthogonalize the basis vectors:
                for (unsigned int i = 0; i < k; i++)
                {
                    alpha = this->m_math.ddot(m_P[i], m_G[k]);
                    this->m_rowComm->AllReduce(alpha,
                                               Nektar::LibUtilities::ReduceSum);
                    alpha /= Mu[i][i];
                    daxpy<ExecSpace>(-alpha, m_G[i], m_G[k], m_G[k]);
                    daxpy<ExecSpace>(-alpha, m_U[i], m_U[k], m_U[k]);
                }

                // Update Mu.
                for (unsigned int i = k; i < m_stage; i++)
                {
                    TData mu = this->m_math.ddot(m_P[i], m_G[k]);
                    this->m_rowComm->AllReduce(mu,
                                               Nektar::LibUtilities::ReduceSum);
                    Mu[i][k] = mu;
                }

                if (Mu[k][k] == 0.0)
                {
                    std::stringstream msg;
                    msg << "Convergence breakdown: ";
                    WARNINGL0(false, msg.str());

                    return;
                }

                // Make m_r orthogonal to m_G.
                beta = Phi[k] / Mu[k][k];
                daxpy<ExecSpace>(-beta, m_G[k], m_r, m_r);
                daxpy<ExecSpace>(beta, m_U[k], out, out);

                eps = this->m_math.ddot(m_r, m_r);
                this->m_rowComm->AllReduce(eps,
                                           Nektar::LibUtilities::ReduceSum);

                ++totalIterations;

                // Test if norm is within tolerance.
                if (eps < this->m_tol * this->m_tol * rhsMagnitude)
                {
                    if (this->m_root)
                    {
                        std::cout
                            << this->name
                            << " iterations made = " << totalIterations
                            << " using tolerance of " << this->m_tol
                            << " error = " << std::sqrt(eps / rhsMagnitude)
                            << " rhs_mag = " << std::sqrt(rhsMagnitude)
                            << std::endl;
                    }
                    return;
                }

                // Update Phi.
                if (k < m_stage - 1)
                {
                    for (unsigned int i = 0; i < k + 1; i++)
                    {
                        Phi[i] = 0.0;
                    }
                    for (unsigned int i = k + 1; i < m_stage; i++)
                    {
                        Phi[i] -= beta * Mu[i][k];
                    }
                }
            }

            ++totalIterations;

            // Test if norm is within tolerance.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                if (this->m_root)
                {
                    std::cout << this->name
                              << " iterations made = " << totalIterations
                              << " using tolerance of " << this->m_tol
                              << " error = " << std::sqrt(eps / rhsMagnitude)
                              << " rhs_mag = " << std::sqrt(rhsMagnitude)
                              << std::endl;
                }
                return;
            }

            if (totalIterations > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp = (this->m_rightPreconditioner) ? m_v : m_r;
            if (this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_r, tmp);
            }
            this->m_lhs->Apply(tmp, m_w);
            this->m_robBndCondOp->Apply(tmp, m_w);
            this->m_assmbScatrZeroDirOp->Apply(m_w);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_w, m_w);
            }

            // Update coefficients.
            omega0 = this->m_math.ddot(m_w, m_r);
            this->m_rowComm->AllReduce(omega0, LibUtilities::ReduceSum);
            omega1 = this->m_math.ddot(m_w, m_w);
            this->m_rowComm->AllReduce(omega1, LibUtilities::ReduceSum);
            omega = omega0 / omega1;
            rho   = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(rho, LibUtilities::ReduceSum);
            rho = std::abs(omega0 / (std::sqrt(omega1) * std::sqrt(rho)));
            if (rho < kappa)
            {
                omega *= kappa / rho;
            }

            // Update solution.
            daxpy<ExecSpace>(omega, tmp, out, out);
            daxpy<ExecSpace>(-omega, m_w, m_r, m_r);

            // Update residual norm.
            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);
        }
    }
};
} // namespace Nektar::Operators::detail
