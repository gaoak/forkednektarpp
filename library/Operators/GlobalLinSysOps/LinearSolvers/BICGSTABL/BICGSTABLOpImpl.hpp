///////////////////////////////////////////////////////////////////////////////
//
// File: BICGSTABLOpImpl.hpp
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
#include "Operators/GlobalLinSysOps/LinearSolvers/BICGSTABL/BICGSTABLOp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class BICGSTABLOpImpl : public BICGSTABLOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BICGSTABLOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : BICGSTABLOp<TData>(expansionList, components),
          m_w(Field<TData, FieldState::Coeff>(
              "BICGSTABLOp w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_acc(Field<TData, FieldState::Coeff>(
              "BICGSTABLOp acc",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_rhs(Field<TData, FieldState::Coeff>(
              "BICGSTABLOp rhs",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_rtilde(Field<TData, FieldState::Coeff>(
              "BICGSTABLOp rtilde",
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
        int leftPreconditioner  = 0;
        int rightPreconditioner = 0;
        session->LoadParameter("NekLinSysMaxIterations", m_maxIter, 5000);
        session->LoadParameter("IterativeSolverTolerance", m_tol, 1.0E-09);
        session->LoadParameter("BICGSTABLstage", m_stage, 4);
        session->LoadParameter("BICGSTABLLeftPrecon", leftPreconditioner, 0);
        session->LoadParameter("BICGSTABLRightPrecon", rightPreconditioner, 0);
        m_leftPreconditioner  = leftPreconditioner;
        m_rightPreconditioner = rightPreconditioner;

        for (unsigned int stage = 0; stage <= m_stage; stage++)
        {
            m_r.push_back(Field<TData, FieldState::Coeff>(
                "BICGSTABLOp r" + std::to_string(stage),
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));
            m_u.push_back(Field<TData, FieldState::Coeff>(
                "BICGSTABLOp u" + std::to_string(stage),
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));
        }

        m_vExchange =
            MemoryRegion<TData>((m_stage + 2) * (m_stage + 1) / 2, ePinned);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<BICGSTABLOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    LibUtilities::CommSharedPtr m_rowComm = nullptr;
    std::unique_ptr<AssmbScatrOpImpl<ExecSpace, TData>> m_assmbScatrOp;
    std::unique_ptr<AssmbScatrZeroDirOpImpl<ExecSpace, TData>>
        m_assmbScatrZeroDirOp;
    bool m_root;

    Math m_math;

    std::shared_ptr<RobBndCondOp<TData>> m_robBndCondOp;

    std::vector<Field<TData, FieldState::Coeff>> m_u;
    std::vector<Field<TData, FieldState::Coeff>> m_r;
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_acc;
    Field<TData, FieldState::Coeff> m_rhs;
    Field<TData, FieldState::Coeff> m_rtilde;

    MemoryRegion<TData> m_vExchange;

    TData m_tol                = 0.0;
    unsigned int m_maxIter     = 0;
    unsigned int m_stage       = 0;
    bool m_accurateUpdate      = true; // Flag for enhanced update
    bool m_leftPreconditioner  = false;
    bool m_rightPreconditioner = false;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Reference:
        // Sleijpen, Gerard LG, Henk A. Van der Vorst, and Diederik R. Fokkema.
        // "BiCGstab (l) and other hybrid Bi-CG methods." Numerical Algorithms
        // 7, no. 1 (1994): 75-109.
        //
        // Fokkema, Diederik R. Enhanced implementation of BiCGstab (l) for
        // solving linear systems of equations. Universiteit Utrecht.
        // Mathematisch Instituut, 1996.

        // Convergence parameters.
        unsigned int totalIterations = 0;
        unsigned int ldz             = m_stage + 1;
        TData rhsMagnitude, eps;
        TData delta = 0.01, kappa = 0.7;
        TData alpha, beta, rho, rho_new;
        TData omega, gamma, zeta0, zeta, kappa0, kappal;
        TData MaxResApprox, MaxResTrue;
        std::vector<std::vector<TData>> Zvec(ldz), wsp(ldz - 2);
        std::vector<TData> y0(ldz), yl(ldz), sol(ldz - 2);

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);

        // Calculate inital rhs magnitude.
        m_r[0].template Copy<MemSpace>(in);
        m_assmbScatrOp->Apply(m_r[0]);
        rhsMagnitude = m_math.ddot(in, m_r[0]);
        m_rowComm->AllReduce(rhsMagnitude, Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r[0].template Copy<MemSpace>(in);
        m_assmbScatrZeroDirOp->Apply(m_r[0]);

        if (m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r[0], m_r[0]);
        }

        eps = m_math.ddot(m_r[0], m_r[0]);
        m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        zeta0        = std::sqrt(eps);
        zeta         = zeta0;
        MaxResApprox = zeta0;
        MaxResTrue   = zeta0;
        alpha        = 0.0;
        rho          = 1.0;
        omega        = 1.0;
        m_u[0].template Initialize<MemSpace>(0);
        m_acc.template Initialize<MemSpace>(0);
        if (m_accurateUpdate)
        {
            m_rhs.template Copy<MemSpace>(m_r[0]);
        }
        else
        {
            m_rhs.template Copy<MemSpace>(in);
        }
        m_rtilde.template Copy<MemSpace>(m_r[0]);
        for (unsigned int i = 0; i < ldz; ++i)
        {
            Zvec[i] = std::vector<TData>(ldz);
        }
        for (unsigned int i = 0; i < ldz - 2; ++i)
        {
            wsp[i] = std::vector<TData>(ldz - 2);
        }
        while (true)
        {
            if (totalIterations > m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            //  --- CGS part ---
            rho *= -omega;
            for (unsigned int ii = 0; ii < m_stage; ii++)
            {
                // Update coefficients.
                rho_new = m_math.ddot(m_r[ii], m_rtilde);
                m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
                beta = alpha * (rho_new / rho);
                rho  = rho_new;

                if (rho_new == 0.0)
                {
                    std::stringstream msg;
                    msg << "Iteration diverged: " << totalIterations;
                    WARNINGL0(false, msg.str());

                    return;
                }

                // Update search vectors.
                for (unsigned int i = 0; i <= ii; i++)
                {
                    daxpy<ExecSpace>(-beta, m_u[i], m_r[i], m_u[i]);
                }

                // Perform the method-specific matrix-vector multiply operation.
                auto &tmp = (m_rightPreconditioner) ? m_w : m_u[ii];
                if (m_rightPreconditioner)
                {
                    this->m_precon->Apply(m_u[ii], tmp);
                }
                this->m_lhs->Apply(tmp, m_u[ii + 1]);
                m_robBndCondOp->Apply(tmp, m_u[ii + 1]);
                m_assmbScatrZeroDirOp->Apply(m_u[ii + 1]);
                if (m_leftPreconditioner)
                {
                    this->m_precon->Apply(m_u[ii + 1], m_u[ii + 1]);
                }

                // Update coefficients.
                alpha = m_math.ddot(m_u[ii + 1], m_rtilde);
                m_rowComm->AllReduce(alpha, LibUtilities::ReduceSum);

                if (alpha == 0.0)
                {
                    std::stringstream msg;
                    msg << "Iteration diverged: " << totalIterations;
                    WARNINGL0(false, msg.str());

                    return;
                }

                alpha = rho_new / alpha;

                // Update solution.
                daxpy<ExecSpace>(alpha, m_u[0], m_acc, m_acc);

                // Update residual.
                for (unsigned int i = 0; i <= ii; i++)
                {
                    daxpy<ExecSpace>(-alpha, m_u[i + 1], m_r[i], m_r[i]);
                }

                // Perform the method-specific matrix-vector multiply operation.
                auto &tmp2 = (m_rightPreconditioner) ? m_w : m_r[ii];
                if (m_rightPreconditioner)
                {
                    this->m_precon->Apply(m_r[ii], tmp2);
                }
                this->m_lhs->Apply(tmp2, m_r[ii + 1]);
                m_robBndCondOp->Apply(tmp2, m_r[ii + 1]);
                m_assmbScatrZeroDirOp->Apply(m_r[ii + 1]);
                if (m_leftPreconditioner)
                {
                    this->m_precon->Apply(m_r[ii + 1], m_r[ii + 1]);
                }

                // Test if norm is within tolerance.
                eps = m_math.ddot(m_r[0], m_r[0]);
                m_rowComm->AllReduce(eps, LibUtilities::ReduceSum);
                zeta = std::sqrt(eps);

                totalIterations++;

                // Test if norm is within tolerance.
                if (eps < m_tol * m_tol * rhsMagnitude)
                {
                    if (m_rightPreconditioner)
                    {
                        this->m_precon->Apply(m_acc, m_acc);
                    }
                    add<ExecSpace>(m_acc, out, out);

                    if (m_root)
                    {
                        std::cout << "iterations: " << totalIterations
                                  << " eps: " << std::sqrt(eps)
                                  << " rhs_mag: " << rhsMagnitude << std::endl;
                    }
                    return;
                }

                if (m_accurateUpdate)
                {
                    MaxResApprox = std::max(zeta, MaxResApprox);
                    MaxResTrue   = std::max(zeta, MaxResTrue);
                }
            }

            //  --- Polynomial part ---
            // Reset device memory.
            auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();
            for (unsigned int ii = 0, cnt = 0; ii <= m_stage; ++ii)
            {
                for (unsigned int i = 0; i <= ii; ++i, ++cnt)
                {
                    ddot<ExecSpace>(m_r[ii], m_r[i], exchange + cnt);
                }
            }
            m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                           Nektar::LibUtilities::ReduceSum);

            // Device-to-host copy.
            auto exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            for (unsigned int ii = 0, cnt = 0; ii <= m_stage; ++ii)
            {
                for (unsigned int i = 0; i <= ii; ++i, ++cnt)
                {
                    Zvec[ii][i] = exchangeHost[cnt];
                    if (i != ii)
                    {
                        Zvec[i][ii] = exchangeHost[cnt];
                    }
                }
            }

            if (m_stage == 1)
            {
                // Solve for y0.
                y0[0] = -1.0;
                y0[1] = Zvec[1][0] / Zvec[1][1];

                // Solve for yl.
                yl[0] = 0.0;
                yl[1] = 0.0;
            }
            else
            {
                // Solver for y0.
                std::copy_n(&Zvec[0][1], ldz - 2, &sol[0]);
                for (unsigned int i = 0; i < ldz - 2; ++i)
                {
                    std::copy_n(&Zvec[i + 1][1], ldz - 2, &wsp[i][0]);
                }
                DirectSolve(wsp, sol);
                y0[0]       = -1.0;
                y0[m_stage] = 0.0;
                std::copy_n(&sol[0], ldz - 2, &y0[1]);

                // Solver for yl.
                std::copy_n(&Zvec[ldz - 1][1], ldz - 2, &sol[0]);
                for (unsigned int i = 0; i < ldz - 2; ++i)
                {
                    std::copy_n(&Zvec[i + 1][1], ldz - 2, &wsp[i][0]);
                }
                DirectSolve(wsp, sol);
                yl[0]       = 0.0;
                yl[m_stage] = -1.0;
                std::copy_n(&sol[0], ldz - 2, &yl[1]);
            }

            //  --- Convex combination ---
            if (m_stage > 1)
            {
                kappa0 = 0.0;
                kappal = 0.0;
                gamma  = 0.0;
                for (unsigned int ii = 0; ii <= m_stage; ++ii)
                {
                    TData s0 = 0.0;
                    TData sl = 0.0;
                    for (unsigned int i = 0; i <= m_stage; ++i)
                    {
                        s0 += Zvec[ii][i] * y0[i];
                        sl += Zvec[ii][i] * yl[i];
                    }
                    kappa0 += y0[ii] * s0;
                    kappal += yl[ii] * sl;
                    gamma += yl[ii] * s0;
                }

                kappa0 = std::sqrt(std::abs(kappa0));
                kappal = std::sqrt(std::abs(kappal));
                gamma /= kappa0 * kappal;

                if (kappa0 != 0.0 && kappal != 0.0)
                {
                    if (gamma > 0.0)
                    {
                        gamma = std::max(std::abs(gamma), kappa);
                    }
                    else
                    {
                        gamma = -std::max(std::abs(gamma), kappa);
                    }

                    for (unsigned int i = 0; i <= m_stage; i++)
                    {
                        y0[i] -= gamma * (kappa0 / kappal) * yl[i];
                    }
                }
            }

            // Update solution.
            omega = y0[m_stage];
            for (unsigned int ii = 1; ii <= m_stage; ++ii)
            {
                daxpy<ExecSpace>(y0[ii], m_r[ii - 1], m_acc, m_acc);
                daxpy<ExecSpace>(-y0[ii], m_u[ii], m_u[0], m_u[0]);
                daxpy<ExecSpace>(-y0[ii], m_r[ii], m_r[0], m_r[0]);
            }
            eps = m_math.ddot(m_r[0], m_r[0]);
            m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);
            zeta = std::sqrt(eps);

            // Accurate update.
            if (m_accurateUpdate)
            {
                MaxResApprox = std::max(zeta, MaxResApprox);
                MaxResTrue   = std::max(zeta, MaxResTrue);

                const bool update_app =
                    (zeta < delta * zeta0) && (zeta0 <= MaxResApprox);
                const bool compute_res =
                    (zeta < delta * MaxResTrue) && (zeta0 <= MaxResTrue);
                if (compute_res || update_app)
                {
                    // Perform the method-specific matrix-vector multiply
                    // operation.
                    auto &tmp3 = (m_rightPreconditioner) ? m_w : m_acc;
                    if (m_rightPreconditioner)
                    {
                        this->m_precon->Apply(m_acc, tmp3);
                    }
                    this->m_lhs->Apply(tmp3, m_r[0]);
                    m_robBndCondOp->Apply(tmp3, m_r[0]);
                    m_assmbScatrZeroDirOp->Apply(m_r[0]);
                    if (m_leftPreconditioner)
                    {
                        this->m_precon->Apply(m_r[0], m_r[0]);
                    }

                    // Compute exact residual.
                    sub<ExecSpace>(m_rhs, m_r[0], m_r[0]);

                    MaxResTrue = zeta;
                    if (update_app)
                    {
                        if (m_rightPreconditioner)
                        {
                            add<ExecSpace>(m_w, out, out);
                        }
                        else
                        {
                            add<ExecSpace>(m_acc, out, out);
                        }
                        m_acc.template Initialize<MemSpace>(0);
                        m_rhs.template Copy<MemSpace>(m_r[0]);

                        MaxResApprox = zeta;
                    }
                }
            }

            // Test if norm is within tolerance.
            if (eps < m_tol * m_tol * rhsMagnitude)
            {
                if (m_root)
                {
                    std::cout << "iterations: " << totalIterations
                              << " eps: " << std::sqrt(eps)
                              << " rhs_mag: " << rhsMagnitude << std::endl;
                }
                break;
            }
        }

        // Final update.
        if (m_rightPreconditioner)
        {
            this->m_precon->Apply(m_acc, m_acc);
        }
        add<ExecSpace>(m_acc, out, out);
    }

    void DirectSolve(std::vector<std::vector<TData>> &A, std::vector<TData> &b)
    {
        unsigned int n = A.size();

        // Forward Elimination with Partial Pivoting.
        for (unsigned int k = 0; k < n; ++k)
        {
            // --- Partial Pivoting ---
            unsigned int maxRow = k;
            TData maxVal        = std::abs(A[k][k]);
            for (unsigned int i = k + 1; i < n; ++i)
            {
                if (std::abs(A[i][k]) > maxVal)
                {
                    maxVal = std::abs(A[i][k]);
                    maxRow = i;
                }
            }

            std::swap(A[k], A[maxRow]);
            std::swap(b[k], b[maxRow]);

            // --- Elimination Stage ---
            for (unsigned int i = k + 1; i < n; ++i)
            {
                TData factor = A[i][k] / A[k][k];
                b[i] -= factor * b[k];
                for (unsigned int j = k; j < n; ++j)
                {
                    A[i][j] -= factor * A[k][j];
                }
            }
        }

        // Backward Substitution.
        for (int i = n - 1; i >= 0; --i)
        {
            TData sum = 0;
            for (unsigned int j = i + 1; j < n; ++j)
            {
                sum += A[i][j] * b[j];
            }
            b[i] = (b[i] - sum) / A[i][i];
        }
    }
};

} // namespace Nektar::Operators::detail
