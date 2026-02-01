///////////////////////////////////////////////////////////////////////////////
//
// File: TFQMROpImpl.hpp
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
#include "Operators/GlobalLinSysOps/LinearSolvers/TFQMR/TFQMROp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class TFQMROpImpl : public TFQMROp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    TFQMROpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : TFQMROp<TData>(expansionList, components),
          m_w(Field<TData, FieldState::Coeff>(
              "TFQMROp w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_s(Field<TData, FieldState::Coeff>(
              "TFQMROp s",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_u(Field<TData, FieldState::Coeff>(
              "TFQMROp u",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_p(Field<TData, FieldState::Coeff>(
              "TFQMROp p",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_d(Field<TData, FieldState::Coeff>(
              "TFQMROp p",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r(Field<TData, FieldState::Coeff>(
              "TFQMROp r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_rtilde(Field<TData, FieldState::Coeff>(
              "TFQMROp rtilde",
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
        session->LoadParameter("TFQMRLeftPrecon", leftPreconditioner, 0);
        session->LoadParameter("TFQMRRightPrecon", rightPreconditioner, 0);
        m_leftPreconditioner  = leftPreconditioner;
        m_rightPreconditioner = rightPreconditioner;
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<TFQMROpImpl<ExecSpace, TData>>(expansionList,
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

    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_s;
    Field<TData, FieldState::Coeff> m_u;
    Field<TData, FieldState::Coeff> m_p;
    Field<TData, FieldState::Coeff> m_d;
    Field<TData, FieldState::Coeff> m_r;
    Field<TData, FieldState::Coeff> m_rtilde;

    TData m_tol                = 0.0;
    unsigned int m_maxIter     = 0;
    bool m_leftPreconditioner  = false;
    bool m_rightPreconditioner = false;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Adapted from:
        // https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/tfqmr/tfqmr.py
        // Based on R. W. Freund, *A Transpose-Free Quasi-Minimal Residual
        // Method for Non-Hermitian Linear Systems*, SIAM Journal on Scientific
        // Computing, **14** (2), pp. 470--482, 1993.

        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, rho, rho_new;
        TData sigma, theta, eta, tau;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);
        m_d.template Initialize<MemSpace>(0);

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        m_assmbScatrOp->Apply(m_r);
        rhsMagnitude = m_math.ddot(in, m_r);
        m_rowComm->AllReduce(rhsMagnitude, Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        m_assmbScatrZeroDirOp->Apply(m_r);

        if (m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_r);
        }

        eps = m_math.ddot(in, m_r);
        m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        m_p.template Initialize<MemSpace>(0);
        m_u.template Copy<MemSpace>(m_r);
        m_rtilde.template Copy<MemSpace>(m_r);
        rho_new = m_math.ddot(m_rtilde, m_r);
        m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
        tau   = std::sqrt(rho_new);
        theta = 0.0;
        eta   = 0.0;
        beta  = 0.0;
        while (true)
        {
            if (totalIterations > m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Update vectors.
            if (totalIterations > 0)
            {
                daxpy<ExecSpace>(beta, m_u, m_r, m_u);
                daxpy<ExecSpace>(beta, m_p, m_s, m_p);
            }

            ++totalIterations;

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp = (m_rightPreconditioner) ? m_w : m_u;
            if (m_rightPreconditioner)
            {
                this->m_precon->Apply(m_u, tmp);
            }
            this->m_lhs->Apply(tmp, m_s);
            m_robBndCondOp->Apply(tmp, m_s);
            m_assmbScatrZeroDirOp->Apply(m_s);
            if (m_leftPreconditioner)
            {
                this->m_precon->Apply(m_s, m_s);
            }

            // Update vectors.
            daxpy<ExecSpace>(beta, m_p, m_s, m_p);

            // Update coefficients.
            alpha = m_math.ddot(m_rtilde, m_p);
            m_rowComm->AllReduce(alpha, Nektar::LibUtilities::ReduceSum);
            alpha = rho_new / alpha;

            // --- First pass ---
            // Update vectors.
            daxpy<ExecSpace>(-alpha, m_s, m_r, m_r);
            daxpy<ExecSpace>(theta * theta * eta / alpha, m_d, tmp, m_d);

            // Update coefficients.
            theta = m_math.ddot(m_r, m_r);
            m_rowComm->AllReduce(theta, Nektar::LibUtilities::ReduceSum);
            theta = std::sqrt(theta) / tau;
            sigma = 1.0 / std::sqrt(1.0 + theta * theta);
            tau *= theta * sigma;

            // Update solution.
            eta = sigma * sigma * alpha;
            daxpy<ExecSpace>(eta, m_d, out, out);

            // Test if norm is within tolerance.
            eps = tau * tau * (2 * totalIterations);
            if (eps < m_tol * m_tol * rhsMagnitude)
            {
                if (m_root)
                {
                    std::cout
                        << "iterations: " << totalIterations
                        << " eps: " << tau * std::sqrt(2 * totalIterations)
                        << " rhs_mag: " << rhsMagnitude << std::endl;
                }
                break;
            }

            // --- Second pass ---
            // Update vectors.
            daxpy<ExecSpace>(-alpha, m_p, m_u, m_u);

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp2 = (m_rightPreconditioner) ? m_w : m_u;
            if (m_rightPreconditioner)
            {
                this->m_precon->Apply(m_u, tmp2);
            }
            this->m_lhs->Apply(tmp, m_s);
            m_robBndCondOp->Apply(tmp, m_s);
            m_assmbScatrZeroDirOp->Apply(m_s);
            if (m_leftPreconditioner)
            {
                this->m_precon->Apply(m_s, m_s);
            }

            // Update vectors.
            daxpy<ExecSpace>(-alpha, m_s, m_r, m_r);
            daxpy<ExecSpace>(theta * theta * eta / alpha, m_d, tmp2, m_d);

            // Update coefficients.
            theta = m_math.ddot(m_r, m_r);
            m_rowComm->AllReduce(theta, Nektar::LibUtilities::ReduceSum);
            theta = std::sqrt(theta) / tau;
            sigma = 1.0 / std::sqrt(1.0 + theta * theta);
            tau *= theta * sigma;

            // Update solution.
            eta = sigma * sigma * alpha;
            daxpy<ExecSpace>(eta, m_d, out, out);

            // Test if norm is within tolerance.
            eps = tau * tau * (2 * totalIterations + 1);
            if (eps < m_tol * m_tol * rhsMagnitude)
            {
                if (m_root)
                {
                    std::cout
                        << "iterations: " << totalIterations
                        << " eps: " << tau * std::sqrt(2 * totalIterations + 1)
                        << " rhs_mag: " << rhsMagnitude << std::endl;
                }
                break;
            }

            // Update coefficients.
            rho     = rho_new;
            rho_new = m_math.ddot(m_rtilde, m_r);
            m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
            beta = rho_new / rho;
        }
    }
};

} // namespace Nektar::Operators::detail
