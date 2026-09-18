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

#include "SolverCore/GlobalLinSysOps/LinearSolvers/TFQMR/TFQMROp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class TFQMROpImpl : public TFQMROp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    TFQMROpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : TFQMROp<TData>(expansionList, components),
          m_w(LibUtilities::Field<TData, FieldState::Coeff>(
              "TFQMROp w",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_s(LibUtilities::Field<TData, FieldState::Coeff>(
              "TFQMROp s",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_u(LibUtilities::Field<TData, FieldState::Coeff>(
              "TFQMROp u",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_p(LibUtilities::Field<TData, FieldState::Coeff>(
              "TFQMROp p",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_d(LibUtilities::Field<TData, FieldState::Coeff>(
              "TFQMROp p",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r(LibUtilities::Field<TData, FieldState::Coeff>(
              "TFQMROp r",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_rtilde(LibUtilities::Field<TData, FieldState::Coeff>(
              "TFQMROp rtilde",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1))
    {
        this->template SetLinearSolver<ExecSpace>();

        auto session = this->m_expansionList->GetSession();

        // Set parameters.
        int leftPreconditioner  = 0;
        int rightPreconditioner = 0;
        session->LoadParameter("LinSysLeftPrecon", leftPreconditioner, 0);
        session->LoadParameter("LinSysRightPrecon", rightPreconditioner, 1);
        this->m_leftPreconditioner  = leftPreconditioner;
        this->m_rightPreconditioner = rightPreconditioner;
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<TFQMROpImpl<ExecSpace, TData>>(expansionList,
                                                               components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_w;
    LibUtilities::Field<TData, FieldState::Coeff> m_s;
    LibUtilities::Field<TData, FieldState::Coeff> m_u;
    LibUtilities::Field<TData, FieldState::Coeff> m_p;
    LibUtilities::Field<TData, FieldState::Coeff> m_d;
    LibUtilities::Field<TData, FieldState::Coeff> m_r;
    LibUtilities::Field<TData, FieldState::Coeff> m_rtilde;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // Based on R. W. Freund, *A Transpose-Free Quasi-Minimal Residual
        // Method for Non-Hermitian Linear Systems*, SIAM Journal on Scientific
        // Computing, **14** (2), pp. 470--482, 1993.

        // Convergence parameters.
        this->m_niter = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, rho, rho_new;
        TData sigma, theta, eta, tau;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);
        out.SetInterleaveWidth(in);
        m_d.template Initialize<MemSpace>(0);
        m_d.SetInterleaveWidth(in);

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
        m_p.template Initialize<MemSpace>(0);
        m_p.SetInterleaveWidth(in);
        m_u.template Copy<MemSpace>(m_r);
        m_rtilde.template Copy<MemSpace>(m_r);
        rho_new = this->m_math.ddot(m_rtilde, m_r);
        this->m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
        tau   = std::sqrt(rho_new);
        theta = 0.0;
        eta   = 0.0;
        beta  = 0.0;
        while (true)
        {
            if (this->m_niter > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << this->m_niter
                    << ". Increase NekLinSysMaxIterations in the session "
                       "PARAMETERS section to allow more iterations.";
                NEKERROR(ErrorUtil::efatal, msg.str());

                return;
            }

            // Update vectors.
            if (this->m_niter > 0)
            {
                Math::daxpy<ExecSpace>(beta, m_u, m_r, m_u);
                Math::daxpy<ExecSpace>(beta, m_p, m_s, m_p);
            }

            ++this->m_niter;

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp = (this->m_rightPreconditioner) ? m_w : m_u;
            if (this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_u, tmp);
            }
            this->m_lhs->Apply(tmp, m_s);
            this->m_robBndCondOp->Apply(tmp, m_s);
            this->m_assmbScatrZeroDirOp->Apply(m_s);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_s, m_s);
            }

            // Update vectors.
            Math::daxpy<ExecSpace>(beta, m_p, m_s, m_p);

            // Update coefficients.
            alpha = this->m_math.ddot(m_rtilde, m_p);
            this->m_rowComm->AllReduce(alpha, Nektar::LibUtilities::ReduceSum);
            alpha = rho_new / alpha;

            // --- First pass ---
            // Update vectors.
            Math::daxpy<ExecSpace>(-alpha, m_s, m_r, m_r);
            Math::daxpy<ExecSpace>(theta * theta * eta / alpha, m_d, tmp, m_d);

            // Update coefficients.
            theta = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(theta, Nektar::LibUtilities::ReduceSum);
            theta = std::sqrt(theta) / tau;
            sigma = 1.0 / std::sqrt(1.0 + theta * theta);
            tau *= theta * sigma;

            // Update solution.
            eta = sigma * sigma * alpha;
            Math::daxpy<ExecSpace>(eta, m_d, out, out);

            // Test if norm is within tolerance.
            eps = tau * tau * (2 * this->m_niter);
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(this->name, "error",
                                         tau * std::sqrt(2 * this->m_niter),
                                         rhsMagnitude);
                break;
            }

            // --- Second pass ---
            // Update vectors.
            Math::daxpy<ExecSpace>(-alpha, m_p, m_u, m_u);

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp2 = (this->m_rightPreconditioner) ? m_w : m_u;
            if (this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_u, tmp2);
            }
            this->m_lhs->Apply(tmp, m_s);
            this->m_robBndCondOp->Apply(tmp, m_s);
            this->m_assmbScatrZeroDirOp->Apply(m_s);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_s, m_s);
            }

            // Update vectors.
            Math::daxpy<ExecSpace>(-alpha, m_s, m_r, m_r);
            Math::daxpy<ExecSpace>(theta * theta * eta / alpha, m_d, tmp2, m_d);

            // Update coefficients.
            theta = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(theta, Nektar::LibUtilities::ReduceSum);
            theta = std::sqrt(theta) / tau;
            sigma = 1.0 / std::sqrt(1.0 + theta * theta);
            tau *= theta * sigma;

            // Update solution.
            eta = sigma * sigma * alpha;
            Math::daxpy<ExecSpace>(eta, m_d, out, out);

            // Test if norm is within tolerance.
            eps = tau * tau * (2 * this->m_niter + 1);
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(this->name, "error",
                                         tau * std::sqrt(2 * this->m_niter + 1),
                                         rhsMagnitude);
                break;
            }

            // Update coefficients.
            rho     = rho_new;
            rho_new = this->m_math.ddot(m_rtilde, m_r);
            this->m_rowComm->AllReduce(rho_new,
                                       Nektar::LibUtilities::ReduceSum);
            beta = rho_new / rho;
        }
    }
};

} // namespace Nektar::SolverCore::detail
