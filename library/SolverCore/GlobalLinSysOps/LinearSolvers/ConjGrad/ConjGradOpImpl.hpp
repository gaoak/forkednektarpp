///////////////////////////////////////////////////////////////////////////////
//
// File: ConjGradOpImpl.hpp
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

#include "SolverCore/GlobalLinSysOps/LinearSolvers/ConjGrad/ConjGradOp.hpp"

#include "SolverCore/GlobalLinSysOps/LinearSolvers/ConjGrad/ConjGradKernels.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class ConjGradOpImpl : public ConjGradOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ConjGradOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : ConjGradOp<TData>(expansionList, components),
          m_w(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjGrad w",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_s(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjGrad s",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjGrad r",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_q(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjGrad q",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_p(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjGrad p",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_vExchange(LibUtilities::MemoryRegion<TData>(4, eHostPinned))
    {
        this->template SetLinearSolver<ExecSpace>();

        auto session = this->m_expansionList->GetSession();

        // Set parameters.
        this->m_leftPreconditioner =
            session->DefinesParameter("LinSysLeftPrecon")
                ? session->GetParameter("LinSysLeftPrecon")
                : true;
        this->m_rightPreconditioner =
            session->DefinesParameter("LinSysRightPrecon")
                ? session->GetParameter("LinSysRightPrecon")
                : false;
        m_flexible = session->DefinesParameter("FlexibleConjugateGradient")
                         ? session->GetParameter("FlexibleConjugateGradient")
                         : false;

        ASSERTL0(!this->m_rightPreconditioner,
                 "ConjGradOpImpl: Only left preconditioner is supported");

        // Fill mask.
        this->template SetMask<ExecSpace>();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<ConjGradOpImpl<ExecSpace, TData>>(expansionList,
                                                                  components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_w;
    LibUtilities::Field<TData, FieldState::Coeff> m_s;
    LibUtilities::Field<TData, FieldState::Coeff> m_r;
    LibUtilities::Field<TData, FieldState::Coeff> m_q;
    LibUtilities::Field<TData, FieldState::Coeff> m_p;
    LibUtilities::MemoryRegion<TData> m_vExchange;

    bool m_flexible;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // Reshape mask if required.
        this->template ReshapeMask<ExecSpace>(in);

        // Convergence parameters.
        this->m_niter = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, rho, rho_new, rho_star, mu;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);
        out.SetInterleaveWidth(in);

        // Reset device memory.
        auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrOp->Apply(m_r);
        Math::ddot<ExecSpace>(in, m_r, exchange + 1);

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrZeroDirOp->Apply(m_r);
        Math::ddot<ExecSpace>(in, m_r, exchange + 0);

        // Communication.
        this->m_rowComm->template AllReduce<MemSpace>(
            m_vExchange, Nektar::LibUtilities::ReduceSum);

        // Device-to-host copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        rhsMagnitude = this->GetRhsMagnitude(exchangeHost[1]);
        eps          = exchangeHost[0];

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            this->PrintVerboseOutput(this->name, "error",
                                     std::sqrt(eps / rhsMagnitude),
                                     rhsMagnitude);
            return;
        }

        // Apply preconditioner - output is assembled
        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_w);
        }
        else
        {
            m_w.template Copy<MemSpace>(m_r);
        }

        // Iteration >= 1
        alpha    = 1.0;
        beta     = 0.0;
        rho      = 1.0;
        rho_star = 0.0;
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

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            if (this->m_niter == 0)
            {
                m_p.template Initialize<MemSpace>(0);
                m_p.SetInterleaveWidth(in);
                m_q.template Initialize<MemSpace>(0);
                m_q.SetInterleaveWidth(in);
            }
            else
            {
                // Assemble matrix output from previous matrix-vector multiply
                // could be moved around loop if optimal elsewhere.
                this->m_assmbScatrZeroDirOp->Apply(m_s);

                // Compute new search direction.
                // Math::daxpy<ExecSpace>(beta, m_p, m_w, m_p);
                // Math::daxpy<ExecSpace>(beta, m_q, m_s, m_q);
                // Math::daxpy<ExecSpace>(alpha, m_p, out, out);
                // Math::daxpy<ExecSpace>(-alpha, m_q, m_r, m_r);
                UpdateConjGradSearchDirection<ExecSpace>(alpha, beta, m_w, m_s,
                                                         m_p, m_q, m_r, out);

                // <r_{k+1}, r_{k+1}>
                Math::ddot<ExecSpace>(this->m_mask, m_r, m_r, exchange + 0);

                if (m_flexible)
                {
                    // <r_{k+1}, w_{k}>
                    Math::ddot<ExecSpace>(this->m_mask, m_r, m_w, exchange + 3);
                }

                // Apply preconditioner - output is assumeed holding global dof
                if (this->m_leftPreconditioner)
                {
                    this->m_precon->Apply(m_r, m_w);
                }
                else
                {
                    m_w.template Copy<MemSpace>(m_r);
                }
            }

            // <r_{k+1}, w_{k+1}>
            Math::ddot<ExecSpace>(this->m_mask, m_r, m_w, exchange + 1);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_w, m_s);
            this->m_robBndCondOp->Apply(m_w, m_s);

            // <w_{k+1}, s_{k+1}>
            Math::ddot<ExecSpace>(m_w, m_s, exchange + 2);

            // Communication.
            this->m_rowComm->template AllReduce<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            eps     = exchangeHost[0];
            rho_new = exchangeHost[1];
            mu      = exchangeHost[2];
            if (m_flexible)
            {
                rho_star = exchangeHost[3];
            }

            ++this->m_niter;

            // Test if norm is within tolerance.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(this->name, "error",
                                         std::sqrt(eps / rhsMagnitude),
                                         rhsMagnitude);
                break;
            }

            // Update coefficients.
            beta  = (this->m_niter > 1) ? (rho_new - rho_star) / rho : 0.0;
            alpha = rho_new / (mu - rho_new * beta / alpha);
            rho   = rho_new;
        }
    }
};

} // namespace Nektar::SolverCore::detail
