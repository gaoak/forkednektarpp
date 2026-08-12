///////////////////////////////////////////////////////////////////////////////
//
// File: PipeConjGradOpImpl.hpp
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

#include "Operators/GlobalLinSysOps/LinearSolvers/PipeConjGrad/PipeConjGradOp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class PipeConjGradOpImpl : public PipeConjGradOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    PipeConjGradOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                       const std::vector<std::string> &components)
        : PipeConjGradOp<TData>(expansionList, components),
          m_m(MultiRegions::Field<TData, FieldState::Coeff>(
              "PipeConjGrad m",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_n(MultiRegions::Field<TData, FieldState::Coeff>(
              "PipeConjGrad n",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_z(MultiRegions::Field<TData, FieldState::Coeff>(
              "PipeConjGrad z",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_w(MultiRegions::Field<TData, FieldState::Coeff>(
              "PipeConjGrad w",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_u(MultiRegions::Field<TData, FieldState::Coeff>(
              "PipeConjGrad u",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_s(MultiRegions::Field<TData, FieldState::Coeff>(
              "PipeConjGrad s",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r(MultiRegions::Field<TData, FieldState::Coeff>(
              "PipeConjGrad r",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_q(MultiRegions::Field<TData, FieldState::Coeff>(
              "PipeConjGrad q",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_p(MultiRegions::Field<TData, FieldState::Coeff>(
              "PipeConjGrad p",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_vExchange(LibUtilities::MemoryRegion<TData>(3, eHostPinned))
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

        m_request = this->m_rowComm->CreateRequest(1);

        ASSERTL0(!this->m_rightPreconditioner,
                 "PipeConjGradOpImpl: Only left preconditioner is supported");
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<PipeConjGradOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    MultiRegions::Field<TData, FieldState::Coeff> m_m;
    MultiRegions::Field<TData, FieldState::Coeff> m_n;
    MultiRegions::Field<TData, FieldState::Coeff> m_z;
    MultiRegions::Field<TData, FieldState::Coeff> m_w;
    MultiRegions::Field<TData, FieldState::Coeff> m_u;
    MultiRegions::Field<TData, FieldState::Coeff> m_s;
    MultiRegions::Field<TData, FieldState::Coeff> m_r;
    MultiRegions::Field<TData, FieldState::Coeff> m_q;
    MultiRegions::Field<TData, FieldState::Coeff> m_p;
    LibUtilities::MemoryRegion<TData> m_vExchange;

    LibUtilities::CommRequestSharedPtr m_request;

    void v_Apply(MultiRegions::Field<TData, FieldState::Coeff> &in,
                 MultiRegions::Field<TData, FieldState::Coeff> &out) override
    {
        // Based on the pipelined conjugate gradiant method
        //
        // Reference:
        // Ghysels, Pieter, and Wim Vanroose. "Hiding global synchronization
        // latency in the preconditioned conjugate gradient algorithm." Parallel
        // Computing 40, no. 7 (2014): 224-238.

        // Convergence parameters.
        this->m_niter                        = 0;
        unsigned int residualReplacementFreq = 50;
        TData rhsMagnitude, eps, scale;
        TData alpha, beta, rho, rho_new, mu;

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
        this->m_assmbScatrZeroDirOp->Apply(m_r, m_u);
        Math::ddot<ExecSpace>(m_r, m_u, exchange + 0);

        // Apply preconditioner
        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_u, m_u);
        }
        Math::ddot<ExecSpace>(m_u, m_u, exchange + 2);

        // Communication.
        this->m_rowComm->template AllReduce<MemSpace>(
            m_vExchange, Nektar::LibUtilities::ReduceSum);

        // Device-to-host copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        rhsMagnitude = (exchangeHost[1] > 1.0e-6) ? exchangeHost[1] : 1.0;
        eps          = exchangeHost[0];
        scale        = exchangeHost[2] / eps;

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        alpha = 1.0;
        rho   = 1.0;
        this->m_lhs->Apply(m_u, m_w);
        this->m_robBndCondOp->Apply(m_u, m_w);
        while (true)
        {
            if (this->m_niter > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << this->m_niter;
                WARNINGL0(false, msg.str());

                return;
            }

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // Residual replacement strategy.
            if (this->m_niter > 0 &&
                this->m_niter % residualReplacementFreq == 0)
            {
                this->m_lhs->Apply(out, m_r);
                this->m_robBndCondOp->Apply(out, m_r);
                Math::sub<ExecSpace>(in, m_r, m_r);
                this->m_assmbScatrZeroDirOp->Apply(m_r, m_u);
                if (this->m_leftPreconditioner)
                {
                    this->m_precon->Apply(m_u, m_u);
                }
                this->m_lhs->Apply(m_u, m_w);
                this->m_robBndCondOp->Apply(m_u, m_w);
            }

            // <u_{k+1}, u_{k+1}>
            Math::ddot<ExecSpace>(m_u, m_u, exchange + 0);

            // <r_{k+1}, u_{k+1}>
            Math::ddot<ExecSpace>(m_r, m_u, exchange + 1);

            // <w_{k+1}, u_{k+1}>
            Math::ddot<ExecSpace>(m_w, m_u, exchange + 2);

            // Begin communication.
            this->m_rowComm->template AllReduceBegin<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum, m_request);

            // Overlap communication with matrix-vector multiply operation.
            this->m_assmbScatrZeroDirOp->Apply(m_w, m_m);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_m, m_m);
            }
            this->m_lhs->Apply(m_m, m_n);
            this->m_robBndCondOp->Apply(m_m, m_n);

            // End communication.
            this->m_rowComm->template AllReduceEnd<MemSpace>(m_vExchange,
                                                             m_request);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            eps     = exchangeHost[0];
            rho_new = exchangeHost[1];
            mu      = exchangeHost[2];

            // Update coefficients.
            beta  = (this->m_niter > 0) ? rho_new / rho : 0.0;
            alpha = rho_new / (mu - rho_new * beta / alpha);
            rho   = rho_new;

            ++this->m_niter;

            // Test if norm is within tolerance.
            if (eps < 100 * scale * this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(this->name, "error",
                                         std::sqrt(eps / rhsMagnitude),
                                         rhsMagnitude);
                break;
            }

            // Compute new search direction.
            if (this->m_niter == 1)
            {
                m_z.template Copy<MemSpace>(m_n);
                m_q.template Copy<MemSpace>(m_m);
                m_s.template Copy<MemSpace>(m_w);
                m_p.template Copy<MemSpace>(m_u);
            }
            else
            {
                Math::daxpy<ExecSpace>(beta, m_z, m_n, m_z);
                Math::daxpy<ExecSpace>(beta, m_q, m_m, m_q);
                Math::daxpy<ExecSpace>(beta, m_s, m_w, m_s);
                Math::daxpy<ExecSpace>(beta, m_p, m_u, m_p);
            }
            Math::daxpy<ExecSpace>(alpha, m_p, out, out);
            Math::daxpy<ExecSpace>(-alpha, m_s, m_r, m_r);
            Math::daxpy<ExecSpace>(-alpha, m_q, m_u, m_u);
            Math::daxpy<ExecSpace>(-alpha, m_z, m_w, m_w);
        }
    }
};

} // namespace Nektar::Operators::detail
