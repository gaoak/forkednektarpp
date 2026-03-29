///////////////////////////////////////////////////////////////////////////////
//
// File: PipeConjGrad2OpImpl.hpp
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

#include "Operators/GlobalLinSysOps/LinearSolvers/PipeConjGrad2/PipeConjGrad2Op.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class PipeConjGrad2OpImpl : public PipeConjGrad2Op<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    PipeConjGrad2OpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                        const std::vector<std::string> &components)
        : PipeConjGrad2Op<TData>(expansionList, components),
          m_w(Field<TData, FieldState::Coeff>(
              "PipeConjGrad2 w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_u(Field<TData, FieldState::Coeff>(
              "PipeConjGrad2 u",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_s(Field<TData, FieldState::Coeff>(
              "PipeConjGrad2 s",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r(Field<TData, FieldState::Coeff>(
              "PipeConjGrad2 r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_q(Field<TData, FieldState::Coeff>(
              "PipeConjGrad2 q",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_p(Field<TData, FieldState::Coeff>(
              "PipeConjGrad2 p",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_vExchange(MemoryRegion<TData>(2, eHostPinned))
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
                 "PipeConjGrad2OpImpl: Only left preconditioner is supported");
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<PipeConjGrad2OpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_u;
    Field<TData, FieldState::Coeff> m_s;
    Field<TData, FieldState::Coeff> m_r;
    Field<TData, FieldState::Coeff> m_q;
    Field<TData, FieldState::Coeff> m_p;
    MemoryRegion<TData> m_vExchange;

    LibUtilities::CommRequestSharedPtr m_request;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Based on the Gropp conjugate gradiant method
        //
        // Reference:
        // Ghysels, Pieter, and Wim Vanroose. "Hiding global synchronization
        // latency in the preconditioned conjugate gradient algorithm." Parallel
        // Computing 40, no. 7 (2014): 224-238.

        // Convergence parameters.
        this->m_niter = 0;
        TData rhsMagnitude, eps, scale;
        TData alpha, beta, rho, rho_new, delta;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);

        // Reset device memory.
        auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrOp->Apply(m_r);
        ddot<ExecSpace>(in, m_r, exchange + 1);

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrZeroDirOp->Apply(m_r, m_u);
        ddot<ExecSpace>(m_r, m_u, exchange + 0);

        // Begin communication.
        this->m_rowComm->template AllReduceBegin<MemSpace>(
            m_vExchange, Nektar::LibUtilities::ReduceSum, m_request);

        // Overlap communication with matrix-vector multiply operation.
        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_u, m_u);
        }

        // End communication.
        this->m_rowComm->template AllReduceEnd<MemSpace>(m_vExchange,
                                                         m_request);

        // Device-to-host copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        rhsMagnitude = (exchangeHost[1] > 1.0e-6) ? exchangeHost[1] : 1.0;
        eps          = exchangeHost[0];

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Reset device memory.
        m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        ddot<ExecSpace>(m_u, m_u, exchange + 0);
        ddot<ExecSpace>(m_u, m_r, exchange + 1);

        // Begin communication.
        this->m_rowComm->template AllReduceBegin<MemSpace>(
            m_vExchange, Nektar::LibUtilities::ReduceSum, m_request);

        // Overlap communication with matrix-vector multiply operation.
        m_p.template Copy<MemSpace>(m_u);
        this->m_lhs->Apply(m_p, m_s);
        this->m_robBndCondOp->Apply(m_p, m_s);

        // End communication.
        this->m_rowComm->template AllReduceEnd<MemSpace>(m_vExchange,
                                                         m_request);

        // Device-to-host copy.
        exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        // Iteration >= 1
        alpha = 1.0;
        scale = exchangeHost[0] / eps;
        rho   = exchangeHost[1];
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

            // <p_{k+1}, s_{k+1}>
            ddot<ExecSpace>(m_p, m_s, exchange + 0);

            // Begin communication.
            this->m_rowComm->template AllReduceBegin<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum, m_request);

            // Overlap communication with matrix-vector multiply operation.
            this->m_assmbScatrZeroDirOp->Apply(m_s, m_q);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_q, m_q);
            }

            // End communication.
            this->m_rowComm->template AllReduceEnd<MemSpace>(m_vExchange,
                                                             m_request);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            // Update coefficient
            delta = exchangeHost[0];
            alpha = rho / delta;

            // Compute new search direction.
            daxpy<ExecSpace>(alpha, m_p, out, out);
            daxpy<ExecSpace>(-alpha, m_s, m_r, m_r);
            daxpy<ExecSpace>(-alpha, m_q, m_u, m_u);

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // <u_{k+1}, u_{k+1}>
            ddot<ExecSpace>(m_u, m_u, exchange + 0);

            // <r_{k+1}, u_{k+1}>
            ddot<ExecSpace>(m_r, m_u, exchange + 1);

            // Begin communication.
            this->m_rowComm->template AllReduceBegin<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum, m_request);

            // Overlap communication with matrix-vector multiply operation.
            this->m_lhs->Apply(m_u, m_w);
            this->m_robBndCondOp->Apply(m_u, m_w);

            // End communication.
            this->m_rowComm->template AllReduceEnd<MemSpace>(m_vExchange,
                                                             m_request);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            eps     = exchangeHost[0];
            rho_new = exchangeHost[1];

            ++this->m_niter;

            // Test if norm is within tolerance.
            if (eps < 100 * scale * this->m_tol * this->m_tol * rhsMagnitude)
            {
                if (this->m_root)
                {
                    std::cout << this->name
                              << " iterations made = " << this->m_niter
                              << " using tolerance of " << this->m_tol
                              << " error = " << std::sqrt(eps / rhsMagnitude)
                              << " rhs_mag = " << std::sqrt(rhsMagnitude)
                              << std::endl;
                }
                break;
            }

            // Update coefficients.
            beta = rho_new / rho;
            rho  = rho_new;

            // Compute new search direction.
            daxpy<ExecSpace>(beta, m_p, m_u, m_p);
            daxpy<ExecSpace>(beta, m_s, m_w, m_s);
        }
    }
};

} // namespace Nektar::Operators::detail
