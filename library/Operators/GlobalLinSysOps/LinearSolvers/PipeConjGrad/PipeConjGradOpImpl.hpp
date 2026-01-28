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

#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
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
          m_m(Field<TData, FieldState::Coeff>(
              "PipeConjGrad m",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_n(Field<TData, FieldState::Coeff>(
              "PipeConjGrad n",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_z(Field<TData, FieldState::Coeff>(
              "PipeConjGrad z",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_w(Field<TData, FieldState::Coeff>(
              "PipeConjGrad w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_u(Field<TData, FieldState::Coeff>(
              "PipeConjGrad u",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_s(Field<TData, FieldState::Coeff>(
              "PipeConjGrad s",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r(Field<TData, FieldState::Coeff>(
              "PipeConjGrad r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_q(Field<TData, FieldState::Coeff>(
              "PipeConjGrad q",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_p(Field<TData, FieldState::Coeff>(
              "PipeConjGrad p",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_vExchange(MemoryRegion<TData>(3, ePinned))
    {
        auto session = expansionList->GetSession();

        // Set operators.
        m_assmbScatrOp = std::make_unique<AssmbScatrOpImpl<ExecSpace, TData>>(
            this->m_expansionList, components);
        m_assmbScatrZeroDirOp =
            std::make_unique<AssmbScatrZeroDirOpImpl<ExecSpace, TData>>(
                this->m_expansionList, components);
        m_robBndCondOp = RobBndCondOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
        m_rowComm = session->GetComm()->GetRowComm();
        m_root    = m_rowComm->GetRank() == 0;
        m_request = m_rowComm->CreateRequest(1);

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
        return std::make_unique<PipeConjGradOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    LibUtilities::CommRequestSharedPtr m_request;
    LibUtilities::CommSharedPtr m_rowComm = nullptr;
    std::unique_ptr<AssmbScatrOpImpl<ExecSpace, TData>> m_assmbScatrOp;
    std::unique_ptr<AssmbScatrZeroDirOpImpl<ExecSpace, TData>>
        m_assmbScatrZeroDirOp;
    bool m_root;

    std::shared_ptr<RobBndCondOp<TData>> m_robBndCondOp;

    Field<TData, FieldState::Coeff> m_m;
    Field<TData, FieldState::Coeff> m_n;
    Field<TData, FieldState::Coeff> m_z;
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_u;
    Field<TData, FieldState::Coeff> m_s;
    Field<TData, FieldState::Coeff> m_r;
    Field<TData, FieldState::Coeff> m_q;
    Field<TData, FieldState::Coeff> m_p;

    MemoryRegion<TData> m_vExchange;

    TData m_tol            = 0.0;
    unsigned int m_maxIter = 0;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Based on the pipelined conjugate gradiant method
        //
        // Reference:
        // Ghysels, Pieter, and Wim Vanroose. "Hiding global synchronization
        // latency in the preconditioned conjugate gradient algorithm." Parallel
        // Computing 40, no. 7 (2014): 224-238.

        // Convergence parameters.
        unsigned int totalIterations = 0, residualReplacementFreq = 50;
        TData rhsMagnitude, eps, mu, scale;
        TData alpha = 1.0, beta, rho = 1.0, rho_new = 1.0;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);

        // Reset device memory.
        auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        m_assmbScatrOp->Apply(m_r);
        ddot<ExecSpace>(in, m_r, exchange + 1);

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        ddot<ExecSpace>(m_r, m_r, exchange + 0);

        // Apply preconditioner
        m_assmbScatrZeroDirOp->Apply(m_r, m_u);
        this->m_precon->Apply(m_u, m_u);
        ddot<ExecSpace>(m_u, m_u, exchange + 2);

        // Communication.
        m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                       Nektar::LibUtilities::ReduceSum);

        // Device-to-host copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        rhsMagnitude = (exchangeHost[1] > 1.0e-6) ? exchangeHost[1] : 1.0;
        eps          = exchangeHost[0];
        scale        = exchangeHost[2] / eps;

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        this->m_lhs->Apply(m_u, m_w);
        m_robBndCondOp->Apply(m_u, m_w);
        while (true)
        {
            if (totalIterations > m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // Residual replacement strategy.
            if (totalIterations > 0 &&
                totalIterations % residualReplacementFreq == 0)
            {
                this->m_lhs->Apply(out, m_r);
                m_robBndCondOp->Apply(out, m_r);
                sub<ExecSpace>(in, m_r, m_r);
                m_assmbScatrZeroDirOp->Apply(m_r, m_u);
                this->m_precon->Apply(m_u, m_u);
                this->m_lhs->Apply(m_u, m_w);
                m_robBndCondOp->Apply(m_u, m_w);
            }

            // <u_{k+1}, u_{k+1}>
            ddot<ExecSpace>(m_u, m_u, exchange + 0);

            // <r_{k+1}, u_{k+1}>
            ddot<ExecSpace>(m_r, m_u, exchange + 1);

            // <w_{k+1}, u_{k+1}>
            ddot<ExecSpace>(m_w, m_u, exchange + 2);

            // Begin communication.
            m_rowComm->AllReduceBegin<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum, m_request);

            // Overlap communication with matrix-vector multiply operation.
            m_assmbScatrZeroDirOp->Apply(m_w, m_m);
            this->m_precon->Apply(m_m, m_m);
            this->m_lhs->Apply(m_m, m_n);
            m_robBndCondOp->Apply(m_m, m_n);

            // End communication.
            m_rowComm->AllReduceEnd<MemSpace>(m_vExchange, m_request);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            eps     = exchangeHost[0];
            rho_new = exchangeHost[1];
            mu      = exchangeHost[2];

            // Update coefficients.
            beta  = (totalIterations > 0) ? rho_new / rho : 0.0;
            alpha = rho_new / (mu - rho_new * beta / alpha);
            rho   = rho_new;

            ++totalIterations;

            // Test if norm is within tolerance.
            if (eps < 100 * scale * m_tol * m_tol * rhsMagnitude)
            {
                if (m_root)
                {
                    std::cout << "iterations: " << totalIterations
                              << " eps: " << std::sqrt(eps)
                              << " rhs_mag: " << rhsMagnitude << std::endl;
                }
                break;
            }

            // Compute new search direction.
            if (totalIterations == 1)
            {
                m_z.template Copy<MemSpace>(m_n);
                m_q.template Copy<MemSpace>(m_m);
                m_s.template Copy<MemSpace>(m_w);
                m_p.template Copy<MemSpace>(m_u);
            }
            else
            {
                daxpy<ExecSpace>(beta, m_z, m_n, m_z);
                daxpy<ExecSpace>(beta, m_q, m_m, m_q);
                daxpy<ExecSpace>(beta, m_s, m_w, m_s);
                daxpy<ExecSpace>(beta, m_p, m_u, m_p);
            }
            daxpy<ExecSpace>(alpha, m_p, out, out);
            daxpy<ExecSpace>(-alpha, m_s, m_r, m_r);
            daxpy<ExecSpace>(-alpha, m_q, m_u, m_u);
            daxpy<ExecSpace>(-alpha, m_z, m_w, m_w);
        }
    }
};

} // namespace Nektar::Operators::detail
