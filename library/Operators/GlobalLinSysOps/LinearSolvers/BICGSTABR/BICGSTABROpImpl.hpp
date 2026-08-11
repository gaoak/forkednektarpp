///////////////////////////////////////////////////////////////////////////////
//
// File: BICGSTABROpImpl.hpp
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

#include "Operators/GlobalLinSysOps/LinearSolvers/BICGSTABR/BICGSTABROp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class BICGSTABROpImpl : public BICGSTABROp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BICGSTABROpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : BICGSTABROp<TData>(expansionList, components),
          m_p(MultiRegions::Field<TData, FieldState::Coeff>(
              "BICGSTABROp p",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_v(MultiRegions::Field<TData, FieldState::Coeff>(
              "BICGSTABROp v",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_w(MultiRegions::Field<TData, FieldState::Coeff>(
              "BICGSTABROp w",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_z(MultiRegions::Field<TData, FieldState::Coeff>(
              "BICGSTABROp z",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_s(MultiRegions::Field<TData, FieldState::Coeff>(
              "BICGSTABROp s",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r(MultiRegions::Field<TData, FieldState::Coeff>(
              "BICGSTABROp r",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_rtilde(MultiRegions::Field<TData, FieldState::Coeff>(
              "BICGSTABROp rtilde",
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
                : false;
        this->m_rightPreconditioner =
            session->DefinesParameter("LinSysRightPrecon")
                ? session->GetParameter("LinSysRightPrecon")
                : true;
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<BICGSTABROpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    MultiRegions::Field<TData, FieldState::Coeff> m_p;
    MultiRegions::Field<TData, FieldState::Coeff> m_v;
    MultiRegions::Field<TData, FieldState::Coeff> m_w;
    MultiRegions::Field<TData, FieldState::Coeff> m_z;
    MultiRegions::Field<TData, FieldState::Coeff> m_s;
    MultiRegions::Field<TData, FieldState::Coeff> m_r;
    MultiRegions::Field<TData, FieldState::Coeff> m_rtilde;

    LibUtilities::MemoryRegion<TData> m_vExchange;

    void v_Apply(MultiRegions::Field<TData, FieldState::Coeff> &in,
                 MultiRegions::Field<TData, FieldState::Coeff> &out) override
    {
        // BICGSTAB implementation adpated from  FBiCGStab-R PETSC
        // implementation. This version has only 2 MPI calls per iterations
        // (comparatively to 3 MPI calls).

        // Reference:
        // https://github.com/petsc/petsc/blob/main/src/ksp/ksp/impls/bcgs/fbcgsr/fbcgsr.c

        // Convergence parameters.
        this->m_niter = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, sigma, tau;
        TData omega, omega0, omega1;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);
        out.SetInterleaveWidth(in);

        // Reset device memory.
        auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrOp->Apply(m_r);
        ddot<ExecSpace>(in, m_r, exchange + 1);

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrZeroDirOp->Apply(m_r);

        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_r);
        }

        ddot<ExecSpace>(m_r, m_r, exchange + 0);

        // Communication.
        this->m_rowComm->template AllReduce<MemSpace>(
            m_vExchange, Nektar::LibUtilities::ReduceSum);

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

        // Iteration >= 1
        beta  = 0.0;
        omega = 0.0;
        m_p.template Copy<MemSpace>(m_r);
        m_rtilde.template Copy<MemSpace>(m_r);
        while (true)
        {
            if (this->m_niter > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << this->m_niter;
                WARNINGL0(false, msg.str());

                return;
            }

            // Update search vectors.
            if (this->m_niter > 0)
            {
                daxpy<ExecSpace>(-omega, m_v, m_p, m_p);
                daxpy<ExecSpace>(beta, m_p, m_r, m_p);
            }

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp = (this->m_rightPreconditioner) ? m_w : m_p;
            if (this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_p, tmp);
            }
            this->m_lhs->Apply(tmp, m_v);
            this->m_robBndCondOp->Apply(tmp, m_v);
            this->m_assmbScatrZeroDirOp->Apply(m_v);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_v, m_v);
            }

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // Reduction.
            ddot<ExecSpace>(m_r, m_rtilde, exchange + 0);
            ddot<ExecSpace>(m_v, m_rtilde, exchange + 1);
            // ddot<ExecSpace>(m_r, m_r, exchange + 2);

            // Communication.
            this->m_rowComm->template AllReduce<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            // Update coefficients.
            tau   = exchangeHost[0];
            sigma = exchangeHost[1];
            // eps = exchangeHost[2];
            alpha = tau / sigma;

            // Update solution.
            daxpy<ExecSpace>(alpha, tmp, out, out);
            daxpy<ExecSpace>(-alpha, m_v, m_r, m_s);

            /*// Test if norm is within tolerance.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(
                    this->name, "error", std::sqrt(eps / rhsMagnitude),
                    rhsMagnitude);
                break;
            }*/

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp2 = (this->m_rightPreconditioner) ? m_w : m_s;
            if (this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_s, tmp2);
            }
            this->m_lhs->Apply(tmp2, m_z);
            this->m_robBndCondOp->Apply(tmp2, m_z);
            this->m_assmbScatrZeroDirOp->Apply(m_z);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_z, m_z);
            }

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // Reduction.
            ddot<ExecSpace>(m_s, m_s, exchange + 0);
            ddot<ExecSpace>(m_z, m_s, exchange + 1);
            ddot<ExecSpace>(m_z, m_z, exchange + 2);
            ddot<ExecSpace>(m_z, m_rtilde, exchange + 3);

            // Communication.
            this->m_rowComm->template AllReduce<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            // Update coefficients.
            omega0 = exchangeHost[1];
            omega1 = exchangeHost[2];
            omega  = omega0 / omega1;
            beta   = -exchangeHost[3] / sigma;
            eps    = std::abs(exchangeHost[0] - omega0 * omega0 / omega1);

            // Update solution.
            daxpy<ExecSpace>(omega, tmp2, out, out);
            daxpy<ExecSpace>(-omega, m_z, m_s, m_r);

            ++this->m_niter;

            // Test if norm is within tolerance.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(this->name, "error",
                                         std::sqrt(eps / rhsMagnitude),
                                         rhsMagnitude);
                break;
            }
        }
    }
};

} // namespace Nektar::Operators::detail
