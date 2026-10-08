///////////////////////////////////////////////////////////////////////////////
//
// File: BICGSTABOpImpl.hpp
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

#include "SolverCore/GlobalLinSysOps/LinearSolvers/BICGSTAB/BICGSTABOp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class BICGSTABOpImpl : public BICGSTABOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BICGSTABOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : BICGSTABOp<TData>(expansionList, components),
          m_p(LibUtilities::Field<TData, FieldState::Coeff>(
              "BICGSTABOp p",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_v(LibUtilities::Field<TData, FieldState::Coeff>(
              "BICGSTABOp v",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_w(LibUtilities::Field<TData, FieldState::Coeff>(
              "BICGSTABOp w",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_z(LibUtilities::Field<TData, FieldState::Coeff>(
              "BICGSTABOp z",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r(LibUtilities::Field<TData, FieldState::Coeff>(
              "BICGSTABOp r",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_rtilde(LibUtilities::Field<TData, FieldState::Coeff>(
              "BICGSTABOp rtilde",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_vExchange(LibUtilities::MemoryRegion<TData>(2, eHostPinned))
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
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<BICGSTABOpImpl<ExecSpace, TData>>(expansionList,
                                                                  components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_p;
    LibUtilities::Field<TData, FieldState::Coeff> m_v;
    LibUtilities::Field<TData, FieldState::Coeff> m_w;
    LibUtilities::Field<TData, FieldState::Coeff> m_z;
    LibUtilities::Field<TData, FieldState::Coeff> m_r;
    LibUtilities::Field<TData, FieldState::Coeff> m_rtilde;
    LibUtilities::MemoryRegion<TData> m_vExchange;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // Convergence parameters.
        this->m_niter = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, rho, rho_new;
        TData omega, omega0, omega1;

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

        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_r);
        }

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
            return;
        }

        // Iteration >= 1
        beta  = 0.0;
        omega = 0.0;
        m_p.template Copy<MemSpace>(m_r);
        m_rtilde.template Copy<MemSpace>(m_r);
        rho_new = this->m_math.ddot(m_rtilde, m_r);
        this->m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
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

            // Update search vectors.
            if (this->m_niter > 0)
            {
                Math::daxpbypz<ExecSpace>(-beta * omega, m_v, beta, m_p, m_r,
                                          m_p);
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

            // Update coefficients.
            alpha = this->m_math.ddot(m_v, m_rtilde);
            this->m_rowComm->AllReduce(alpha, LibUtilities::ReduceSum);
            alpha = rho_new / alpha;

            // Update solution.
            Math::daxpy<ExecSpace>(alpha, tmp, out, out);
            Math::daxpy<ExecSpace>(-alpha, m_v, m_r, m_r);

            // Test if norm is within tolerance.
            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, LibUtilities::ReduceSum);
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(this->name, "error",
                                         std::sqrt(eps / rhsMagnitude),
                                         rhsMagnitude);
                break;
            }

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp2 = (this->m_rightPreconditioner) ? m_w : m_r;
            if (this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_r, tmp2);
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

            // Update coefficients.
            Math::ddot<ExecSpace>(m_r, m_z, exchange + 0);
            Math::ddot<ExecSpace>(m_z, m_z, exchange + 1);

            // Communication.
            this->m_rowComm->template AllReduce<MemSpace>(
                m_vExchange, LibUtilities::ReduceSum);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            omega0 = exchangeHost[0];
            omega1 = exchangeHost[1];
            omega  = omega0 / omega1;

            // Update solution.
            Math::daxpy<ExecSpace>(omega, tmp2, out, out);
            Math::daxpy<ExecSpace>(-omega, m_z, m_r, m_r);

            ++this->m_niter;

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // Test if norm is within tolerance.
            Math::ddot<ExecSpace>(m_r, m_r, exchange + 0);
            Math::ddot<ExecSpace>(m_rtilde, m_r, exchange + 1);

            // Communication.
            this->m_rowComm->template AllReduce<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            eps = exchangeHost[0];
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(this->name, "error",
                                         std::sqrt(eps / rhsMagnitude),
                                         rhsMagnitude);
                break;
            }

            // Update coefficients.
            rho     = rho_new;
            rho_new = exchangeHost[1];
            beta    = rho_new / rho * (alpha / omega);
        }
    }
};

} // namespace Nektar::SolverCore::detail
