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

#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
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
          m_p(Field<TData, FieldState::Coeff>(
              "BICGSTABROp p",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_v(Field<TData, FieldState::Coeff>(
              "BICGSTABROp v",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_w(Field<TData, FieldState::Coeff>(
              "BICGSTABROp w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_z(Field<TData, FieldState::Coeff>(
              "BICGSTABROp z",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_s(Field<TData, FieldState::Coeff>(
              "BICGSTABROp s",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r(Field<TData, FieldState::Coeff>(
              "BICGSTABROp r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_rtilde(Field<TData, FieldState::Coeff>(
              "BICGSTABROp rtilde",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_vExchange(MemoryRegion<TData>(4, ePinned))
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
        session->LoadParameter("BICGSTABRLeftPrecon", leftPreconditioner, 0);
        session->LoadParameter("BICGSTABRRightPrecon", rightPreconditioner, 0);
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
        return std::make_unique<BICGSTABROpImpl<ExecSpace, TData>>(
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

    Field<TData, FieldState::Coeff> m_p;
    Field<TData, FieldState::Coeff> m_v;
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_z;
    Field<TData, FieldState::Coeff> m_s;
    Field<TData, FieldState::Coeff> m_r;
    Field<TData, FieldState::Coeff> m_rtilde;

    MemoryRegion<TData> m_vExchange;

    TData m_tol                = 0.0;
    unsigned int m_maxIter     = 0;
    bool m_leftPreconditioner  = false;
    bool m_rightPreconditioner = false;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // BICGSTAB implementation adpated from  FBiCGStab-R PETSC
        // implementation. This version has only 2 MPI calls per iterations
        // (comparatively to 3 MPI calls).

        // Reference:
        // https://github.com/petsc/petsc/blob/main/src/ksp/ksp/impls/bcgs/fbcgsr/fbcgsr.c

        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, sigma, tau;
        TData omega, omega0, omega1;

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
        m_assmbScatrZeroDirOp->Apply(m_r);

        if (m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_r);
        }

        ddot<ExecSpace>(m_r, m_r, exchange + 0);

        // Communication.
        m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                       Nektar::LibUtilities::ReduceSum);

        // Device-to-host copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        rhsMagnitude = (exchangeHost[1] > 1.0e-6) ? exchangeHost[1] : 1.0;
        eps          = exchangeHost[0];

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
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
            if (totalIterations > m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Update search vectors.
            if (totalIterations > 0)
            {
                daxpy<ExecSpace>(-omega, m_v, m_p, m_p);
                daxpy<ExecSpace>(beta, m_p, m_r, m_p);
            }

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp = (m_rightPreconditioner) ? m_w : m_p;
            if (m_rightPreconditioner)
            {
                this->m_precon->Apply(m_p, tmp);
            }
            this->m_lhs->Apply(tmp, m_v);
            m_robBndCondOp->Apply(tmp, m_v);
            m_assmbScatrZeroDirOp->Apply(m_v);
            if (m_leftPreconditioner)
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
            m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                           Nektar::LibUtilities::ReduceSum);

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
            if (eps < m_tol * m_tol * rhsMagnitude)
            {
                if (m_root)
                {
                    std::cout << "iterations: " << totalIterations
                              << " eps: " << std::sqrt(eps)
                              << " rhs_mag: " << rhsMagnitude << std::endl;
                }
                break;
            }*/

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp2 = (m_rightPreconditioner) ? m_w : m_s;
            if (m_rightPreconditioner)
            {
                this->m_precon->Apply(m_s, tmp2);
            }
            this->m_lhs->Apply(tmp2, m_z);
            m_robBndCondOp->Apply(tmp2, m_z);
            m_assmbScatrZeroDirOp->Apply(m_z);
            if (m_leftPreconditioner)
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
            m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                           Nektar::LibUtilities::ReduceSum);

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

            ++totalIterations;

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
    }
};

} // namespace Nektar::Operators::detail
