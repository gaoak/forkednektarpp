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

#include <MultiRegions/ContField.h>

#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
#include "Operators/GlobalLinSysOps/ConjGrad/ConjGradOp.hpp"

#include <iomanip>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class ConjGradOpImpl : public ConjGradOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ConjGradOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : ConjGradOp<TData>(expansionList),
          m_w_A(Field<TData, FieldState::Coeff>(
              "ConjGrad w_A",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1, 1,
              ExecSpace::alignment)),
          m_s_A(Field<TData, FieldState::Coeff>(
              "ConjGrad s_A",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1, 1,
              ExecSpace::alignment)),
          m_r_A(Field<TData, FieldState::Coeff>(
              "ConjGrad r_A",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1, 1,
              ExecSpace::alignment)),
          m_wk(Field<TData, FieldState::Coeff>(
              "ConjGrad wk",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1, 1,
              ExecSpace::alignment)),
          m_q_A(Field<TData, FieldState::Coeff>(
              "ConjGrad wk",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1, 1,
              ExecSpace::alignment)),
          m_p_A(Field<TData, FieldState::Coeff>(
              "ConjGrad wk",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1, 1,
              ExecSpace::alignment)),
          m_vExchange(MemoryRegion<TData>(4, ExecSpace::alignment, ePinned))
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);

        // Set operators.
        m_assmbScatrOp = std::make_unique<AssmbScatrOpImpl<ExecSpace, TData>>(
            this->m_expansionList);
        m_assmbScatrZeroDirOp =
            std::make_unique<AssmbScatrZeroDirOpImpl<ExecSpace, TData>>(
                this->m_expansionList);
        m_robBndCondOp =
            RobBndCondOp<TData>::Create(this->m_expansionList, ExecSpace::name);
        m_rowComm = contfield->GetSession()->GetComm()->GetRowComm();
        m_root    = m_rowComm->GetRank() == 0;

        // Set parameters.
        contfield->GetSession()->LoadParameter("NekLinSysMaxIterations",
                                               m_maxIter, 5000);
        contfield->GetSession()->LoadParameter("IterativeSolverTolerance",
                                               m_tol, 1.0E-09);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<ConjGradOpImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    LibUtilities::CommSharedPtr m_rowComm = nullptr;
    std::unique_ptr<AssmbScatrOpImpl<ExecSpace, TData>> m_assmbScatrOp;
    std::unique_ptr<AssmbScatrZeroDirOpImpl<ExecSpace, TData>>
        m_assmbScatrZeroDirOp;
    bool m_root;

    std::shared_ptr<RobBndCondOp<TData>> m_robBndCondOp;

    Field<TData, FieldState::Coeff> m_w_A;
    Field<TData, FieldState::Coeff> m_s_A;
    Field<TData, FieldState::Coeff> m_r_A;
    Field<TData, FieldState::Coeff> m_wk;
    Field<TData, FieldState::Coeff> m_q_A;
    Field<TData, FieldState::Coeff> m_p_A;

    MemoryRegion<TData> m_vExchange;

    TData m_tol            = 0.0;
    unsigned int m_maxIter = 0;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Set the fields to zero.
        out.template Initialize<MemSpace>(0);
        m_p_A.template Initialize<MemSpace>(0);
        m_q_A.template Initialize<MemSpace>(0);

        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, mu;
        TData alpha, beta, rho, rho_new;
        long double eps;

        // Copy RHS into initial residual.
        m_r_A.template Copy<MemSpace>(in);

        // Apply preconditioner.
        m_assmbScatrZeroDirOp->Apply(m_r_A, m_wk);
        this->m_precon->Apply(m_wk, m_w_A);

        // Reset device memory.
        auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // <r_{0}, r_{0}>
        ddot<ExecSpace>(m_wk, m_r_A, exchange + 2);

        // Calculate rhs magnitude.
        m_assmbScatrOp->Apply(m_r_A, m_wk);
        ddot<ExecSpace>(in, m_wk, exchange + 3);

        // Communication.
        m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                       Nektar::LibUtilities::ReduceSum);

        // Host-to-device copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        eps          = exchangeHost[2];
        rhsMagnitude = (exchangeHost[3] > 1.0e-6) ? exchangeHost[3] : 1.0;

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Perform the method-specific matrix-vector multiply operation.
        this->m_lhs->Apply(m_w_A, m_s_A);

        m_robBndCondOp->Apply(m_w_A, m_s_A);

        // Reset device memory.
        m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // <r_{1}, w_{1}>
        ddot<ExecSpace>(m_r_A, m_w_A, exchange + 0);

        // <s_{1}, w_{1}>
        ddot<ExecSpace>(m_s_A, m_w_A, exchange + 1);

        // Communication.
        m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                       Nektar::LibUtilities::ReduceSum);

        // Host-to-device copy.
        exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        // Initialize search direction.
        rho             = exchangeHost[0];
        mu              = exchangeHost[1];
        beta            = 0.0;
        alpha           = rho / mu;
        totalIterations = 1;

        while (true)
        {
            if (totalIterations > m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Compute new search direction p_k.
            daxpy<ExecSpace>(beta, m_p_A, m_w_A, m_p_A);

            // Compute new search direction q_k.
            daxpy<ExecSpace>(beta, m_q_A, m_s_A, m_q_A);

            // Update solution x_{k+1}.
            daxpy<ExecSpace>(alpha, m_p_A, out, out);

            // Update residual vector r_{k+1}.
            daxpy<ExecSpace>(-alpha, m_q_A, m_r_A, m_r_A);

            // Apply preconditioner.
            m_assmbScatrZeroDirOp->Apply(m_r_A, m_wk);
            this->m_precon->Apply(m_wk, m_w_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_w_A, m_s_A);

            m_robBndCondOp->Apply(m_w_A, m_s_A);

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // <r_{k+1}, w_{k+1}>
            ddot<ExecSpace>(m_r_A, m_w_A, exchange + 0);

            // <s_{k+1}, w_{k+1}>
            ddot<ExecSpace>(m_s_A, m_w_A, exchange + 1);

            // <r_{k+1}, r_{k+1}>
            ddot<ExecSpace>(m_wk, m_r_A, exchange + 2);

            // Communication.
            m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                           Nektar::LibUtilities::ReduceSum);

            // Host-to-device copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            rho_new = exchangeHost[0];
            mu      = exchangeHost[1];
            eps     = exchangeHost[2];

            ++totalIterations;

            // Test if norm is within tolerance.
            if (eps < m_tol * m_tol * rhsMagnitude)
            {
                if (m_root)
                {
                    std::cout << "iterations: " << totalIterations
                              << " eps: " << std::sqrt((double)eps)
                              << " rhs_mag: " << rhsMagnitude << std::endl;
                }
                break;
            }

            // Compute search direction and solution coefficients.
            beta  = rho_new / rho;
            alpha = rho_new / (mu - rho_new * beta / alpha);
            rho   = rho_new;
        }
    }
};

} // namespace Nektar::Operators::detail
