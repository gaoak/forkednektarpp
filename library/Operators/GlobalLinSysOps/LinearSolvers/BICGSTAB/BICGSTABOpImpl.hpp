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

#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
#include "Operators/GlobalLinSysOps/LinearSolvers/BICGSTAB/BICGSTABOp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class BICGSTABOpImpl : public BICGSTABOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BICGSTABOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : BICGSTABOp<TData>(expansionList, components),
          m_p_A(Field<TData, FieldState::Coeff>(
              "BICGSTABOp p_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_v_A(Field<TData, FieldState::Coeff>(
              "BICGSTABOp v_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_h_A(Field<TData, FieldState::Coeff>(
              "BICGSTABOp h_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_z_A(Field<TData, FieldState::Coeff>(
              "BICGSTABOp z_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_t_A(Field<TData, FieldState::Coeff>(
              "BICGSTABOp t_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r_A(Field<TData, FieldState::Coeff>(
              "BICGSTABOp r_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_rtilde_A(Field<TData, FieldState::Coeff>(
              "BICGSTABOp rtilde_A",
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
        return std::make_unique<BICGSTABOpImpl<ExecSpace, TData>>(expansionList,
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

    Field<TData, FieldState::Coeff> m_p_A;
    Field<TData, FieldState::Coeff> m_v_A;
    Field<TData, FieldState::Coeff> m_h_A;
    Field<TData, FieldState::Coeff> m_z_A;
    Field<TData, FieldState::Coeff> m_t_A;
    Field<TData, FieldState::Coeff> m_r_A;
    Field<TData, FieldState::Coeff> m_rtilde_A;

    TData m_tol            = 0.0;
    unsigned int m_maxIter = 0;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, alpha, beta = 0.0, rho_new, rho, eps;
        TData omega = 0.0, omega0, omega1;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);

        // Calculate inital rhs magnitude.
        m_r_A.template Copy<MemSpace>(in);
        m_assmbScatrOp->Apply(m_r_A);
        rhsMagnitude = m_math.ddot(in, m_r_A);
        m_rowComm->AllReduce(rhsMagnitude, Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r_A.template Copy<MemSpace>(in);
        m_assmbScatrZeroDirOp->Apply(m_r_A);
        m_rtilde_A.template Copy<MemSpace>(m_r_A);

        eps = m_math.ddot(in, m_r_A);
        m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        rho_new = m_math.ddot(m_rtilde_A, m_r_A);
        m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
        m_p_A.template Copy<MemSpace>(m_r_A);
        while (true)
        {
            if (totalIterations > m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            if (totalIterations > 0)
            {
                daxpy<ExecSpace>(-omega, m_v_A, m_p_A, m_p_A);
                daxpy<ExecSpace>(beta, m_p_A, m_r_A, m_p_A);
            }

            // Apply preconditioner.
            this->m_precon->Apply(m_p_A, m_h_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_h_A, m_v_A);
            m_robBndCondOp->Apply(m_h_A, m_v_A);
            m_assmbScatrZeroDirOp->Apply(m_v_A);

            // <s_{k+1}, r_{k+1}>
            alpha = m_math.ddot(m_v_A, m_rtilde_A);
            m_rowComm->AllReduce(alpha, LibUtilities::ReduceSum);
            alpha = rho_new / alpha;

            daxpy<ExecSpace>(alpha, m_h_A, out, m_h_A);
            daxpy<ExecSpace>(-alpha, m_v_A, m_r_A, m_r_A);

            // Test if norm is within tolerance.
            eps = m_math.ddot(m_r_A, m_r_A);
            m_rowComm->AllReduce(eps, LibUtilities::ReduceSum);
            if (eps < m_tol * m_tol * rhsMagnitude)
            {
                out.template Copy<MemSpace>(m_h_A);
                if (m_root)
                {
                    std::cout << "iterations: " << totalIterations
                              << " eps: " << std::sqrt(eps)
                              << " rhs_mag: " << rhsMagnitude << std::endl;
                }
                break;
            }

            // Apply preconditioner.
            this->m_precon->Apply(m_r_A, m_z_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_z_A, m_t_A);
            m_robBndCondOp->Apply(m_z_A, m_t_A);
            m_assmbScatrZeroDirOp->Apply(m_t_A);

            omega0 = m_math.ddot(m_r_A, m_t_A);
            m_rowComm->AllReduce(omega0, LibUtilities::ReduceSum);
            omega1 = m_math.ddot(m_t_A, m_t_A);
            m_rowComm->AllReduce(omega1, LibUtilities::ReduceSum);

            omega = omega0 / omega1;

            daxpy<ExecSpace>(omega, m_z_A, m_h_A, out);
            daxpy<ExecSpace>(-omega, m_t_A, m_r_A, m_r_A);

            ++totalIterations;

            // Test if norm is within tolerance.
            eps = m_math.ddot(m_r_A, m_r_A);
            m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);
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

            // Update coefficients.
            rho     = rho_new;
            rho_new = m_math.ddot(m_rtilde_A, m_r_A);
            m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
            beta = rho_new / rho * (alpha / omega);
        }
    }
};

} // namespace Nektar::Operators::detail
