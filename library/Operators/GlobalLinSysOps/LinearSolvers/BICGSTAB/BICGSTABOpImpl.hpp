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
          m_p(Field<TData, FieldState::Coeff>(
              "BICGSTABOp p",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_v(Field<TData, FieldState::Coeff>(
              "BICGSTABOp v",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_w(Field<TData, FieldState::Coeff>(
              "BICGSTABOp w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_z(Field<TData, FieldState::Coeff>(
              "BICGSTABOp z",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r(Field<TData, FieldState::Coeff>(
              "BICGSTABOp r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_rtilde(Field<TData, FieldState::Coeff>(
              "BICGSTABOp rtilde",
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

    Field<TData, FieldState::Coeff> m_p;
    Field<TData, FieldState::Coeff> m_v;
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_z;
    Field<TData, FieldState::Coeff> m_r;
    Field<TData, FieldState::Coeff> m_rtilde;

    TData m_tol            = 0.0;
    unsigned int m_maxIter = 0;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, rho, rho_new;
        TData omega, omega0, omega1;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        m_assmbScatrOp->Apply(m_r);
        rhsMagnitude = m_math.ddot(in, m_r);
        m_rowComm->AllReduce(rhsMagnitude, Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        m_assmbScatrZeroDirOp->Apply(m_r);
        eps = m_math.ddot(in, m_r);
        m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        beta  = 0.0;
        omega = 0.0;
        m_rtilde.template Copy<MemSpace>(m_r);
        m_p.template Copy<MemSpace>(m_r);
        rho_new = m_math.ddot(m_rtilde, m_r);
        m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
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

            // Apply preconditioner.
            this->m_precon->Apply(m_p, m_w);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_w, m_v);
            m_robBndCondOp->Apply(m_w, m_v);
            m_assmbScatrZeroDirOp->Apply(m_v);

            // Update coefficients.
            alpha = m_math.ddot(m_v, m_rtilde);
            m_rowComm->AllReduce(alpha, LibUtilities::ReduceSum);
            alpha = rho_new / alpha;

            // Update solution.
            daxpy<ExecSpace>(alpha, m_w, out, out);
            daxpy<ExecSpace>(-alpha, m_v, m_r, m_r);

            // Test if norm is within tolerance.
            eps = m_math.ddot(m_r, m_r);
            m_rowComm->AllReduce(eps, LibUtilities::ReduceSum);
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

            // Apply preconditioner.
            this->m_precon->Apply(m_r, m_w);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_w, m_z);
            m_robBndCondOp->Apply(m_w, m_z);
            m_assmbScatrZeroDirOp->Apply(m_z);

            // Update coefficients.
            omega0 = m_math.ddot(m_r, m_z);
            omega1 = m_math.ddot(m_z, m_z);
            m_rowComm->AllReduce(omega0, LibUtilities::ReduceSum);
            m_rowComm->AllReduce(omega1, LibUtilities::ReduceSum);
            omega = omega0 / omega1;

            // Update solution.
            daxpy<ExecSpace>(omega, m_w, out, out);
            daxpy<ExecSpace>(-omega, m_z, m_r, m_r);

            ++totalIterations;

            // Test if norm is within tolerance.
            eps = m_math.ddot(m_r, m_r);
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
            rho_new = m_math.ddot(m_rtilde, m_r);
            m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
            beta = rho_new / rho * (alpha / omega);
        }
    }
};

} // namespace Nektar::Operators::detail
