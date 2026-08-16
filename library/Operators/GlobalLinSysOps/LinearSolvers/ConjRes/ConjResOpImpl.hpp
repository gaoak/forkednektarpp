///////////////////////////////////////////////////////////////////////////////
//
// File: ConjResOpImpl.hpp
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

#include "Operators/GlobalLinSysOps/LinearSolvers/ConjRes/ConjResOp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class ConjResOpImpl : public ConjResOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ConjResOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : ConjResOp<TData>(expansionList, components),
          m_w(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjRes w",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_s(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjRes s",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjRes r",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_q(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjRes q",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_p(LibUtilities::Field<TData, FieldState::Coeff>(
              "ConjRes p",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1))
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

        ASSERTL0(!this->m_rightPreconditioner,
                 "ConjResOpImpl: Only left preconditioner is supported");
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<ConjResOpImpl<ExecSpace, TData>>(expansionList,
                                                                 components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_w;
    LibUtilities::Field<TData, FieldState::Coeff> m_s;
    LibUtilities::Field<TData, FieldState::Coeff> m_r;
    LibUtilities::Field<TData, FieldState::Coeff> m_q;
    LibUtilities::Field<TData, FieldState::Coeff> m_p;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // Convergence parameters.
        this->m_niter = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, rho, rho_new;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);
        out.SetInterleaveWidth(in);

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrOp->Apply(m_r);
        rhsMagnitude = this->m_math.ddot(in, m_r);
        this->m_rowComm->AllReduce(rhsMagnitude,
                                   Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrZeroDirOp->Apply(m_r);
        eps = this->m_math.ddot(in, m_r);
        this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Apply preconditioner
        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_r);
        }

        // Iteration >= 1
        alpha   = 0.0;
        rho_new = 1.0;
        while (true)
        {
            if (this->m_niter > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << this->m_niter;
                WARNINGL0(false, msg.str());

                return;
            }

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_r, m_s);
            this->m_robBndCondOp->Apply(m_r, m_s);

            // Update vector.
            if (this->m_niter == 0)
            {
                // Update coefficient.
                rho_new = this->m_math.ddot(m_r, m_s);
                this->m_rowComm->AllReduce(rho_new,
                                           Nektar::LibUtilities::ReduceSum);

                // Update vectors.
                m_p.template Copy<MemSpace>(m_r);
                m_q.template Copy<MemSpace>(m_s);
            }
            else
            {
                // Update coefficient.
                rho     = rho_new;
                rho_new = this->m_math.ddot(m_r, m_s);
                this->m_rowComm->AllReduce(rho_new,
                                           Nektar::LibUtilities::ReduceSum);
                beta = rho_new / rho;

                // Update vectors.
                Math::daxpy<ExecSpace>(beta, m_p, m_r, m_p);
                Math::daxpy<ExecSpace>(beta, m_q, m_s, m_q);
            }

            // Apply preconditioner
            this->m_assmbScatrZeroDirOp->Apply(m_q, m_w);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_w, m_w);
            }

            // Update coefficient.
            alpha = this->m_math.ddot(m_q, m_w);
            this->m_rowComm->AllReduce(alpha, Nektar::LibUtilities::ReduceSum);
            alpha = rho_new / alpha;

            // Update solution.
            Math::daxpy<ExecSpace>(alpha, m_p, out, out);
            Math::daxpy<ExecSpace>(-alpha, m_w, m_r, m_r);

            // Update residual norm.
            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

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
