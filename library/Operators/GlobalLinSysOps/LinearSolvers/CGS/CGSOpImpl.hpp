///////////////////////////////////////////////////////////////////////////////
//
// File: CGSOpImpl.hpp
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

#include "Operators/GlobalLinSysOps/LinearSolvers/CGS/CGSOp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class CGSOpImpl : public CGSOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    CGSOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
              const std::vector<std::string> &components)
        : CGSOp<TData>(expansionList, components),
          m_q(Field<TData, FieldState::Coeff>(
              "CGSOp q",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_w(Field<TData, FieldState::Coeff>(
              "CGSOp w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_s(Field<TData, FieldState::Coeff>(
              "CGSOp s",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_u(Field<TData, FieldState::Coeff>(
              "CGSOp u",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_p(Field<TData, FieldState::Coeff>(
              "CGSOp p",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r(Field<TData, FieldState::Coeff>(
              "CGSOp r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_rtilde(Field<TData, FieldState::Coeff>(
              "CGSOp rtilde",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1))
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
        return std::make_unique<CGSOpImpl<ExecSpace, TData>>(expansionList,
                                                             components);
    }

protected:
    Field<TData, FieldState::Coeff> m_q;
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_s;
    Field<TData, FieldState::Coeff> m_u;
    Field<TData, FieldState::Coeff> m_p;
    Field<TData, FieldState::Coeff> m_r;
    Field<TData, FieldState::Coeff> m_rtilde;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, rho, rho_new;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);

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

        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_r);
        }

        eps = this->m_math.ddot(in, m_r);
        this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        m_rtilde.template Copy<MemSpace>(m_r);
        rho_new = this->m_math.ddot(m_rtilde, m_r);
        this->m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
        while (true)
        {
            if (totalIterations > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Update vectors.
            if (totalIterations == 0)
            {
                m_u.template Copy<MemSpace>(m_r);
                m_p.template Copy<MemSpace>(m_u);
            }
            else
            {
                daxpy<ExecSpace>(beta, m_q, m_r, m_u);
                daxpy<ExecSpace>(beta, m_p, m_q, m_p);
                daxpy<ExecSpace>(beta, m_p, m_u, m_p);
            }

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp = (this->m_rightPreconditioner) ? m_w : m_p;
            if (this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_p, tmp);
            }
            this->m_lhs->Apply(tmp, m_s);
            this->m_robBndCondOp->Apply(tmp, m_s);
            this->m_assmbScatrZeroDirOp->Apply(m_s);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_s, m_s);
            }

            // Update coefficients.
            alpha = this->m_math.ddot(m_s, m_rtilde);
            this->m_rowComm->AllReduce(alpha, LibUtilities::ReduceSum);
            alpha = rho_new / alpha;

            // Update vectors.
            daxpy<ExecSpace>(-alpha, m_s, m_u, m_q);
            add<ExecSpace>(m_u, m_q, m_w);

            // Perform the method-specific matrix-vector multiply operation.
            if (this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_w, m_w);
            }
            this->m_lhs->Apply(m_w, m_s);
            this->m_robBndCondOp->Apply(m_w, m_s);
            this->m_assmbScatrZeroDirOp->Apply(m_s);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_s, m_s);
            }

            // Update solution.
            daxpy<ExecSpace>(alpha, m_w, out, out);

            // Update residual.
            daxpy<ExecSpace>(-alpha, m_s, m_r, m_r);

            // Update coefficients.
            rho     = rho_new;
            rho_new = this->m_math.ddot(m_rtilde, m_r);
            this->m_rowComm->AllReduce(rho_new,
                                       Nektar::LibUtilities::ReduceSum);
            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);
            beta = rho_new / rho;

            ++totalIterations;

            // Test if norm is within tolerance.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                if (this->m_root)
                {
                    std::cout << this->name
                              << " iterations made = " << totalIterations
                              << " using tolerance of " << this->m_tol
                              << " error = " << std::sqrt(eps / rhsMagnitude)
                              << " rhs_mag = " << std::sqrt(rhsMagnitude)
                              << std::endl;
                }
                break;
            }
        }
    }
};

} // namespace Nektar::Operators::detail
