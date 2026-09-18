///////////////////////////////////////////////////////////////////////////////
//
// File: RichardsonOpImpl.hpp
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

#include "SolverCore/GlobalLinSysOps/LinearSolvers/Richardson/RichardsonOp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class RichardsonOpImpl : public RichardsonOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    RichardsonOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : RichardsonOp<TData>(expansionList, components),
          m_w(LibUtilities::Field<TData, FieldState::Coeff>(
              "RichardsonOp w",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r(LibUtilities::Field<TData, FieldState::Coeff>(
              "RichardsonOp r",
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
        m_scale = session->DefinesParameter("RichardsonRelaxation")
                      ? session->GetParameter("RichardsonRelaxation")
                      : 0.25;

        ASSERTL0(!this->m_rightPreconditioner,
                 "RichardsonOpImpl: Only left preconditioner is supported");
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<RichardsonOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_w;
    LibUtilities::Field<TData, FieldState::Coeff> m_r;

    TData m_scale = 0.0;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // Convergence parameters.
        this->m_niter = 0;
        TData rhsMagnitude, eps;

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

        // Apply preconditioner - output is assembled
        this->m_precon->Apply(m_r, m_w);

        // Iteration >= 1
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

            // Update solution.
            Math::daxpy<ExecSpace>(m_scale, m_w, out, out);

            // This is A*x
            this->m_lhs->Apply(out, m_r);
            this->m_robBndCondOp->Apply(out, m_r);

            // This is r = b-A*x
            Math::sub<ExecSpace>(in, m_r, m_r);

            // This is D^-1 * r
            this->m_assmbScatrZeroDirOp->Apply(m_r);
            this->m_precon->Apply(m_r, m_w);

            // <r_{k+1}, r_{k+1}>
            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, LibUtilities::ReduceSum);

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

} // namespace Nektar::SolverCore::detail
