///////////////////////////////////////////////////////////////////////////////
//
// File: GCROpImpl.hpp
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

#include "SolverCore/GlobalLinSysOps/LinearSolvers/GCR/GCROp.hpp"
#include "SolverCore/GlobalLinSysOps/MultiFieldHelper/MultiFieldHelper.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class GCROpImpl : public GCROp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    GCROpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
              const std::vector<std::string> &components)
        : GCROp<TData>(expansionList, components),
          m_r(LibUtilities::Field<TData, FieldState::Coeff>(
              "GCR r",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_Q("GCR Q",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components),
          m_P("GCR P",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components)
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
        this->m_LinSysMaxStorage =
            session->DefinesParameter("LinSysMaxStorage")
                ? session->GetParameter("LinSysMaxStorage")
                : 50;

        ASSERTL0(!this->m_rightPreconditioner,
                 "GCROpImpl: Only left preconditioner is supported");

        // Allocate array storage.
        m_vExchange =
            LibUtilities::MemoryRegion<TData>(m_LinSysMaxStorage, eHostPinned);
        m_coeffs =
            LibUtilities::MemoryRegion<TData>(m_LinSysMaxStorage, eHostPinned);
        m_Q.ResizeNumField(1);
        m_P.ResizeNumField(1);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<GCROpImpl<ExecSpace, TData>>(expansionList,
                                                             components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_r;
    LibUtilities::MultiField<TData, FieldState::Coeff> m_Q;
    LibUtilities::MultiField<TData, FieldState::Coeff> m_P;
    LibUtilities::MemoryRegion<TData> m_vExchange;
    LibUtilities::MemoryRegion<TData> m_coeffs; ///< Orthogonalisation
                                                ///< coefficients.

    unsigned int m_LinSysMaxStorage;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // Generalized Conjugate Residual algorithm.
        // Implementation follow the GCR(k) algorithm in:
        //
        // Eisenstat, Stanley C., Howard C. Elman, and Martin H. Schultz.
        // "Variational iterative methods for nonsymmetric systems of linear
        // equations." SIAM Journal on Numerical Analysis 20, no. 2 (1983):
        // 345-357.

        // Convergence parameters.
        this->m_niter   = 0;
        unsigned int ii = 0;
        TData rhsMagnitude, eps, alpha;
        std::vector<TData> scale(m_LinSysMaxStorage);

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);
        out.SetInterleaveWidth(in);

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrOp->Apply(m_r);
        rhsMagnitude = this->m_math.ddot(in, m_r);
        this->m_rowComm->AllReduce(rhsMagnitude,
                                   Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = this->GetRhsMagnitude(rhsMagnitude);

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero
        // Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrZeroDirOp->Apply(m_r);

        eps = this->m_math.ddot(m_r, m_r);
        this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        while (true)
        {
            ii = this->m_niter % m_LinSysMaxStorage;
            if (this->m_niter > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << this->m_niter
                    << ". Increase NekLinSysMaxIterations in the session "
                       "PARAMETERS section to allow more iterations.";
                NEKERROR(ErrorUtil::efatal, msg.str());

                return;
            }

            // Apply preconditioner
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_r, m_P[ii]);
            }
            else
            {
                m_P[ii].template Copy<MemSpace>(m_r);
            }

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_P[ii], m_Q[ii]);
            this->m_robBndCondOp->Apply(m_P[ii], m_Q[ii]);
            this->m_assmbScatrZeroDirOp->Apply(m_Q[ii]);

            // Update vector: P[ii] and Q[ii] lose their components along
            // the previous directions, with coefficients -beta_i / scale_i,
            // beta_i = (Q[ii], Q[i]).
            if (ii > 0)
            {
                // Reset device memory.
                auto exchange =
                    m_vExchange.template GetPtr<MemSpace, WriteOnly>();
                MultiDot<ExecSpace>(m_Q, 0, ii, m_Q[ii], exchange);

                // Communication.
                this->m_rowComm->template AllReduce<MemSpace>(
                    m_vExchange, Nektar::LibUtilities::ReduceSum);

                // Device-to-host copy.
                auto exchangeHost =
                    m_vExchange
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
                auto coeffsHost =
                    m_coeffs
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned int i = 0; i < ii; i++)
                {
                    coeffsHost[i] = -exchangeHost[i] / scale[i];
                }

                auto coeffs = m_coeffs.template GetPtr<MemSpace, ReadOnly>();
                MultiAxpy<ExecSpace>(1.0, m_P, 0, ii, coeffs, 1.0, m_P[ii]);
                MultiAxpy<ExecSpace>(1.0, m_Q, 0, ii, coeffs, 1.0, m_Q[ii]);
            }

            // Update coefficient.
            alpha = this->m_math.ddot(m_Q[ii], m_r);
            this->m_rowComm->AllReduce(alpha, Nektar::LibUtilities::ReduceSum);
            scale[ii] = this->m_math.ddot(m_Q[ii], m_Q[ii]);
            this->m_rowComm->AllReduce(scale[ii],
                                       Nektar::LibUtilities::ReduceSum);
            alpha /= scale[ii];

            // Update solutions.
            Math::daxpy<ExecSpace>(alpha, m_P[ii], out, out);
            Math::daxpy<ExecSpace>(-alpha, m_Q[ii], m_r, m_r);

            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

            ++this->m_niter;

            // Test if norm is within tolerance.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(this->name, "error",
                                         std::sqrt(eps / rhsMagnitude),
                                         rhsMagnitude);
                return;
            }

            // Allocate memory, if necessary.
            if (this->m_niter < m_LinSysMaxStorage)
            {
                m_P.ResizeNumField(this->m_niter + 1);
                m_Q.ResizeNumField(this->m_niter + 1);
            }
        }
    }
};

} // namespace Nektar::SolverCore::detail
