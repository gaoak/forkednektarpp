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

#include "Operators/GlobalLinSysOps/LinearSolvers/GCR/GCROp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class GCROpImpl : public GCROp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    GCROpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
              const std::vector<std::string> &components)
        : GCROp<TData>(expansionList, components),
          m_r(Field<TData, FieldState::Coeff>(
              "GCR r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1))
    {
        this->template SetLinearSolver<ExecSpace>();

        auto session = expansionList->GetSession();
        session->LoadParameter("LinSysMaxStorage", m_LinSysMaxStorage, 50);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<GCROpImpl<ExecSpace, TData>>(expansionList,
                                                             components);
    }

protected:
    Field<TData, FieldState::Coeff> m_r;
    std::vector<Field<TData, FieldState::Coeff>> m_Q;
    std::vector<Field<TData, FieldState::Coeff>> m_P;

    unsigned int m_LinSysMaxStorage;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Generalized Conjugate Residual algorithm.
        // Implementation follow the GCR(k) algorithm in:
        //
        // Eisenstat, Stanley C., Howard C. Elman, and Martin H. Schultz.
        // "Variational iterative methods for nonsymmetric systems of linear
        // equations." SIAM Journal on Numerical Analysis 20, no. 2 (1983):
        // 345-357.

        // Convergence parameters.
        unsigned int totalIterations = 0, ii = 0;
        TData rhsMagnitude, eps, alpha;
        std::vector<TData> scale(m_LinSysMaxStorage), beta(m_LinSysMaxStorage);

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

        if (m_P.size() == 0)
        {
            m_P.push_back(Field<TData, FieldState::Coeff>(
                "GCR P0",
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));
            m_Q.push_back(Field<TData, FieldState::Coeff>(
                "GCR Q0",
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));
        }

        // Iteration >= 1
        while (true)
        {
            ii = totalIterations % m_LinSysMaxStorage;
            if (totalIterations > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Apply preconditioner
            this->m_precon->Apply(m_r, m_P[ii]);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_P[ii], m_Q[ii]);
            this->m_robBndCondOp->Apply(m_P[ii], m_Q[ii]);
            this->m_assmbScatrZeroDirOp->Apply(m_Q[ii]);

            // Update vector.
            if (totalIterations > 0)
            {
                for (unsigned int i = 0; i < ii; i++)
                {
                    beta[i] = this->m_math.ddot(m_Q[ii], m_Q[i]);
                }
                this->m_rowComm->AllReduce(beta,
                                           Nektar::LibUtilities::ReduceSum);
                for (unsigned int i = ii; i > 0; i--)
                {
                    daxpy<ExecSpace>(-beta[i - 1] / scale[i - 1], m_P[i - 1],
                                     m_P[ii], m_P[ii]);
                    daxpy<ExecSpace>(-beta[i - 1] / scale[i - 1], m_Q[i - 1],
                                     m_Q[ii], m_Q[ii]);
                }
            }

            // Update coefficient.
            alpha = this->m_math.ddot(m_Q[ii], m_r);
            this->m_rowComm->AllReduce(alpha, Nektar::LibUtilities::ReduceSum);
            scale[ii] = this->m_math.ddot(m_Q[ii], m_Q[ii]);
            this->m_rowComm->AllReduce(scale[ii],
                                       Nektar::LibUtilities::ReduceSum);
            alpha /= scale[ii];

            // Update solutions.
            daxpy<ExecSpace>(alpha, m_P[ii], out, out);
            daxpy<ExecSpace>(-alpha, m_Q[ii], m_r, m_r);

            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

            ++totalIterations;

            // Test if norm is within tolerance.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                if (this->m_root)
                {
                    std::cout << "iterations: " << totalIterations
                              << " eps: " << std::sqrt(eps)
                              << " rhs_mag: " << rhsMagnitude << std::endl;
                }
                return;
            }

            // Allocate memory, if necessary.
            if (m_P.size() == totalIterations &&
                m_P.size() < m_LinSysMaxStorage)
            {
                m_P.push_back(Field<TData, FieldState::Coeff>(
                    "GCR P" + std::to_string(totalIterations),
                    GetBlockAttributes<TData, FieldState::Coeff>(
                        this->m_expansionList),
                    this->m_components, 1));
                m_Q.push_back(Field<TData, FieldState::Coeff>(
                    "GCR Q" + std::to_string(totalIterations),
                    GetBlockAttributes<TData, FieldState::Coeff>(
                        this->m_expansionList),
                    this->m_components, 1));
            }
        }
    }
};

} // namespace Nektar::Operators::detail
