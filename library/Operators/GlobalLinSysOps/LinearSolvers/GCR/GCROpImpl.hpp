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

#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
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
        auto session = expansionList->GetSession();

        // Set operators.
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
        session->LoadParameter("LinSysMaxStorage", m_LinSysMaxStorage, 50);
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
        return std::make_unique<GCROpImpl<ExecSpace, TData>>(expansionList,
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

    Field<TData, FieldState::Coeff> m_r;
    std::vector<Field<TData, FieldState::Coeff>> m_Q;
    std::vector<Field<TData, FieldState::Coeff>> m_P;

    TData m_tol            = 0.0;
    unsigned int m_maxIter = 0;
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
        m_assmbScatrOp->Apply(m_r);
        rhsMagnitude = m_math.ddot(in, m_r);
        m_rowComm->AllReduce(rhsMagnitude, Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero
        // Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        m_assmbScatrZeroDirOp->Apply(m_r);

        eps = m_math.ddot(m_r, m_r);
        m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
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
            if (totalIterations > m_maxIter)
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
            m_robBndCondOp->Apply(m_P[ii], m_Q[ii]);
            m_assmbScatrZeroDirOp->Apply(m_Q[ii]);

            // Update vector.
            if (totalIterations > 0)
            {
                for (unsigned int i = 0; i < ii; i++)
                {
                    beta[i] = m_math.ddot(m_Q[ii], m_Q[i]);
                }
                m_rowComm->AllReduce(beta, Nektar::LibUtilities::ReduceSum);
                for (unsigned int i = ii; i > 0; i--)
                {
                    daxpy<ExecSpace>(-beta[i - 1] / scale[i - 1], m_P[i - 1],
                                     m_P[ii], m_P[ii]);
                    daxpy<ExecSpace>(-beta[i - 1] / scale[i - 1], m_Q[i - 1],
                                     m_Q[ii], m_Q[ii]);
                }
            }

            // Update coefficient.
            alpha = m_math.ddot(m_Q[ii], m_r);
            m_rowComm->AllReduce(alpha, Nektar::LibUtilities::ReduceSum);
            scale[ii] = m_math.ddot(m_Q[ii], m_Q[ii]);
            m_rowComm->AllReduce(scale[ii], Nektar::LibUtilities::ReduceSum);
            alpha /= scale[ii];

            // Update solutions.
            daxpy<ExecSpace>(alpha, m_P[ii], out, out);
            daxpy<ExecSpace>(-alpha, m_Q[ii], m_r, m_r);

            eps = m_math.ddot(m_r, m_r);
            m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);

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
