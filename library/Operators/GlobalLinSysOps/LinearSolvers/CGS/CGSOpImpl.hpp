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

#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
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
          m_q_A(Field<TData, FieldState::Coeff>(
              "CGSOp q_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_w_A(Field<TData, FieldState::Coeff>(
              "CGSOp w_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_s_A(Field<TData, FieldState::Coeff>(
              "CGSOp s_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_u_A(Field<TData, FieldState::Coeff>(
              "CGSOp w_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_p_A(Field<TData, FieldState::Coeff>(
              "CGSOp p_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r_A(Field<TData, FieldState::Coeff>(
              "CGSOp r_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_rtilde_A(Field<TData, FieldState::Coeff>(
              "CGSOp r_A",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_mask(Field<std::uint8_t, FieldState::Coeff>(
              "ConjGrad mask",
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

        // Fill mask.
        auto maskptr =
            this->m_dataWarehouse->template GetData<NektarSpaces::HostSpace>(
                LocalToGlobalMaskKey<TData>());
        unsigned cnt = 0;
        for (unsigned blk = 0; blk < m_mask.GetBlocks().size(); ++blk)
        {
            auto &block = m_mask.GetBlocks()[blk];
            auto ptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned i = 0; i < block.CompSize(); ++i)
            {
                ptr[i] = maskptr[cnt++];
            }
        }
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
    LibUtilities::CommSharedPtr m_rowComm = nullptr;
    std::unique_ptr<AssmbScatrOpImpl<ExecSpace, TData>> m_assmbScatrOp;
    std::unique_ptr<AssmbScatrZeroDirOpImpl<ExecSpace, TData>>
        m_assmbScatrZeroDirOp;
    bool m_root;

    Math m_math;

    std::shared_ptr<RobBndCondOp<TData>> m_robBndCondOp;

    Field<TData, FieldState::Coeff> m_q_A;
    Field<TData, FieldState::Coeff> m_w_A;
    Field<TData, FieldState::Coeff> m_s_A;
    Field<TData, FieldState::Coeff> m_u_A;
    Field<TData, FieldState::Coeff> m_p_A;
    Field<TData, FieldState::Coeff> m_r_A;
    Field<TData, FieldState::Coeff> m_rtilde_A;
    Field<std::uint8_t, FieldState::Coeff> m_mask;

    TData m_tol            = 0.0;
    unsigned int m_maxIter = 0;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Reshape mask if required.
        for (unsigned blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            auto &inblk   = in.GetBlocks()[blk];
            auto &maskblk = m_mask.GetBlocks()[blk];

            if (inblk.GetInterleaveWidth() != maskblk.GetInterleaveWidth())
            {
                auto maskPtr = maskblk.template GetPtr<MemSpace, ReadWrite>();
                auto numComp = maskblk.GetNumComponents();

                for (unsigned nc = 0; nc < numComp; ++nc)
                {
                    ReshapeStorage<ExecSpace>(
                        inblk.GetInterleaveWidth(),
                        maskblk.GetInterleaveWidth(),
                        maskblk.GetNumElementsWithPadding(),
                        maskblk.GetNumData(),
                        maskPtr + nc * maskblk.CompSize());
                }
                maskblk.template SetInterleaveWidth<TData>(
                    inblk.GetInterleaveWidth());
            }
        }

        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, alpha, beta, rho_new, rho, eps;

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
        while (true)
        {
            if (totalIterations > m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Compute new search direction.
            if (totalIterations == 0)
            {
                m_u_A.template Copy<MemSpace>(m_r_A);
                m_p_A.template Copy<MemSpace>(m_u_A);
            }
            else
            {
                daxpy<ExecSpace>(beta, m_q_A, m_r_A, m_u_A);
                daxpy<ExecSpace>(beta, m_p_A, m_q_A, m_p_A);
                daxpy<ExecSpace>(beta, m_p_A, m_u_A, m_p_A);
            }

            // Apply preconditioner.
            this->m_precon->Apply(m_p_A, m_w_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_w_A, m_s_A);
            m_robBndCondOp->Apply(m_w_A, m_s_A);
            m_assmbScatrZeroDirOp->Apply(m_s_A);

            // <s_{k+1}, r_{k+1}>
            alpha = m_math.ddot(m_s_A, m_rtilde_A);
            m_rowComm->AllReduce(alpha, LibUtilities::ReduceSum);
            alpha = rho_new / alpha;

            daxpy<ExecSpace>(-alpha, m_s_A, m_u_A, m_q_A);
            add<ExecSpace>(m_u_A, m_q_A, m_u_A);

            // Apply preconditioner.
            this->m_precon->Apply(m_u_A, m_w_A);

            // Update solution.
            daxpy<ExecSpace>(alpha, m_w_A, out, out);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_w_A, m_s_A);
            m_robBndCondOp->Apply(m_w_A, m_s_A);
            m_assmbScatrZeroDirOp->Apply(m_s_A);

            daxpy<ExecSpace>(-alpha, m_s_A, m_r_A, m_r_A);

            // Update coefficients.
            rho     = rho_new;
            rho_new = m_math.ddot(m_rtilde_A, m_r_A);
            m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
            eps = m_math.ddot(m_mask, m_r_A, m_r_A);
            m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);
            beta = rho_new / rho;

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
