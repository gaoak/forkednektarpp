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

#include "Operators/GlobalLinSysOps/ConjGrad/ConjGradKernels.hpp"

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
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              1)),
          m_s_A(Field<TData, FieldState::Coeff>(
              "ConjGrad s_A",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              1)),
          m_r_A(Field<TData, FieldState::Coeff>(
              "ConjGrad r_A",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              1)),
          m_q_A(Field<TData, FieldState::Coeff>(
              "ConjGrad wk",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              1)),
          m_p_A(Field<TData, FieldState::Coeff>(
              "ConjGrad wk",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              1)),
          m_mask(Field<std::uint8_t, FieldState::Coeff>(
              "ConjGrad mask",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              1)),
          m_vExchange(MemoryRegion<TData>(3, ePinned))
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
            for (unsigned i = 0; i < block.size(); ++i)
            {
                ptr[i] = maskptr[cnt++];
            }
        }
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
    Field<TData, FieldState::Coeff> m_q_A;
    Field<TData, FieldState::Coeff> m_p_A;
    Field<std::uint8_t, FieldState::Coeff> m_mask;

    MemoryRegion<TData> m_vExchange;

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
                        maskblk.GetNumData(), maskPtr + nc * maskblk.size());
                }
                maskblk.template SetInterleaveWidth<TData>(
                    inblk.GetInterleaveWidth());
            }
        }

        // Convergence parameters.
        unsigned int totalIterations = 0;
        TData rhsMagnitude, mu;
        TData alpha, beta, rho, rho_new;
        long double eps;

        // Reset device memory.
        auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);
        m_p_A.template Initialize<MemSpace>(0);
        m_q_A.template Initialize<MemSpace>(0);

        // Calculate inital rhs magnitude.
        m_r_A.template Copy<MemSpace>(in);
        m_assmbScatrOp->Apply(m_r_A);
        ddot<ExecSpace>(in, m_r_A, exchange + 0);

        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r_A.template Copy<MemSpace>(in);
        m_assmbScatrZeroDirOp->Apply(m_r_A);
        ddot<ExecSpace>(in, m_r_A, exchange + 1);

        // Communication.
        m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                       Nektar::LibUtilities::ReduceSum);

        // Device-to-host copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        rhsMagnitude = (exchangeHost[0] > 1.0e-6) ? exchangeHost[0] : 1.0;
        eps          = exchangeHost[1];

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Reset device memory.
        m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // Apply preconditioner - output is assembled
        this->m_precon->Apply(m_r_A, m_w_A);

        // <r_{1}, w_{1}>
        ddot<ExecSpace>(m_mask, m_r_A, m_w_A, exchange + 0);

        // Perform the method-specific matrix-vector multiply operation.
        this->m_lhs->Apply(m_w_A, m_s_A);
        m_robBndCondOp->Apply(m_w_A, m_s_A);

        // <w_{1}, s_{1}> - standard ddot since m_s_A is in not assembled
        ddot<ExecSpace>(m_w_A, m_s_A, exchange + 1);

        // Communication.
        m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                       Nektar::LibUtilities::ReduceSum);

        // Device-to-host copy.
        exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

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

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // Assemble matrix output from previous matrix-vector multiply
            // could be moved around loop if optimal elsewhere.
            m_assmbScatrZeroDirOp->Apply(m_s_A);

            // Compute new search direction.
            UpdateConjGradSearchDirection<ExecSpace>(alpha, beta, m_w_A, m_s_A,
                                                     m_p_A, m_q_A, m_r_A, out);

            // <r_{k+1}, r_{k+1}>
            ddot<ExecSpace>(m_mask, m_r_A, m_r_A, exchange + 0);

            // Apply preconditioner - output is assumeed holding global dof
            this->m_precon->Apply(m_r_A, m_w_A);

            // <r_{k+1}, w_{k+1}>
            ddot<ExecSpace>(m_mask, m_r_A, m_w_A, exchange + 1);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_w_A, m_s_A);

            m_robBndCondOp->Apply(m_w_A, m_s_A);

            // <w_{k+1}, s_{k+1}>
            ddot<ExecSpace>(m_w_A, m_s_A, exchange + 2);

            // Communication.
            m_rowComm->AllReduce<MemSpace>(m_vExchange,
                                           Nektar::LibUtilities::ReduceSum);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            eps     = exchangeHost[0];
            rho_new = exchangeHost[1];
            mu      = exchangeHost[2];

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
