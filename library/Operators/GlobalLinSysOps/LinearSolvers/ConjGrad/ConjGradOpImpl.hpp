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

#include "Operators/GlobalLinSysOps/LinearSolvers/ConjGrad/ConjGradOp.hpp"

#include "Operators/GlobalLinSysOps/LinearSolvers/ConjGrad/ConjGradKernels.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class ConjGradOpImpl : public ConjGradOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ConjGradOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : ConjGradOp<TData>(expansionList, components),
          m_w(Field<TData, FieldState::Coeff>(
              "ConjGrad w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_s(Field<TData, FieldState::Coeff>(
              "ConjGrad s",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r(Field<TData, FieldState::Coeff>(
              "ConjGrad r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_q(Field<TData, FieldState::Coeff>(
              "ConjGrad q",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_p(Field<TData, FieldState::Coeff>(
              "ConjGrad p",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_mask(Field<std::uint8_t, FieldState::Coeff>(
              "ConjGrad mask",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_vExchange(MemoryRegion<TData>(4, ePinned))
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
        m_flexible = session->DefinesParameter("FlexibleConjugateGradient")
                         ? session->GetParameter("FlexibleConjugateGradient")
                         : false;

        ASSERTL0(!this->m_rightPreconditioner,
                 "ConjGradOpImpl: Only left preconditioner is supported");

        // Fill mask.
        auto maskptr =
            this->m_dataWarehouse->template GetData<NektarSpaces::HostSpace>(
                LocalToGlobalMaskKey<TData>(this->m_components));
        unsigned cnt = 0;
        for (unsigned blk = 0; blk < m_mask.GetBlocks().size(); ++blk)
        {
            auto &block = m_mask.GetBlocks()[blk];
            auto ptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            // The scalar local-to-global mask is identical for each component.
            for (unsigned n = 0;
                 n < block.GetNumComponents() * m_mask.GetNumHomoModes(); ++n)
            {
                for (unsigned i = 0; i < block.CompSize(); ++i)
                {
                    ptr[n * block.CompSize() + i] = maskptr[cnt + i];
                }
            }
            cnt += block.CompSize();
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<ConjGradOpImpl<ExecSpace, TData>>(expansionList,
                                                                  components);
    }

protected:
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_s;
    Field<TData, FieldState::Coeff> m_r;
    Field<TData, FieldState::Coeff> m_q;
    Field<TData, FieldState::Coeff> m_p;
    Field<std::uint8_t, FieldState::Coeff> m_mask;
    MemoryRegion<TData> m_vExchange;

    bool m_flexible;

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
        this->m_niter = 0;
        TData rhsMagnitude, eps;
        TData alpha, beta, rho, rho_new, rho_star, mu;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);

        // Reset device memory.
        auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrOp->Apply(m_r);
        ddot<ExecSpace>(in, m_r, exchange + 1);

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrZeroDirOp->Apply(m_r);
        ddot<ExecSpace>(in, m_r, exchange + 0);

        // Communication.
        this->m_rowComm->template AllReduce<MemSpace>(
            m_vExchange, Nektar::LibUtilities::ReduceSum);

        // Device-to-host copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        rhsMagnitude = (exchangeHost[1] > 1.0e-6) ? exchangeHost[1] : 1.0;
        eps          = exchangeHost[0];

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Apply preconditioner - output is assembled
        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_w);
        }
        else
        {
            m_w.template Copy<MemSpace>(m_r);
        }

        // Iteration >= 1
        alpha    = 1.0;
        beta     = 0.0;
        rho      = 1.0;
        rho_star = 0.0;
        while (true)
        {
            if (this->m_niter > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << this->m_niter;
                WARNINGL0(false, msg.str());

                return;
            }

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            if (this->m_niter == 0)
            {
                m_p.template Initialize<MemSpace>(0);
                m_q.template Initialize<MemSpace>(0);
            }
            else
            {
                // Assemble matrix output from previous matrix-vector multiply
                // could be moved around loop if optimal elsewhere.
                this->m_assmbScatrZeroDirOp->Apply(m_s);

                // Compute new search direction.
                // daxpy<ExecSpace>(beta, m_p, m_w, m_p);
                // daxpy<ExecSpace>(beta, m_q, m_s, m_q);
                // daxpy<ExecSpace>(alpha, m_p, out, out);
                // daxpy<ExecSpace>(-alpha, m_q, m_r, m_r);
                UpdateConjGradSearchDirection<ExecSpace>(alpha, beta, m_w, m_s,
                                                         m_p, m_q, m_r, out);

                // <r_{k+1}, r_{k+1}>
                ddot<ExecSpace>(m_mask, m_r, m_r, exchange + 0);

                if (m_flexible)
                {
                    // <r_{k+1}, w_{k}>
                    ddot<ExecSpace>(m_mask, m_r, m_w, exchange + 3);
                }

                // Apply preconditioner - output is assumeed holding global dof
                if (this->m_leftPreconditioner)
                {
                    this->m_precon->Apply(m_r, m_w);
                }
                else
                {
                    m_w.template Copy<MemSpace>(m_r);
                }
            }

            // <r_{k+1}, w_{k+1}>
            ddot<ExecSpace>(m_mask, m_r, m_w, exchange + 1);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_lhs->Apply(m_w, m_s);
            this->m_robBndCondOp->Apply(m_w, m_s);

            // <w_{k+1}, s_{k+1}>
            ddot<ExecSpace>(m_w, m_s, exchange + 2);

            // Communication.
            this->m_rowComm->template AllReduce<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            eps     = exchangeHost[0];
            rho_new = exchangeHost[1];
            mu      = exchangeHost[2];
            if (m_flexible)
            {
                rho_star = exchangeHost[3];
            }

            ++this->m_niter;

            // Test if norm is within tolerance.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                if (this->m_root)
                {
                    std::cout << this->name
                              << " iterations made = " << this->m_niter
                              << " using tolerance of " << this->m_tol
                              << " error = " << std::sqrt(eps / rhsMagnitude)
                              << " rhs_mag = " << std::sqrt(rhsMagnitude)
                              << std::endl;
                }
                break;
            }

            // Update coefficients.
            beta  = (this->m_niter > 1) ? (rho_new - rho_star) / rho : 0.0;
            alpha = rho_new / (mu - rho_new * beta / alpha);
            rho   = rho_new;
        }
    }
};

} // namespace Nektar::Operators::detail
