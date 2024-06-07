///////////////////////////////////////////////////////////////////////////////
//
// File: ConjGradImplShared.hpp
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

#include "ConjGradImplBase.hpp"

#include "Operators/CUDAMathKernels.cuh"
#include "Operators/MathKernels.hpp"

#include "Operators/OperatorHelper.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
#if defined(NEKTAR_ENABLE_CUDA)
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
#endif
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)>::type>
class OperatorConjGradImpl
    : public OperatorConjGradImplBase<ExecSpace, Implementation, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorConjGradImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorConjGradImplBase<ExecSpace, Implementation, TData>(
              expansionList)
    {
        m_vExchange = MemoryRegion<TData>::template create<MemSpace>(4);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Set the fields to zero
        out.initialize(0);

        this->m_w_A.initialize(0);
        this->m_s_A.initialize(0);
        this->m_wk.initialize(0);

        this->m_p_A.initialize(0);
        this->m_q_A.initialize(0);

        m_vExchange.initialize(0);

        const TData *inPtr = in.template GetConstPtr<MemSpace>();
        TData *outPtr      = out.template GetPtr<MemSpace>();

        TData *w_APtr = this->m_w_A.template GetPtr<MemSpace>();
        TData *s_APtr = this->m_s_A.template GetPtr<MemSpace>();
        TData *r_APtr = this->m_r_A.template GetPtr<MemSpace>();
        TData *wkPtr  = this->m_wk.template GetPtr<MemSpace>();

        TData *p_APtr = this->m_p_A.template GetPtr<MemSpace>();
        TData *q_APtr = this->m_q_A.template GetPtr<MemSpace>();

        TData *vExchangePtr = m_vExchange.template GetPtr<MemSpace>();
        const TData *vExchangeHostPtr =
            m_vExchange.template GetConstPtr<NektarSpaces::HostSpace>();

        // Convergence parameters (host)
        size_t totalIterations = 0;
        TData rhsMagnitude;
        TData alpha;
        TData beta;
        TData rho;
        TData rho_new;
        TData mu;
        TData eps;

        // Copy RHS into initial residual
        this->m_r_A.template RegionToRegion<MemSpace>(in);

        // Assembly (communication)
        this->m_assmbScatrOp->apply(this->m_r_A, this->m_wk, true);

        dotKernel<ExecSpace, TData>(this->m_nloc, wkPtr, r_APtr,
                                    vExchangePtr + 2);

        m_vExchange.template DeviceToHost<MemSpace>();
        eps = vExchangeHostPtr[2];

        // Calculate rhs magnitude
        this->m_assmbScatrOp->apply(this->m_r_A, this->m_wk);

        dotKernel<ExecSpace, TData>(this->m_nloc, inPtr, wkPtr,
                                    vExchangePtr + 3);

        m_vExchange.template DeviceToHost<MemSpace>();
        rhsMagnitude = vExchangeHostPtr[3];
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Anytime there is a mix of internal kernel calls and
        // external operator calls. The memory region being
        // used must be marked as being valid which more
        // importantly invalidates the sibling memory region.
        this->m_r_A.template setValid<MemSpace>();

        // Apply preconditioner
        this->m_precon->apply(this->m_r_A, this->m_w_A);

        // Perform the method-specific matrix-vector multiply operation.
        this->m_LHS->apply(this->m_w_A, this->m_s_A);

        // Apply Robin BCs
        // this->m_robBndCondOp->apply(this->m_w_A, this->m_s_A);

        dotKernel<ExecSpace, TData>(this->m_nloc, r_APtr, w_APtr,
                                    vExchangePtr + 0);

        dotKernel<ExecSpace, TData>(this->m_nloc, s_APtr, w_APtr,
                                    vExchangePtr + 1);

        m_vExchange.template DeviceToHost<MemSpace>();
        rho = vExchangeHostPtr[0];
        mu  = vExchangeHostPtr[1];

        beta            = 0.0;
        alpha           = rho / mu;
        totalIterations = 1;

        while (true)
        {
            if (totalIterations > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Compute new search direction p_k
            daxpyKernel<ExecSpace, TData>(this->m_nloc, beta, p_APtr, w_APtr,
                                          p_APtr);

            // Compute new search direction q_k
            daxpyKernel<ExecSpace, TData>(this->m_nloc, beta, q_APtr, s_APtr,
                                          q_APtr);

            // Update solution x_{k+1}
            daxpyKernel<ExecSpace, TData>(this->m_nloc, alpha, p_APtr, outPtr,
                                          outPtr);

            // Update residual vector r_{k+1}
            daxpyKernel<ExecSpace, TData>(this->m_nloc, -alpha, q_APtr, r_APtr,
                                          r_APtr);

            // Anytime there is a mix of internal kernel calls and
            // external operator calls. The memory region being
            // used must be marked as being valid which more
            // importantly invalidates the sibling memory region.
            this->m_r_A.template setValid<MemSpace>();

            // Apply preconditioner
            this->m_precon->apply(this->m_r_A, this->m_w_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_LHS->apply(this->m_w_A, this->m_s_A);

            // Apply Robin BCs
            // this->m_robBndCondOp->apply(this->m_w_A, this->m_s_A);

            // <r_{k+1}, w_{k+1}>
            dotKernel<ExecSpace, TData>(this->m_nloc, r_APtr, w_APtr,
                                        vExchangePtr + 0);

            // <s_{k+1}, w_{k+1}>
            dotKernel<ExecSpace, TData>(this->m_nloc, s_APtr, w_APtr,
                                        vExchangePtr + 1);

            // <r_{k+1}, r_{k+1}>
            this->m_assmbScatrOp->apply(this->m_r_A, this->m_wk, true);

            dotKernel<ExecSpace, TData>(this->m_nloc, wkPtr, r_APtr,
                                        vExchangePtr + 2);

            m_vExchange.template DeviceToHost<MemSpace>();
            rho_new = vExchangeHostPtr[0];
            mu      = vExchangeHostPtr[1];
            eps     = vExchangeHostPtr[2];

            // std::cout << "Iteration " << totalIterations << " -- eps = " <<
            // eps
            //           << "\n";

            totalIterations++;

            // Test if norm is within tolerance
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                break;
            }

            // Compute search direction and solution coefficients
            beta  = rho_new / rho;
            alpha = rho_new / (mu - rho_new * beta / alpha);
            rho   = rho_new;
        }
    }

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorConjGradImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    MemoryRegion<TData> m_vExchange;
};

} // namespace Nektar::Operators::detail
