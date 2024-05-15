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
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)
#if defined(NEKTAR_ENABLE_CUDA)
              || (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
                  std::is_same<Implementation, Operators::SumFac>::value)
#endif
              >::type>
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

        m_buffer = MemoryRegion<TData>::template create<MemSpace>(m_gridSize);
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

        m_buffer.initialize(0);

        const TData *p_in = in.template GetConstPtr<MemSpace>();
        TData *p_out      = out.template GetPtr<MemSpace>();

        TData *p_w_A = this->m_w_A.template GetPtr<MemSpace>();
        TData *p_s_A = this->m_s_A.template GetPtr<MemSpace>();
        TData *p_r_A = this->m_r_A.template GetPtr<MemSpace>();
        TData *p_wk  = this->m_wk.template GetPtr<MemSpace>();

        TData *p_p_A = this->m_p_A.template GetPtr<MemSpace>();
        TData *p_q_A = this->m_q_A.template GetPtr<MemSpace>();

        TData *p_buffer = m_buffer.template GetPtr<MemSpace>();

        TData *p_vExchange = m_vExchange.template GetPtr<MemSpace>();
        const TData *p_vExchangeHost =
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

        dotKernel<ExecSpace, TData, m_blockSize>(
            m_gridSize, m_blockSize, this->m_nloc, p_wk, p_r_A, p_buffer);
        reduceKernel<ExecSpace, TData, m_gridSize>(1, m_gridSize, m_gridSize,
                                                   p_buffer, p_vExchange + 2);

        m_vExchange.template DeviceToHost<MemSpace>();
        eps = p_vExchangeHost[2];

        // Calculate rhs magnitude
        this->m_assmbScatrOp->apply(this->m_r_A, this->m_wk);

        dotKernel<ExecSpace, TData, m_blockSize>(
            m_gridSize, m_blockSize, this->m_nloc, p_in, p_wk, p_buffer);
        reduceKernel<ExecSpace, TData, m_gridSize>(1, m_gridSize, m_gridSize,
                                                   p_buffer, p_vExchange + 3);

        m_vExchange.template DeviceToHost<MemSpace>();
        rhsMagnitude = p_vExchangeHost[3];
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Apply preconditioner
        this->m_precon->apply(this->m_r_A, this->m_w_A);

        // Perform the method-specific matrix-vector multiply operation.
        this->m_LHS->apply(this->m_w_A, this->m_s_A);

        // Apply Robin BCs
        // this->m_robBndCondOp->apply(this->m_w_A, this->m_s_A);

        dotKernel<ExecSpace, TData, m_blockSize>(
            m_gridSize, m_blockSize, this->m_nloc, p_r_A, p_w_A, p_buffer);
        reduceKernel<ExecSpace, TData, m_gridSize>(1, m_gridSize, m_gridSize,
                                                   p_buffer, p_vExchange + 0);

        dotKernel<ExecSpace, TData, m_blockSize>(
            m_gridSize, m_blockSize, this->m_nloc, p_s_A, p_w_A, p_buffer);
        reduceKernel<ExecSpace, TData, m_gridSize>(1, m_gridSize, m_gridSize,
                                                   p_buffer, p_vExchange + 1);

        m_vExchange.template DeviceToHost<MemSpace>();
        rho = p_vExchangeHost[0];
        mu  = p_vExchangeHost[1];

        beta            = 0.0;
        alpha           = rho / mu;
        totalIterations = 1;

        while (true)
        {
            if (totalIterations > this->m_maxIter)
            {
                std::cout << "Exceeded max iterations\n";
                return;
            }

            // Compute new search direction p_k
            daxpyKernel<ExecSpace, TData>(m_gridSize, m_blockSize, this->m_nloc,
                                          beta, p_p_A, p_w_A, p_p_A);

            // Compute new search direction q_k
            daxpyKernel<ExecSpace, TData>(m_gridSize, m_blockSize, this->m_nloc,
                                          beta, p_q_A, p_s_A, p_q_A);

            // Update solution x_{k+1}
            daxpyKernel<ExecSpace, TData>(m_gridSize, m_blockSize, this->m_nloc,
                                          alpha, p_p_A, p_out, p_out);

            // Update residual vector r_{k+1}
            daxpyKernel<ExecSpace, TData>(m_gridSize, m_blockSize, this->m_nloc,
                                          -alpha, p_q_A, p_r_A, p_r_A);

            // Apply preconditioner
            this->m_precon->apply(this->m_r_A, this->m_w_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_LHS->apply(this->m_w_A, this->m_s_A);

            // Apply Robin BCs
            // this->m_robBndCondOp->apply(this->m_w_A, this->m_s_A);

            // <r_{k+1}, w_{k+1}>
            dotKernel<ExecSpace, TData, m_blockSize>(
                m_gridSize, m_blockSize, this->m_nloc, p_r_A, p_w_A, p_buffer);
            reduceKernel<ExecSpace, TData, m_gridSize>(
                1, m_gridSize, m_gridSize, p_buffer, p_vExchange + 0);

            // <s_{k+1}, w_{k+1}>
            dotKernel<ExecSpace, TData, m_blockSize>(
                m_gridSize, m_blockSize, this->m_nloc, p_s_A, p_w_A, p_buffer);
            reduceKernel<ExecSpace, TData, m_gridSize>(
                1, m_gridSize, m_gridSize, p_buffer, p_vExchange + 1);

            // <r_{k+1}, r_{k+1}>
            this->m_assmbScatrOp->apply(this->m_r_A, this->m_wk, true);

            dotKernel<ExecSpace, TData, m_blockSize>(
                m_gridSize, m_blockSize, this->m_nloc, p_wk, p_r_A, p_buffer);
            reduceKernel<ExecSpace, TData, m_gridSize>(
                1, m_gridSize, m_gridSize, p_buffer, p_vExchange + 2);

            m_vExchange.template DeviceToHost<MemSpace>();
            rho_new = p_vExchangeHost[0];
            mu      = p_vExchangeHost[1];
            eps     = p_vExchangeHost[2];

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
    MemoryRegion<TData> m_buffer;

    static constexpr size_t m_gridSize  = 1024;
    static constexpr size_t m_blockSize = 256;
};

} // namespace Nektar::Operators::detail
