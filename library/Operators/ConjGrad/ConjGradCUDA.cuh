///////////////////////////////////////////////////////////////////////////////
//
// File: ConjGradCUDA.cuh
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

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <memory>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ContField.h>

#include "Operators/CUDAMathKernels.cuh"
#include "Operators/MemoryRegionCUDA.hpp"
#include "Operators/OperatorAssmbScatr.hpp"
#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorHelper.hpp"
#include "Operators/OperatorRobBndCond.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorConjGradImpl<TData, ImplCUDA> : public OperatorConjGrad<TData>
{
public:
    OperatorConjGradImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorConjGrad<TData>(expansionList),
          m_w_A(Field<TData, FieldState::Coeff>::template create<
                MemoryRegionCUDA>(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_s_A(Field<TData, FieldState::Coeff>::template create<
                MemoryRegionCUDA>(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_r_A(Field<TData, FieldState::Coeff>::template create<
                MemoryRegionCUDA>(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_wk(Field<TData, FieldState::Coeff>::template create<
               MemoryRegionCUDA>(
              GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        contfield->GetSession()->LoadParameter("NekLinSysMaxIterations",
                                               m_maxIter, 5000);
        contfield->GetSession()->LoadParameter("IterativeSolverTolerance",
                                               m_tol, 1.0E-09);
        m_nloc       = contfield->GetLocalToGlobalMap()->GetNumLocalCoeffs();
        m_assmbScatr = AssmbScatr<TData>::create(this->m_expansionList, "CUDA");
        // m_robBndCond = RobBndCond<TData>::create(this->m_expansionList,
        // "CUDA");
        cudaMalloc((void **)&m_p_A, sizeof(TData) * m_nloc);
        cudaMalloc((void **)&m_q_A, sizeof(TData) * m_nloc);
        cudaMalloc((void **)&m_vExchange, sizeof(TData) * 4);
        cudaMalloc((void **)&m_buffer, sizeof(TData) * m_gridSize);
    }

    ~OperatorConjGradImpl(void)
    {
        cudaFree(m_p_A);
        cudaFree(m_q_A);
        cudaFree(m_vExchange);
        cudaFree(m_buffer);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Store pointers to temporary fields (device)
        auto *p_in  = in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *p_out = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *p_w_A = m_w_A.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *p_s_A = m_s_A.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *p_r_A = m_r_A.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *p_wk  = m_wk.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto p_p_A  = m_p_A;
        auto p_q_A  = m_q_A;

        // Set the fields to zero (device)
        cudaMemset(p_out, 0, sizeof(TData) * m_nloc);
        cudaMemset(p_w_A, 0, sizeof(TData) * m_nloc);
        cudaMemset(p_s_A, 0, sizeof(TData) * m_nloc);
        cudaMemset(p_wk, 0, sizeof(TData) * m_nloc);
        cudaMemset(p_p_A, 0, sizeof(TData) * m_nloc);
        cudaMemset(p_q_A, 0, sizeof(TData) * m_nloc);
        cudaMemset(m_buffer, 0, sizeof(TData) * m_gridSize);

        // Convergence parameters (host)
        size_t totalIterations = 0;
        TData rhsMagnitude;
        TData alpha;
        TData beta;
        TData rho;
        TData rho_new;
        TData mu;
        TData eps;

        // Temporary (device)
        cudaMemset(m_vExchange, 0, sizeof(TData) * 4);

        // Copy RHS into initial residual
        cudaMemcpy(p_r_A, p_in, sizeof(TData) * m_nloc,
                   cudaMemcpyDeviceToDevice);

        // Assembly (communication)
        m_assmbScatr->apply(m_r_A, m_wk, true);
        dotKernel<m_blockSize>
            <<<m_gridSize, m_blockSize>>>(m_nloc, p_wk, p_r_A, m_buffer);
        reduceKernel<m_gridSize>
            <<<1, m_gridSize>>>(m_gridSize, m_buffer, m_vExchange + 2);
        cudaMemcpy(&eps, m_vExchange + 2, sizeof(TData),
                   cudaMemcpyDeviceToHost);

        // Calculate rhs magnitude
        m_assmbScatr->apply(m_r_A, m_wk);
        dotKernel<m_blockSize>
            <<<m_gridSize, m_blockSize>>>(m_nloc, p_in, p_wk, m_buffer);
        reduceKernel<m_gridSize>
            <<<1, m_gridSize>>>(m_gridSize, m_buffer, m_vExchange + 3);
        cudaMemcpy(&rhsMagnitude, m_vExchange + 3, sizeof(TData),
                   cudaMemcpyDeviceToHost);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // If input residual is less than tolerance skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Apply preconditioner
        this->m_precon->apply(m_r_A, m_w_A);

        // Perform the method-specific matrix-vector multiply operation.
        this->m_LHS->apply(m_w_A, m_s_A);

        // Apply Robin BCs
        // m_robBndCond->apply(m_w_A, m_s_A);

        cudaMemset(m_vExchange, 0, sizeof(TData) * 3);

        dotKernel<m_blockSize>
            <<<m_gridSize, m_blockSize>>>(m_nloc, p_r_A, p_w_A, m_buffer);
        reduceKernel<m_gridSize>
            <<<1, m_gridSize>>>(m_gridSize, m_buffer, m_vExchange + 0);

        dotKernel<m_blockSize>
            <<<m_gridSize, m_blockSize>>>(m_nloc, p_s_A, p_w_A, m_buffer);
        reduceKernel<m_gridSize>
            <<<1, m_gridSize>>>(m_gridSize, m_buffer, m_vExchange + 1);

        cudaMemcpy(&rho, m_vExchange + 0, sizeof(TData),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&mu, m_vExchange + 1, sizeof(TData), cudaMemcpyDeviceToHost);
        beta            = 0.0;
        alpha           = rho / mu;
        totalIterations = 1;
        while (true)
        {
            if (totalIterations > m_maxIter)
            {
                std::cout << "Exceeded max iterations\n";
                return;
            }

            // Compute new search direction p_k
            daxpyKernel<<<m_gridSize, m_blockSize>>>(m_nloc, beta, p_p_A, p_w_A,
                                                     p_p_A);

            // Compute new search direction q_k
            daxpyKernel<<<m_gridSize, m_blockSize>>>(m_nloc, beta, p_q_A, p_s_A,
                                                     p_q_A);

            // Update solution x_{k+1}
            daxpyKernel<<<m_gridSize, m_blockSize>>>(m_nloc, alpha, p_p_A,
                                                     p_out, p_out);

            // Update residual vector r_{k+1}
            daxpyKernel<<<m_gridSize, m_blockSize>>>(m_nloc, -alpha, p_q_A,
                                                     p_r_A, p_r_A);

            // Apply preconditioner
            this->m_precon->apply(m_r_A, m_w_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_LHS->apply(m_w_A, m_s_A);

            // Apply Robin BCs
            // m_robBndCond->apply(m_w_A, m_s_A);

            cudaMemset(m_vExchange, 0, sizeof(TData) * 3);

            // <r_{k+1}, w_{k+1}>
            dotKernel<m_blockSize>
                <<<m_gridSize, m_blockSize>>>(m_nloc, p_r_A, p_w_A, m_buffer);
            reduceKernel<m_gridSize>
                <<<1, m_gridSize>>>(m_gridSize, m_buffer, m_vExchange + 0);

            // <s_{k+1}, w_{k+1}>
            dotKernel<m_blockSize>
                <<<m_gridSize, m_blockSize>>>(m_nloc, p_s_A, p_w_A, m_buffer);
            reduceKernel<m_gridSize>
                <<<1, m_gridSize>>>(m_gridSize, m_buffer, m_vExchange + 1);

            // <r_{k+1}, r_{k+1}>
            m_assmbScatr->apply(m_r_A, m_wk, true);
            dotKernel<m_blockSize>
                <<<m_gridSize, m_blockSize>>>(m_nloc, p_wk, p_r_A, m_buffer);
            reduceKernel<m_gridSize>
                <<<1, m_gridSize>>>(m_gridSize, m_buffer, m_vExchange + 2);

            cudaMemcpy(&rho_new, m_vExchange + 0, sizeof(TData),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(&mu, m_vExchange + 1, sizeof(TData),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(&eps, m_vExchange + 2, sizeof(TData),
                       cudaMemcpyDeviceToHost);

            std::cout << "Iteration " << totalIterations << " -- eps = " << eps
                      << "\n";

            totalIterations++;

            // Test if norm is within tolerance
            if (eps < m_tol * m_tol * rhsMagnitude)
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
        return std::make_unique<OperatorConjGradImpl<TData, ImplCUDA>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatr;
    std::shared_ptr<OperatorRobBndCond<TData>> m_robBndCond;
    Field<TData, FieldState::Coeff> m_w_A;
    Field<TData, FieldState::Coeff> m_s_A;
    Field<TData, FieldState::Coeff> m_r_A;
    Field<TData, FieldState::Coeff> m_wk;
    TData *m_buffer;
    TData *m_q_A;
    TData *m_p_A;
    TData *m_vExchange;
    TData m_tol;
    size_t m_nloc;
    size_t m_maxIter;
    static constexpr size_t m_gridSize  = 1024;
    static constexpr size_t m_blockSize = 256;
};

} // namespace Nektar::Operators::detail
