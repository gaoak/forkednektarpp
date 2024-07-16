///////////////////////////////////////////////////////////////////////////////
//
// File: ConjGradImpl.hpp
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

#include "Operators/MathKernels/MathKernels.hpp"
#include "Operators/OperatorConjGrad.hpp"

#include "Operators/OperatorAssmbScatr.hpp"
#include "Operators/OperatorHelper.hpp"
#include "Operators/OperatorRobBndCond.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ContField.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <memory>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

// Generic implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorConjGradImpl : public OperatorConjGrad<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorConjGradImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorConjGrad<TData>(expansionList),
          m_w_A(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad w_A",
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment)),
          m_s_A(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad s_A",
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment)),
          m_r_A(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad r_A",
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment)),
          m_wk(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad wk",
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment)),
          m_q_A(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad wk",
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment)),
          m_p_A(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad wk",
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment))
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        contfield->GetSession()->LoadParameter("NekLinSysMaxIterations",
                                               m_maxIter, 5000);
        contfield->GetSession()->LoadParameter("IterativeSolverTolerance",
                                               m_tol, 1.0E-09);

        m_nloc = contfield->GetLocalToGlobalMap()->GetNumLocalCoeffs();

        m_assmbScatrOp =
            AssmbScatr<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_robBndCondOp =
            RobBndCond<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);

        if constexpr (std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
                      std::is_same<ExecSpace, NektarSpaces::AVX>::value)
        {
            m_rowComm = contfield->GetSession()->GetComm()->GetRowComm();

            m_vExchangeArray = Array<OneD, TData>(4, 0.0);
        }
        else if constexpr (std::is_same<ExecSpace, NektarSpaces::CUDA>::value)
        {
            m_vExchange = MemoryRegion<TData>::template create<MemSpace>(4);
        }
        else if constexpr (std::is_same<ExecSpace,
                                        Kokkos::DefaultExecutionSpace>::value)
        {
            // For Kokkos, all reduction values are on the host side.
            m_vExchange =
                MemoryRegion<TData>::template create<NektarSpaces::HostSpace>(
                    4);
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Set the fields to zero
        out.initialize(0);

        m_w_A.initialize(0);
        m_s_A.initialize(0);
        m_wk.initialize(0);

        m_p_A.initialize(0);
        m_q_A.initialize(0);

        TData *vExchangePtr;
        TData const *vExchangeHostPtr;

        // For Serial, all reduction values are on the host side.
        if constexpr (std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
                      std::is_same<ExecSpace, NektarSpaces::AVX>::value)
        {
            vExchangePtr     = m_vExchangeArray.get();
            vExchangeHostPtr = m_vExchangeArray.get();
        }
        // For Kokkos, all reduction values are on the host side.
        else if constexpr (std::is_same<ExecSpace,
                                        Kokkos::DefaultExecutionSpace>::value &&
                           std::is_same<Implementation,
                                        Operators::StdMat>::value)
        {
            m_vExchange.initialize(0);

            vExchangePtr =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
            vExchangeHostPtr =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        }
        // For all others, the reduction values are on the MemSpace side.
        else
        {
            m_vExchange.initialize(0);

            vExchangePtr = m_vExchange.template GetPtr<MemSpace, ReadWrite>();
            vExchangeHostPtr =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        }

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
        m_r_A.template copyField<MemSpace>(in);

        // Assembly (communication)
        m_assmbScatrOp->apply(m_r_A, m_wk, true);

        ddot<ExecSpace, TData>(m_wk, m_r_A, vExchangePtr + 2);

        // Calculate rhs magnitude
        m_assmbScatrOp->apply(m_r_A, m_wk);

        ddot<ExecSpace, TData>(in, m_wk, vExchangePtr + 3);

        reduceMemcpy();

        eps          = vExchangeHostPtr[2];
        rhsMagnitude = vExchangeHostPtr[3];
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // If the input residual is less than tolerance then skip solve.
        if (eps < m_tol * m_tol * rhsMagnitude)
        {
            return;
        }

        // Apply preconditioner
        this->m_precon->apply(m_r_A, m_w_A);

        // Perform the method-specific matrix-vector multiply operation.
        this->m_LHS->apply(m_w_A, m_s_A);

        // Apply Robin BCs - AVX???
        if constexpr (std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
                      std::is_same<ExecSpace, NektarSpaces::AVX>::value)
        {
            m_robBndCondOp->apply(m_w_A, m_s_A);
        }

        ddot<ExecSpace, TData>(m_r_A, m_w_A, vExchangePtr + 0);

        ddot<ExecSpace, TData>(m_s_A, m_w_A, vExchangePtr + 1);

        reduceMemcpy();

        rho = vExchangeHostPtr[0];
        mu  = vExchangeHostPtr[1];

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

            // Compute new search direction p_k
            daxpy<ExecSpace, TData>(beta, m_p_A, m_w_A, m_p_A);

            // Compute new search direction q_k
            daxpy<ExecSpace, TData>(beta, m_q_A, m_s_A, m_q_A);

            // Update solution x_{k+1}
            daxpy<ExecSpace, TData>(alpha, m_p_A, out, out);

            // Update residual vector r_{k+1}
            daxpy<ExecSpace, TData>(-alpha, m_q_A, m_r_A, m_r_A);

            // Apply preconditioner
            this->m_precon->apply(m_r_A, m_w_A);

            // Perform the method-specific matrix-vector multiply
            // operation.
            this->m_LHS->apply(m_w_A, m_s_A);

            // Apply Robin BCs - AVX???
            if constexpr (std::is_same<ExecSpace,
                                       NektarSpaces::Serial>::value ||
                          std::is_same<ExecSpace, NektarSpaces::AVX>::value)
            {
                m_robBndCondOp->apply(m_w_A, m_s_A);
            }

            // <r_{k+1}, w_{k+1}>
            ddot<ExecSpace, TData>(m_r_A, m_w_A, vExchangePtr + 0);

            // <s_{k+1}, w_{k+1}>
            ddot<ExecSpace, TData>(m_s_A, m_w_A, vExchangePtr + 1);

            // <r_{k+1}, r_{k+1}>
            m_assmbScatrOp->apply(m_r_A, m_wk, true);

            ddot<ExecSpace, TData>(m_wk, m_r_A, vExchangePtr + 2);

            reduceMemcpy();

            rho_new = vExchangeHostPtr[0];
            mu      = vExchangeHostPtr[1];
            eps     = vExchangeHostPtr[2];

            // std::cout << "Iteration " << totalIterations << "
            // -- eps = " << eps
            //           << "\n";

            ++totalIterations;

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
        return std::make_unique<
            OperatorConjGradImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    // Helper method to reduce the host side and memcpy device to host.
    void reduceMemcpy()
    {
        if constexpr (std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
                      std::is_same<ExecSpace, NektarSpaces::AVX>::value)
        {
            m_rowComm->AllReduce(m_vExchangeArray,
                                 Nektar::LibUtilities::ReduceSum);
        }

        // For CUDA reduction values are on the device side.
        if constexpr (std::is_same<ExecSpace, NektarSpaces::CUDA>::value)
        {
            m_vExchange.template DeviceToHostCopy<MemSpace>();
        }
    }

    LibUtilities::CommSharedPtr m_rowComm = nullptr;

    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatrOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_robBndCondOp;

    Field<TData, FieldState::Coeff> m_w_A;
    Field<TData, FieldState::Coeff> m_s_A;
    Field<TData, FieldState::Coeff> m_r_A;
    Field<TData, FieldState::Coeff> m_wk;
    Field<TData, FieldState::Coeff> m_q_A;
    Field<TData, FieldState::Coeff> m_p_A;

    MemoryRegion<TData> m_vExchange;

    Array<OneD, TData> m_vExchangeArray;

    TData m_tol;
    size_t m_nloc;
    size_t m_maxIter;
};

} // namespace Nektar::Operators::detail
