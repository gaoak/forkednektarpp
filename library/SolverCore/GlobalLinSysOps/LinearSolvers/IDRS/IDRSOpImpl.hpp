///////////////////////////////////////////////////////////////////////////////
//
// File: IDRSOpImpl.hpp
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

#include "SolverCore/GlobalLinSysOps/LinearSolvers/IDRS/IDRSOp.hpp"
#include "SolverCore/GlobalLinSysOps/MultiFieldHelper/MultiFieldHelper.hpp"

#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

#include <iomanip>
#include <random>

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class IDRSOpImpl : public IDRSOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IDRSOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
               const std::vector<std::string> &components)
        : IDRSOp<TData>(expansionList, components),
          m_P("IDRSOp P",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components),
          m_U("IDRSOp U",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components),
          m_G("IDRSOp G",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components),
          m_v(LibUtilities::Field<TData, FieldState::Coeff>(
              "IDRSOp v",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_w(LibUtilities::Field<TData, FieldState::Coeff>(
              "IDRSOp t",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r(LibUtilities::Field<TData, FieldState::Coeff>(
              "IDRSOp r",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1))
    {
        this->template SetLinearSolver<ExecSpace>();

        auto session = this->m_expansionList->GetSession();

        // Set parameters.
        this->m_leftPreconditioner =
            session->DefinesParameter("LinSysLeftPrecon")
                ? session->GetParameter("LinSysLeftPrecon")
                : false;
        this->m_rightPreconditioner =
            session->DefinesParameter("LinSysRightPrecon")
                ? session->GetParameter("LinSysRightPrecon")
                : true;
        m_stage     = session->DefinesParameter("IDRstage")
                          ? session->GetParameter("IDRstage")
                          : 4;
        m_vExchange = LibUtilities::MemoryRegion<TData>(std::max(m_stage, 2u),
                                                        eHostPinned);
        m_coeffs    = LibUtilities::MemoryRegion<TData>(std::max(m_stage, 1u),
                                                        eHostPinned);

        // Set-up storage.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        m_U.ResizeNumField(m_stage);
        m_G.ResizeNumField(m_stage);
        m_P.ResizeNumField(m_stage);
        for (unsigned int stage = 0; stage < m_stage; stage++)
        {
            // Initialzie m_P with normalized random number.
            for (unsigned blk = 0; blk < m_P[stage].GetBlocks().size(); ++blk)
            {
                auto &block = m_P[stage].GetBlocks()[blk];
                auto ptr =
                    block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                std::generate(ptr,
                              ptr + block.GetNumComponents() * block.CompSize(),
                              [&]() { return dis(gen); });
            }
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<IDRSOpImpl<ExecSpace, TData>>(expansionList,
                                                              components);
    }

protected:
    LibUtilities::MultiField<TData, FieldState::Coeff> m_P;
    LibUtilities::MultiField<TData, FieldState::Coeff> m_U;
    LibUtilities::MultiField<TData, FieldState::Coeff> m_G;
    LibUtilities::Field<TData, FieldState::Coeff> m_v;
    LibUtilities::Field<TData, FieldState::Coeff> m_w;
    LibUtilities::Field<TData, FieldState::Coeff> m_r;
    LibUtilities::MemoryRegion<TData> m_vExchange;
    LibUtilities::MemoryRegion<TData> m_coeffs; ///< gamma, for the updates.

    unsigned int m_stage = 0;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // Reshape m_P if required.
        for (unsigned int stage = 0; stage < m_stage; stage++)
        {
            for (unsigned blk = 0; blk < in.GetBlocks().size(); ++blk)
            {
                const unsigned int streamID = blk + 1;

                auto &inblk = in.GetBlocks()[blk];
                auto &pblk  = m_P[stage].GetBlocks()[blk];

                if (inblk.GetInterleaveWidth() != pblk.GetInterleaveWidth())
                {
                    auto ptr =
                        pblk.template GetPtr<MemSpace, ReadWrite>(streamID);
                    auto numComp = pblk.GetNumComponents();

                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inblk.GetInterleaveWidth(), pblk.GetInterleaveWidth(),
                        pblk.GetNumElementsWithPadding() * numComp,
                        pblk.GetNumData(), ptr, streamID);

                    pblk.template SetInterleaveWidth<TData>(
                        inblk.GetInterleaveWidth());
                }
            }
        }

        // Implement IDR(s) iterative method as described in:
        //
        // Reference:
        // Van Gijzen, Martin B., and Peter Sonneveld. "Algorithm 913: An
        // elegant IDR (s) variant that efficiently exploits biorthogonality
        // properties." ACM Transactions on Mathematical Software (TOMS) 38, no.
        // 1 (2011): 1-19.

        // Convergence parameters.
        this->m_niter = 0;
        TData rhsMagnitude, eps;
        TData omega, omega0, omega1, rho, alpha, beta, kappa = 0.7;
        std::vector<TData> Phi(m_stage), gamma(m_stage);
        std::vector<std::vector<TData>> Mu(m_stage);

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);
        out.SetInterleaveWidth(in);

        // Reset device memory.
        auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrOp->Apply(m_r);
        Math::ddot<ExecSpace>(in, m_r, exchange + 1);

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrZeroDirOp->Apply(m_r);

        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_r);
        }

        Math::ddot<ExecSpace>(in, m_r, exchange + 0);

        // Communication.
        this->m_rowComm->template AllReduce<MemSpace>(
            m_vExchange, Nektar::LibUtilities::ReduceSum);

        // Device-to-host copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        rhsMagnitude = this->GetRhsMagnitude(exchangeHost[1]);
        eps          = exchangeHost[0];

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        omega = 1.0;
        for (unsigned int k = 0; k < m_stage; k++)
        {
            Mu[k]    = std::vector<TData>(m_stage);
            Mu[k][k] = 1.0;
            m_G[k].template Initialize<MemSpace>(0.0);
            m_G[k].SetInterleaveWidth(in);
            m_U[k].template Initialize<MemSpace>(0.0);
            m_U[k].SetInterleaveWidth(in);
        }
        while (true)
        {
            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // Compute Phi.
            MultiDot<ExecSpace>(m_P, 0, m_stage, m_r, exchange);

            // Communication.
            this->m_rowComm->template AllReduce<MemSpace>(
                m_vExchange, Nektar::LibUtilities::ReduceSum);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            std::copy_n(exchangeHost, m_stage, Phi.begin());

            // Inner iteration.
            for (unsigned int k = 0; k < m_stage; k++)
            {
                if (this->m_niter > this->m_maxIter)
                {
                    std::stringstream msg;
                    msg << "Exceeded max iterations: " << this->m_niter
                        << ". Increase NekLinSysMaxIterations in the session "
                           "PARAMETERS section to allow more iterations.";
                    NEKERROR(ErrorUtil::efatal, msg.str());

                    return;
                }

                // Directly solve the small linear system.
                if (m_stage - k == 1)
                {
                    gamma[k] = Phi[k] / Mu[k][k];
                }
                else
                {
                    std::vector<std::vector<TData>> Mu_k(m_stage - k);
                    std::vector<TData> Phi_k(m_stage - k);
                    for (unsigned int i = 0; i < m_stage - k; i++)
                    {
                        Mu_k[i] = std::vector<TData>(m_stage - k);
                        std::copy(Mu[i + k].begin() + k, Mu[i + k].end(),
                                  Mu_k[i].data());
                    }
                    std::copy(Phi.begin() + k, Phi.end(), Phi_k.begin());
                    this->DirectSolve(Mu_k, Phi_k);
                    std::copy(Phi_k.begin(), Phi_k.end(), gamma.begin() + k);
                }

                // gamma[k + 1:] for the updates of m_v and m_U[k].
                const TData *coeffs = nullptr;
                if (this->m_niter > 0 && k + 1 < m_stage)
                {
                    auto coeffsHost =
                        m_coeffs.template GetPtr<NektarSpaces::HostSpace,
                                                 WriteOnly>();
                    std::copy(gamma.begin() + k + 1, gamma.end(), coeffsHost);
                    coeffs = m_coeffs.template GetPtr<MemSpace, ReadOnly>();
                }

                // Compute m_v.
                if (this->m_niter == 0)
                {
                    m_v.template Copy<MemSpace>(m_r);
                }
                else
                {
                    Math::daxpy<ExecSpace>(-gamma[k], m_G[k], m_r, m_v);
                    if (k + 1 < m_stage)
                    {
                        MultiAxpy<ExecSpace>(-1.0, m_G, k + 1, m_stage, coeffs,
                                             1.0, m_v);
                    }
                }

                // Apply preconditioner.
                if (this->m_rightPreconditioner)
                {
                    this->m_precon->Apply(m_v, m_v);
                }

                // Compute new U.
                if (this->m_niter == 0)
                {
                    Math::mul<ExecSpace>(omega, m_v, m_U[k]);
                }
                else
                {
                    Math::daxpby<ExecSpace>(omega, m_v, gamma[k], m_U[k],
                                            m_U[k]);
                    if (k + 1 < m_stage)
                    {
                        MultiAxpy<ExecSpace>(1.0, m_U, k + 1, m_stage, coeffs,
                                             1.0, m_U[k]);
                    }
                }

                // Perform the method-specific matrix-vector Math::multiply
                // operation.
                this->m_lhs->Apply(m_U[k], m_G[k]);
                this->m_robBndCondOp->Apply(m_U[k], m_G[k]);
                this->m_assmbScatrZeroDirOp->Apply(m_G[k]);
                if (this->m_leftPreconditioner)
                {
                    this->m_precon->Apply(m_G[k], m_G[k]);
                }

                // Bi-Orthogonalize the basis vectors:
                for (unsigned int i = 0; i < k; i++)
                {
                    alpha = this->m_math.ddot(m_P[i], m_G[k]);
                    this->m_rowComm->AllReduce(alpha,
                                               Nektar::LibUtilities::ReduceSum);
                    alpha /= Mu[i][i];
                    Math::daxpy<ExecSpace>(-alpha, m_G[i], m_G[k], m_G[k]);
                    Math::daxpy<ExecSpace>(-alpha, m_U[i], m_U[k], m_U[k]);
                }

                // Reset device memory.
                m_vExchange.template GetPtr<MemSpace, WriteOnly>();

                // Update Mu.
                MultiDot<ExecSpace>(m_P, k, m_stage, m_G[k], exchange);

                // Communication.
                this->m_rowComm->template AllReduce<MemSpace>(
                    m_vExchange, Nektar::LibUtilities::ReduceSum);

                // Device-to-host copy.
                exchangeHost =
                    m_vExchange
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

                for (unsigned int i = k; i < m_stage; i++)
                {
                    Mu[i][k] = exchangeHost[i - k];
                }

                if (Mu[k][k] == 0.0)
                {
                    std::stringstream msg;
                    msg << "Convergence breakdown: ";
                    NEKERROR(ErrorUtil::efatal, msg.str());

                    return;
                }

                // Make m_r orthogonal to m_G.
                beta = Phi[k] / Mu[k][k];
                Math::daxpy<ExecSpace>(-beta, m_G[k], m_r, m_r);
                Math::daxpy<ExecSpace>(beta, m_U[k], out, out);

                eps = this->m_math.ddot(m_r, m_r);
                this->m_rowComm->AllReduce(eps,
                                           Nektar::LibUtilities::ReduceSum);

                ++this->m_niter;

                // Test if norm is within tolerance.
                if (eps < this->m_tol * this->m_tol * rhsMagnitude)
                {
                    this->PrintVerboseOutput(this->name, "error",
                                             std::sqrt(eps / rhsMagnitude),
                                             rhsMagnitude);

                    return;
                }

                // Update Phi.
                if (k < m_stage - 1)
                {
                    for (unsigned int i = 0; i < k + 1; i++)
                    {
                        Phi[i] = 0.0;
                    }
                    for (unsigned int i = k + 1; i < m_stage; i++)
                    {
                        Phi[i] -= beta * Mu[i][k];
                    }
                }
            }

            ++this->m_niter;

            // Test if norm is within tolerance.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                this->PrintVerboseOutput(this->name, "error",
                                         std::sqrt(eps / rhsMagnitude),
                                         rhsMagnitude);

                return;
            }

            if (this->m_niter > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << this->m_niter
                    << ". Increase NekLinSysMaxIterations in the session "
                       "PARAMETERS section to allow more iterations.";
                NEKERROR(ErrorUtil::efatal, msg.str());

                return;
            }

            // Perform the method-specific matrix-vector Math::multiply
            // operation.
            auto &tmp = (this->m_rightPreconditioner) ? m_v : m_r;
            if (this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_r, tmp);
            }
            this->m_lhs->Apply(tmp, m_w);
            this->m_robBndCondOp->Apply(tmp, m_w);
            this->m_assmbScatrZeroDirOp->Apply(m_w);
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_w, m_w);
            }

            // Reset device memory.
            m_vExchange.template GetPtr<MemSpace, WriteOnly>();

            // Update coefficients.
            Math::ddot<ExecSpace>(m_w, m_r, exchange + 0);
            Math::ddot<ExecSpace>(m_w, m_w, exchange + 1);

            // Communication.
            this->m_rowComm->template AllReduce<MemSpace>(
                m_vExchange, LibUtilities::ReduceSum);

            // Device-to-host copy.
            exchangeHost =
                m_vExchange
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            omega0 = exchangeHost[0];
            omega1 = exchangeHost[1];
            omega  = omega0 / omega1;
            rho    = std::abs(omega0 / (std::sqrt(omega1) * std::sqrt(eps)));
            if (rho < kappa)
            {
                omega *= kappa / rho;
            }

            // Update solution.
            Math::daxpy<ExecSpace>(omega, tmp, out, out);
            Math::daxpy<ExecSpace>(-omega, m_w, m_r, m_r);

            // Update residual norm.
            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);
        }
    }
};
} // namespace Nektar::SolverCore::detail
