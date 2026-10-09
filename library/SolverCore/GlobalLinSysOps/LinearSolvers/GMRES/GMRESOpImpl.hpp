///////////////////////////////////////////////////////////////////////////////
//
// File: GMRESOpImpl.hpp
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

#include "SolverCore/GlobalLinSysOps/LinearSolvers/GMRES/GMRESOp.hpp"
#include "SolverCore/GlobalLinSysOps/MultiFieldHelper/MultiFieldHelper.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class GMRESOpImpl : public GMRESOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    GMRESOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : GMRESOp<TData>(expansionList, components),
          m_w(LibUtilities::Field<TData, FieldState::Coeff>(
              "GMRES w",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r0(LibUtilities::Field<TData, FieldState::Coeff>(
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_V("GMRES V",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components),
          m_Z("GMRES Z",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components),
          m_delta("GMRES delta",
                  MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                      expansionList),
                  components)
    {
        this->template SetLinearSolver<ExecSpace>();
        this->template SetMask<ExecSpace>();

        auto session = this->m_expansionList->GetSession();

        this->m_leftPreconditioner =
            session->DefinesParameter("LinSysLeftPrecon")
                ? session->GetParameter("LinSysLeftPrecon")
                : false;
        this->m_rightPreconditioner =
            session->DefinesParameter("LinSysRightPrecon")
                ? session->GetParameter("LinSysRightPrecon")
                : true;
        this->m_LinSysMaxStorage =
            session->DefinesParameter("LinSysMaxStorage")
                ? session->GetParameter("LinSysMaxStorage")
                : 100;
        // LGMRES parameter
        // Reference:
        // Baker, Allison H., Elizabeth R. Jessup, and Thomas Manteuffel. "A
        // technique for accelerating the convergence of restarted GMRES." SIAM
        // Journal on Matrix Analysis and Applications 26, no. 4 (2005):
        // 962-984.
        session->LoadParameter("GMRESDeltaDirection", m_GMRESDeltaDirection, 0);

        session->LoadParameter("GMRESMaxHessMatBand", m_KrylovMaxHessMatBand,
                               m_LinSysMaxStorage + 1);
        session->MatchSolverInfo("GMRESCentralDifference", "True",
                                 m_GMRESCentralDifference, false);
        m_flexible = session->DefinesParameter("FlexibleGMRES")
                         ? session->GetParameter("FlexibleGMRES")
                         : false;
        m_isModifiedGramSchmidt =
            session->DefinesParameter("ModifiedGramSchmidt")
                ? session->GetParameter("ModifiedGramSchmidt")
                : true;
        m_reorthogonalize = session->DefinesParameter("GMRESReorthogonalize")
                                ? session->GetParameter("GMRESReorthogonalize")
                                : false;

        ASSERTL0(!(m_flexible && this->m_leftPreconditioner),
                 "Flexible GMRES only avaible with right preconditioner");

        ASSERTL0(!(m_flexible && m_GMRESDeltaDirection),
                 "Can't both use Flexible GMRES and GMRESDeltaDirection "
                 "(LGMRES) at the same time");

        ASSERTL0(m_GMRESDeltaDirection < m_LinSysMaxStorage,
                 "GMRESDeltaDirection (LGMRES) must be smaller than "
                 "LinSysMaxStorage");

        // Allocate array storage.
        if (!m_isModifiedGramSchmidt)
        {
            m_vExchange = LibUtilities::MemoryRegion<TData>(
                (m_reorthogonalize ? 2 : 1) * m_LinSysMaxStorage, eHostPinned);
        }
        m_coeffs = LibUtilities::MemoryRegion<TData>(
            m_LinSysMaxStorage + m_GMRESDeltaDirection, eHostPinned);
        m_V.ResizeNumField(1);
        m_Z.ResizeNumField(1);

        m_truncted = (m_KrylovMaxHessMatBand > 0);
        m_hes      = std::vector<std::vector<TData>>(m_LinSysMaxStorage);
        m_upper    = std::vector<std::vector<TData>>(m_LinSysMaxStorage);
        m_id       = std::vector<unsigned int>(m_LinSysMaxStorage);
        m_id_start = std::vector<unsigned int>(m_LinSysMaxStorage);
        m_id_end   = std::vector<unsigned int>(m_LinSysMaxStorage);
        for (unsigned int nd = 0; nd < m_LinSysMaxStorage; nd++)
        {
            m_hes[nd]    = std::vector<TData>(m_LinSysMaxStorage + 1, 0.0);
            m_upper[nd]  = std::vector<TData>(m_LinSysMaxStorage + 1, 0.0);
            m_id[nd]     = nd;
            m_id_end[nd] = nd + 1;
            if (m_truncted && m_id_end[nd] > m_KrylovMaxHessMatBand)
            {
                m_id_start[nd] = m_id_end[nd] - m_KrylovMaxHessMatBand;
            }
            else
            {
                m_id_start[nd] = 0;
            }
        }

        // Set storage of LGMRES.
        m_delta.ResizeNumField(m_GMRESDeltaDirection);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<GMRESOpImpl<ExecSpace, TData>>(expansionList,
                                                               components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_w;
    LibUtilities::Field<TData, FieldState::Coeff> m_r0;
    LibUtilities::MultiField<TData, FieldState::Coeff> m_V;
    LibUtilities::MultiField<TData, FieldState::Coeff> m_Z;
    /// LGMRES directions, as a ring: the newest is at m_deltaHead.
    LibUtilities::MultiField<TData, FieldState::Coeff> m_delta;
    std::vector<std::vector<TData>> m_hes;
    std::vector<std::vector<TData>> m_upper;
    std::vector<unsigned int> m_id;
    std::vector<unsigned int> m_id_start;
    std::vector<unsigned int> m_id_end;

    LibUtilities::MemoryRegion<TData> m_vExchange;
    LibUtilities::MemoryRegion<TData> m_coeffs; ///< Solution coefficients.

    unsigned int m_deltaHead = 0;
    bool m_flexible;
    bool m_truncted;
    bool m_isModifiedGramSchmidt = true;
    bool m_reorthogonalize       = false;
    bool m_GMRESCentralDifference;
    unsigned int m_GMRESDeltaDirection;
    unsigned int m_LinSysMaxStorage;
    unsigned int m_KrylovMaxHessMatBand;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // Allocate array storage.
        // Residual
        std::vector<TData> eta(m_LinSysMaxStorage + 1);
        // Givens rotation c
        std::vector<TData> cs(m_LinSysMaxStorage);
        // Givens rotation s
        std::vector<TData> sn(m_LinSysMaxStorage);
        // Total coefficients
        std::vector<TData> yn(m_LinSysMaxStorage);
        // Search direction order
        this->m_niter   = 0;
        unsigned int ii = 0, outerIterations = 0;
        bool converged    = false;
        TData prec_factor = 1.0, eps = 0.0, eps0 = 1.0;

        // Reshape mask if required.
        this->template ReshapeMask<ExecSpace>(in);

        // Calculate rhs magnitude.
        this->m_assmbScatrOp->Apply(in, m_w);
        TData rhsMagnitude = this->m_math.ddot(in, m_w);
        this->m_rowComm->AllReduce(rhsMagnitude,
                                   Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = this->GetRhsMagnitude(rhsMagnitude);

        // Calculate prefactor.
        if (this->m_leftPreconditioner)
        {
            this->m_assmbScatrZeroDirOp->Apply(in, m_w);
            prec_factor = this->m_math.ddot(in, m_w);
            this->m_rowComm->AllReduce(prec_factor, LibUtilities::ReduceSum);
        }

        // GMRES with restart.
        while (true)
        {
            if (this->m_niter == this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << this->m_niter
                    << ". Increase NekLinSysMaxIterations in the session "
                       "PARAMETERS section to allow more iterations.";
                NEKERROR(ErrorUtil::efatal, msg.str());

                break;
            }

            std::fill_n(eta.begin(), m_LinSysMaxStorage + 1, 0.0);
            if (outerIterations == 0)
            {
                // Set the fields to zero.
                out.template Initialize<MemSpace>(0);
                out.SetInterleaveWidth(in);

                // If not restarted, x0 should be zero
                m_r0.template Copy<MemSpace>(in);
            }
            else
            {
                // This is A*x
                this->m_lhs->Apply(out, m_r0);
                this->m_robBndCondOp->Apply(out, m_r0);

                // This is r0 = b-A*x
                Math::sub<ExecSpace>(in, m_r0, m_r0);
            }

            // Assemble the residual. The search vectors are assembled and
            // scattered, with zero Dirichlet coefficients, so that each
            // global coefficient has one value however many elements share
            // it.
            this->m_assmbScatrZeroDirOp->Apply(m_r0);

            // Apply preconditioner.
            if (this->m_leftPreconditioner)
            {
                this->m_precon->Apply(m_r0, m_r0);
            }

            // Norm of (r0)
            eps = this->m_math.ddot(this->m_mask, m_r0, m_r0);
            this->m_rowComm->AllReduce(eps, LibUtilities::ReduceSum);
            if (this->m_leftPreconditioner && outerIterations == 0)
            {
                eps0 = eps;
            }

            // If the input residual is less than tolerance then skip solve.
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                return;
            }

            if (this->m_leftPreconditioner)
            {
                eta[0] = std::sqrt(prec_factor * eps / eps0);
            }
            else
            {
                eta[0] = std::sqrt(eps);
            }

            // Initial search vector.
            Math::mul<ExecSpace>((TData)1.0 / std::sqrt(eps), m_r0, m_V[0]);

            // Inner loop.
            while (true)
            {
                if ((ii == m_LinSysMaxStorage) ||
                    (this->m_niter == this->m_maxIter))
                {
                    break;
                }

                // For LGMRES use m_delta for the last m_GMRESDeltaDirection
                // iterations.
                bool cond =
                    ii >= (m_LinSysMaxStorage - m_GMRESDeltaDirection) &&
                    outerIterations >= m_GMRESDeltaDirection;
                unsigned int index =
                    ii - (m_LinSysMaxStorage - m_GMRESDeltaDirection);
                auto &V1 =
                    (cond)
                        ? m_delta[(m_deltaHead + index) % m_GMRESDeltaDirection]
                        : m_V[ii];

                auto &Z1              = (this->m_rightPreconditioner)
                                            ? m_Z[(m_flexible) ? ii : 0]
                                            : V1;
                auto &h1              = m_hes[ii];
                auto &h2              = m_upper[ii];
                unsigned int idtem    = m_id[ii];
                unsigned int starttem = m_id_start[idtem];
                unsigned int endtem   = m_id_end[idtem];

                // Apply preconditioner.
                if (this->m_rightPreconditioner)
                {
                    this->m_precon->Apply(V1, Z1);
                }

                // -- Begin Arnoldi --
                // Apply lhs and assemble.
                this->m_lhs->Apply(Z1, m_w);
                this->m_robBndCondOp->Apply(Z1, m_w);
                this->m_assmbScatrZeroDirOp->Apply(m_w);

                // Apply preconditioner. Its scaling of m_w is applied with
                // the first Gram-Schmidt update.
                TData wScale = 1.0;
                if (this->m_leftPreconditioner)
                {
                    this->m_precon->Apply(m_w, m_w);
                    wScale = std::sqrt(prec_factor / eps0);
                }

                if (m_isModifiedGramSchmidt)
                {
                    // Modified Gram-Schmidt.
                    for (unsigned int i = starttem; i < endtem; ++i)
                    {
                        h1[i] = this->m_math.ddot(this->m_mask, m_w, m_V[i]);
                        this->m_rowComm->AllReduce(h1[i],
                                                   LibUtilities::ReduceSum);
                        if (i == starttem)
                        {
                            h1[i] *= wScale;
                            Math::daxpby<ExecSpace>(-h1[i], m_V[i], wScale, m_w,
                                                    m_w);
                        }
                        else
                        {
                            Math::daxpy<ExecSpace>(-h1[i], m_V[i], m_w, m_w);
                        }
                    }

                    // Calculate the L2 norm and normalize.
                    h1[endtem] = this->m_math.ddot(this->m_mask, m_w, m_w);
                    this->m_rowComm->AllReduce(h1[endtem],
                                               LibUtilities::ReduceSum);
                    h1[endtem] = std::sqrt(h1[endtem]);
                }
                else
                {
                    // Classical Gram-Schmidt: the dot products d with the
                    // basis, then w = wScale (w - V d), so that h = wScale d.
                    // Reset device memory.
                    auto exchange =
                        m_vExchange.template GetPtr<MemSpace, WriteOnly>();
                    MultiDot<ExecSpace>(this->m_mask, m_V, starttem, endtem,
                                        m_w, exchange + starttem);

                    // Communication.
                    this->m_rowComm->template AllReduce<MemSpace>(
                        m_vExchange, LibUtilities::ReduceSum);

                    MultiAxpy<ExecSpace>(
                        -wScale, m_V, starttem, endtem,
                        m_vExchange.template GetPtr<MemSpace, ReadOnly>() +
                            starttem,
                        wScale, m_w);

                    // Device-to-host copy.
                    auto exchangeHost =
                        m_vExchange.template GetPtr<NektarSpaces::HostSpace,
                                                    ReadOnly>();
                    for (unsigned int i = starttem; i < endtem; ++i)
                    {
                        h1[i] = wScale * exchangeHost[i];
                    }

                    // Second pass, to recover the orthogonality lost to
                    // rounding.
                    if (m_reorthogonalize)
                    {
                        const unsigned int offset =
                            m_LinSysMaxStorage + starttem;

                        // Reset device memory.
                        exchange =
                            m_vExchange.template GetPtr<MemSpace, WriteOnly>();
                        MultiDot<ExecSpace>(this->m_mask, m_V, starttem, endtem,
                                            m_w, exchange + offset);

                        // Communication.
                        this->m_rowComm->template AllReduce<MemSpace>(
                            m_vExchange, LibUtilities::ReduceSum);

                        MultiAxpy<ExecSpace>(
                            -1.0, m_V, starttem, endtem,
                            m_vExchange.template GetPtr<MemSpace, ReadOnly>() +
                                offset,
                            1.0, m_w);

                        // Device-to-host copy.
                        exchangeHost =
                            m_vExchange.template GetPtr<NektarSpaces::HostSpace,
                                                        ReadOnly>();
                        for (unsigned int i = starttem; i < endtem; ++i)
                        {
                            h1[i] += exchangeHost[m_LinSysMaxStorage + i];
                        }
                    }

                    // Calculate the L2 norm and normalize.
                    h1[endtem] = this->m_math.ddot(this->m_mask, m_w, m_w);
                    this->m_rowComm->AllReduce(h1[endtem],
                                               LibUtilities::ReduceSum);
                    h1[endtem] = std::sqrt(h1[endtem]);
                }
                // -- End Arnoldi --

                if (starttem > 0)
                {
                    starttem = starttem - 1;
                }

                std::copy_n(h1.data(), m_LinSysMaxStorage + 1, h2.data());
                this->DoGivensRotation(starttem, endtem, cs, sn, h2, eta);

                eps = eta[ii + 1] * eta[ii + 1];

                ii++;
                this->m_niter++;

                // This Gmres merge truncted Gmres to accelerate.
                // If truncted, cannot jump out because
                // the last term of eta is not residual
                if ((!m_truncted) || (ii <= m_KrylovMaxHessMatBand))
                {
                    if (eps < this->m_tol * this->m_tol * rhsMagnitude)
                    {
                        converged = true;
                        break;
                    }
                }

                // Allocate new storage, if necessary.
                m_V.ResizeNumField(ii + 1);
                if (m_flexible)
                {
                    m_Z.ResizeNumField(ii + 1);
                }

                // Compute new search vector.
                Math::mul<ExecSpace>((TData)1.0 / h1[endtem], m_w, m_V[ii]);
            }

            // Do backward substitution.
            this->DoBackward(ii, m_upper, eta, yn);

            // Calculate solution delta. For LGMRES the last
            // m_GMRESDeltaDirection directions are those of m_delta.
            auto &Z = (m_flexible) ? m_Z : m_V;
            const unsigned int nDelta =
                (outerIterations >= m_GMRESDeltaDirection &&
                 ii > m_LinSysMaxStorage - m_GMRESDeltaDirection)
                    ? ii - (m_LinSysMaxStorage - m_GMRESDeltaDirection)
                    : 0;
            const unsigned int nZ = ii - nDelta;

            auto coeffsHost =
                m_coeffs.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            std::copy_n(yn.data(), nZ, coeffsHost);
            for (unsigned int k = 0; k < nDelta; ++k)
            {
                coeffsHost[m_LinSysMaxStorage +
                           (m_deltaHead + k) % m_GMRESDeltaDirection] =
                    yn[nZ + k];
            }
            for (unsigned int k = nDelta; k < m_GMRESDeltaDirection; ++k)
            {
                coeffsHost[m_LinSysMaxStorage +
                           (m_deltaHead + k) % m_GMRESDeltaDirection] = 0.0;
            }
            auto coeffs = m_coeffs.template GetPtr<MemSpace, ReadOnly>();

            TData scaleW = 0.0;
            if (nZ > 0)
            {
                MultiAxpy<ExecSpace>(1.0, Z, 0, nZ, coeffs, 0.0, m_w);
                scaleW = 1.0;
            }
            if (nDelta > 0)
            {
                MultiAxpy<ExecSpace>(1.0, m_delta, 0, m_GMRESDeltaDirection,
                                     coeffs + m_LinSysMaxStorage, scaleW, m_w);
            }

            // Store last m_GMRESDeltaDirection delta for LGMRES, the oldest
            // replaced by the newest.
            if (m_GMRESDeltaDirection)
            {
                m_deltaHead = (m_deltaHead + m_GMRESDeltaDirection - 1) %
                              m_GMRESDeltaDirection;
                m_delta[m_deltaHead].template Copy<MemSpace>(m_w);
            }

            // Apply preconditioner.
            if (!m_flexible && this->m_rightPreconditioner)
            {
                this->m_precon->Apply(m_w, m_w);
            }

            // Update solution.
            Math::add<ExecSpace>(m_w, out, out);

            ii = 0;
            outerIterations++;

            if (converged)
            {
                break;
            }
        }

        // Print output.
        if (this->IsVerboseOutputEnabled())
        {
            TData eps_real;

            // Calculate difference in residual of solution.
            this->m_lhs->Apply(out, m_r0);
            this->m_robBndCondOp->Apply(out, m_r0);
            Math::sub<ExecSpace>(in, m_r0, m_r0);
            this->m_assmbScatrZeroDirOp->Apply(m_r0, m_w);
            eps_real = this->m_math.ddot(m_w, m_r0);
            this->m_rowComm->AllReduce(eps_real, LibUtilities::ReduceSum);

            this->PrintVerboseOutput(
                this->name, "error",
                std::sqrt(eps / eps0 * prec_factor / rhsMagnitude),
                rhsMagnitude, [&](std::ostream &out) {
                    out << " WITH (GMRES eps = " << eps
                        << " REAL eps= " << eps_real << ")";
                    out << (converged ? " CONVERGED"
                                      : " WARNING: Exceeded maxIt");
                });
        }

        WARNINGL1(converged, "GMRES did not converge.");
    }
};

} // namespace Nektar::SolverCore::detail
