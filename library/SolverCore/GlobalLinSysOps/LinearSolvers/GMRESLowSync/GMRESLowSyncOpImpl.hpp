///////////////////////////////////////////////////////////////////////////////
//
// File: GMRESLowSyncOpImpl.hpp
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
// Description: GMRES with the low-synchronisation modified Gram-Schmidt
// process.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "SolverCore/GlobalLinSysOps/LinearSolvers/GMRESLowSync/GMRESLowSyncOp.hpp"
#include "SolverCore/GlobalLinSysOps/MultiFieldHelper/MultiFieldHelper.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

/**
 * @brief Restarted GMRES orthogonalising its Krylov basis with a
 * low-synchronisation Gram-Schmidt process, taking one global reduction per
 * iteration.
 *
 * Each basis vector is normalised an iteration late: its norm, the last entry
 * of the previous Hessenberg column, comes with the products of the next
 * iteration.
 *
 * With the modified process, the projections of a new vector w on the basis
 * V are those of its inverse compact WY form, h = (I + L)^{-1} V^T w, L being
 * the strictly lower part of V^T V. The basis is stored unnormalised, column
 * k scaled by m_basisScale[k].
 *
 * With the classical process applied twice, the second projection of each
 * basis vector is also an iteration late: the products of the next iteration
 * give its coefficients, added to the previous Hessenberg column, and its
 * norm. The matrix-vector product is taken before it, and the Hessenberg
 * column of the product corrected for it.
 *
 * Reference:
 * Swirydowicz, Katarzyna, Julien Langou, Shreyas Ananthan, Ulrike Yang, and
 * Stephen Thomas. "Low synchronization Gram-Schmidt and generalized minimal
 * residual algorithms." Numerical Linear Algebra with Applications 28, no. 2
 * (2021): e2343.
 */
template <typename ExecSpace, typename TData>
class GMRESLowSyncOpImpl : public GMRESLowSyncOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    GMRESLowSyncOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                       const std::vector<std::string> &components)
        : GMRESLowSyncOp<TData>(expansionList, components),
          m_w(LibUtilities::Field<TData, FieldState::Coeff>(
              "GMRESLowSync w",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_r0(LibUtilities::Field<TData, FieldState::Coeff>(
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_V("GMRESLowSync V",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components),
          m_Z("GMRESLowSync Z",
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
        m_LinSysMaxStorage = session->DefinesParameter("LinSysMaxStorage")
                                 ? session->GetParameter("LinSysMaxStorage")
                                 : 100;
        m_flexible         = session->DefinesParameter("FlexibleGMRES")
                                 ? session->GetParameter("FlexibleGMRES")
                                 : false;
        m_isModifiedGramSchmidt =
            session->DefinesParameter("ModifiedGramSchmidt")
                ? session->GetParameter("ModifiedGramSchmidt")
                : true;

        ASSERTL0(!(m_flexible && this->m_leftPreconditioner),
                 "Flexible GMRES only avaible with right preconditioner");

        // Allocate array storage: the products of the basis with two
        // vectors, and two entries at least, for the reductions before the
        // iterations.
        m_vExchange = LibUtilities::MemoryRegion<TData>(
            std::max(2u * m_LinSysMaxStorage, 2u), eHostPinned);
        m_coeffs = LibUtilities::MemoryRegion<TData>(2 * m_LinSysMaxStorage,
                                                     eHostPinned);
        m_V.ResizeNumField(2);
        m_Z.ResizeNumField(1);

        m_hes   = std::vector<std::vector<TData>>(m_LinSysMaxStorage);
        m_upper = std::vector<std::vector<TData>>(m_LinSysMaxStorage);
        for (unsigned int nd = 0; nd < m_LinSysMaxStorage; nd++)
        {
            m_hes[nd]   = std::vector<TData>(m_LinSysMaxStorage + 1, 0.0);
            m_upper[nd] = std::vector<TData>(m_LinSysMaxStorage + 1, 0.0);
        }

        m_basisScale = std::vector<TData>(m_LinSysMaxStorage, 1.0);
        m_basisGram  = std::vector<std::vector<TData>>(
            m_LinSysMaxStorage, std::vector<TData>(m_LinSysMaxStorage));
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<GMRESLowSyncOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_w;
    LibUtilities::Field<TData, FieldState::Coeff> m_r0;
    LibUtilities::MultiField<TData, FieldState::Coeff> m_V;
    LibUtilities::MultiField<TData, FieldState::Coeff> m_Z;
    std::vector<std::vector<TData>> m_hes;
    std::vector<std::vector<TData>> m_upper;
    /// Factor of each basis vector to its normalised value.
    std::vector<TData> m_basisScale;
    /// Strictly lower part of the Gram matrix of the normalised basis, by
    /// row.
    std::vector<std::vector<TData>> m_basisGram;

    LibUtilities::MemoryRegion<TData> m_vExchange;
    LibUtilities::MemoryRegion<TData> m_coeffs; ///< Solution coefficients.

    bool m_flexible;
    bool m_isModifiedGramSchmidt = true;
    unsigned int m_LinSysMaxStorage;

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
        this->m_niter   = 0;
        unsigned int ii = 0, outerIterations = 0;
        bool converged    = false;
        TData prec_factor = 1.0, eps = 0.0, eps0 = 1.0;

        // Reshape mask if required.
        this->template ReshapeMask<ExecSpace>(in);

        // Reset device memory.
        auto exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();

        // Calculate rhs magnitude.
        this->m_assmbScatrOp->Apply(in, m_w);
        Math::ddot<ExecSpace>(in, m_w, exchange + 0);

        // Calculate prefactor.
        if (this->m_leftPreconditioner)
        {
            this->m_lhs->Assemble(in, m_w);
            Math::ddot<ExecSpace>(in, m_w, exchange + 1);
        }

        // Communication.
        this->m_rowComm->template AllReduce<MemSpace>(
            m_vExchange, Nektar::LibUtilities::ReduceSum);

        // Device-to-host copy.
        auto exchangeHost =
            m_vExchange.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        TData rhsMagnitude = this->GetRhsMagnitude(exchangeHost[0]);
        if (this->m_leftPreconditioner)
        {
            prec_factor = exchangeHost[1];
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
                this->m_lhs->ApplyLocal(out, m_r0);

                // This is r0 = b-A*x
                Math::sub<ExecSpace>(in, m_r0, m_r0);
            }

            // Assemble the residual. The search vectors are assembled and
            // scattered, with zero Dirichlet coefficients, so that each
            // global coefficient has one value however many elements share
            // it.
            this->m_lhs->Assemble(m_r0);

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

            // Inner loop. Each iteration completes the previous Hessenberg
            // column and computes the projections of the next one.
            while (true)
            {
                if ((ii == m_LinSysMaxStorage) ||
                    (this->m_niter == this->m_maxIter))
                {
                    break;
                }

                auto &V1 = m_V[ii];
                auto &W  = m_V[ii + 1];
                auto &Z1 = (this->m_rightPreconditioner)
                               ? m_Z[(m_flexible) ? ii : 0]
                               : V1;
                auto &h1 = m_hes[ii];

                // Apply preconditioner.
                if (this->m_rightPreconditioner)
                {
                    this->m_precon->Apply(V1, Z1);
                }

                // -- Begin Arnoldi --
                // Apply lhs and assemble.
                this->m_lhs->Apply(Z1, W);

                // Apply preconditioner. Its scaling of W is applied with the
                // projections.
                TData wScale = 1.0;
                if (this->m_leftPreconditioner)
                {
                    this->m_precon->Apply(W, W);
                    wScale = std::sqrt(prec_factor / eps0);
                }

                // Products of V[0], ..., V[ii] with V[ii] and with W.
                // Reset device memory.
                exchange = m_vExchange.template GetPtr<MemSpace, WriteOnly>();
                MultiDot2<ExecSpace>(this->m_mask, m_V, 0, ii + 1, V1, W,
                                     exchange);

                // Communication.
                this->m_rowComm->template AllReduce<MemSpace>(
                    m_vExchange, LibUtilities::ReduceSum);

                // Device-to-host copy.
                exchangeHost =
                    m_vExchange
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
                const TData *dotV = exchangeHost;
                const TData *dotW = exchangeHost + ii + 1;

                // Complete the previous column with the norm of V[ii]. With
                // the classical process, V[ii] is first projected again on
                // V[0], ..., V[ii - 1], with coefficients a = dotV, which are
                // added to the column; its norm follows by Pythagoras.
                TData norm2 = dotV[ii];
                if (!m_isModifiedGramSchmidt)
                {
                    for (unsigned int k = 0; k < ii; ++k)
                    {
                        norm2 -= dotV[k] * dotV[k];
                    }
                }
                const TData norm = std::sqrt(norm2);
                if (ii > 0)
                {
                    auto &h2 = m_upper[ii - 1];
                    if (!m_isModifiedGramSchmidt)
                    {
                        for (unsigned int k = 0; k < ii; ++k)
                        {
                            m_hes[ii - 1][k] += dotV[k];
                        }
                    }
                    m_hes[ii - 1][ii] = norm;
                    std::copy_n(m_hes[ii - 1].data(), m_LinSysMaxStorage + 1,
                                h2.data());
                    this->DoGivensRotation(0, ii, cs, sn, h2, eta);

                    eps = eta[ii] * eta[ii];
                    if (eps < this->m_tol * this->m_tol * rhsMagnitude)
                    {
                        converged = true;
                        break;
                    }
                }

                auto coeffsHost =
                    m_coeffs
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                if (m_isModifiedGramSchmidt)
                {
                    // Normalise V[ii]: its scale and its row of L.
                    m_basisScale[ii] = 1.0 / norm;
                    for (unsigned int k = 0; k < ii; ++k)
                    {
                        m_basisGram[ii][k] =
                            m_basisScale[ii] * m_basisScale[k] * dotV[k];
                    }

                    // Projections h = (I + L)^{-1} V^T w, with w = wFactor W.
                    const TData wFactor = wScale * m_basisScale[ii];
                    for (unsigned int k = 0; k <= ii; ++k)
                    {
                        h1[k] = m_basisScale[k] * wFactor * dotW[k];
                        for (unsigned int i = 0; i < k; ++i)
                        {
                            h1[k] -= m_basisGram[k][i] * h1[i];
                        }
                    }

                    // W = wFactor W - V h, the orthogonalised, unnormalised
                    // vector.
                    for (unsigned int k = 0; k <= ii; ++k)
                    {
                        coeffsHost[k] = h1[k] * m_basisScale[k];
                    }
                    MultiAxpy<ExecSpace>(
                        -1.0, m_V, 0, ii + 1,
                        m_coeffs.template GetPtr<MemSpace, ReadOnly>(), wFactor,
                        W);
                }
                else
                {
                    // Projections h of w = wScale A v on V[0], ..., V[ii],
                    // v = (V[ii] - V a) / norm being the normalised V[ii]. W
                    // is the product with V[ii], so w = wFactor W - H a /
                    // norm, H being the previous Hessenberg columns.
                    const TData wFactor = wScale / norm;
                    TData dotVW         = dotW[ii];
                    for (unsigned int k = 0; k < ii; ++k)
                    {
                        dotVW -= dotV[k] * dotW[k];
                    }
                    dotVW /= norm;
                    for (unsigned int k = 0; k < ii; ++k)
                    {
                        h1[k] = wFactor * dotW[k];
                    }
                    h1[ii] = wFactor * dotVW;
                    for (unsigned int l = 0; l < ii; ++l)
                    {
                        for (unsigned int k = 0; k <= l + 1; ++k)
                        {
                            h1[k] -= m_hes[l][k] * dotV[l] / norm;
                        }
                    }

                    // V[ii] = (V[ii] - V a) / norm and the orthogonalised,
                    // unnormalised vector W = wFactor (W - V dotW - c (V[ii]
                    // - V a)), c = dotVW / norm, V being V[0..ii - 1]: both
                    // products with V in one pass, scaled by 1 / norm, then
                    // W = wScale (W - c V[ii]).
                    const TData c = dotVW / norm;
                    for (unsigned int k = 0; k < ii; ++k)
                    {
                        coeffsHost[k]      = dotV[k];
                        coeffsHost[ii + k] = dotW[k];
                    }
                    TData scaleW = wScale;
                    if (ii > 0)
                    {
                        auto coeffs =
                            m_coeffs.template GetPtr<MemSpace, ReadOnly>();
                        MultiAxpy2<ExecSpace>(-1.0 / norm, m_V, 0, ii, coeffs,
                                              1.0 / norm, m_V, ii);
                        if (m_flexible)
                        {
                            MultiAxpy<ExecSpace>(-1.0 / norm, m_Z, 0, ii,
                                                 coeffs, 1.0 / norm, Z1);
                        }
                    }
                    else
                    {
                        Math::mul<ExecSpace>((TData)1.0 / norm, V1, V1);
                        if (m_flexible)
                        {
                            Math::mul<ExecSpace>((TData)1.0 / norm, Z1, Z1);
                        }
                        scaleW = wFactor;
                    }
                    Math::daxpby<ExecSpace>(scaleW, W, -wScale * c, V1, W);
                }
                // -- End Arnoldi --

                ii++;
                this->m_niter++;

                // Allocate new storage, if necessary.
                m_V.ResizeNumField(ii + 2);
                if (m_flexible)
                {
                    m_Z.ResizeNumField(ii + 1);
                }
            }

            // Complete the last column with the norm of V[ii], projected
            // again on V[0], ..., V[ii - 1] with the classical process.
            if (!converged)
            {
                TData norm2 = 0.0;
                if (m_isModifiedGramSchmidt)
                {
                    norm2 = this->m_math.ddot(this->m_mask, m_V[ii], m_V[ii]);
                    this->m_rowComm->AllReduce(norm2, LibUtilities::ReduceSum);
                }
                else
                {
                    // Products of V[0], ..., V[ii] with V[ii].
                    // Reset device memory.
                    exchange =
                        m_vExchange.template GetPtr<MemSpace, WriteOnly>();
                    MultiDot<ExecSpace>(this->m_mask, m_V, 0, ii + 1, m_V[ii],
                                        exchange);

                    // Communication.
                    this->m_rowComm->template AllReduce<MemSpace>(
                        m_vExchange, LibUtilities::ReduceSum);

                    // Device-to-host copy.
                    exchangeHost =
                        m_vExchange.template GetPtr<NektarSpaces::HostSpace,
                                                    ReadOnly>();

                    norm2 = exchangeHost[ii];
                    for (unsigned int k = 0; k < ii; ++k)
                    {
                        m_hes[ii - 1][k] += exchangeHost[k];
                        norm2 -= exchangeHost[k] * exchangeHost[k];
                    }
                }

                auto &h2          = m_upper[ii - 1];
                m_hes[ii - 1][ii] = std::sqrt(norm2);
                std::copy_n(m_hes[ii - 1].data(), m_LinSysMaxStorage + 1,
                            h2.data());
                this->DoGivensRotation(0, ii, cs, sn, h2, eta);

                eps       = eta[ii] * eta[ii];
                converged = eps < this->m_tol * this->m_tol * rhsMagnitude;
            }

            // Do backward substitution.
            this->DoBackward(ii, m_upper, eta, yn);

            // Calculate solution delta, scaling the coefficients of the
            // unnormalised basis.
            auto &Z = (m_flexible) ? m_Z : m_V;
            auto coeffsHost =
                m_coeffs.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int k = 0; k < ii; ++k)
            {
                coeffsHost[k] = yn[k] * m_basisScale[k];
            }
            MultiAxpy<ExecSpace>(1.0, Z, 0, ii,
                                 m_coeffs.template GetPtr<MemSpace, ReadOnly>(),
                                 0.0, m_w);

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
            this->m_lhs->ApplyLocal(out, m_r0);
            Math::sub<ExecSpace>(in, m_r0, m_r0);
            this->m_lhs->Assemble(m_r0, m_w);
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

        WARNINGL1(converged, "GMRESLowSync did not converge.");
    }
};

} // namespace Nektar::SolverCore::detail
