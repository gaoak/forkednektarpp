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

#include "Operators/GlobalLinSysOps/LinearSolvers/GMRES/GMRESOp.hpp"

#include <iomanip>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class GMRESOpImpl : public GMRESOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    GMRESOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : GMRESOp<TData>(expansionList, components),
          m_w(Field<TData, FieldState::Coeff>(
              "GMRES w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_wk(Field<TData, FieldState::Coeff>(
              "GMRES wk",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r0(Field<TData, FieldState::Coeff>(
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1))
    {
        this->template SetLinearSolver<ExecSpace>();

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
                : 50;
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

        ASSERTL0(!(m_flexible && this->m_leftPreconditioner),
                 "Flexible GMRES only avaible with right preconditioner");

        ASSERTL0(!(m_flexible && m_GMRESDeltaDirection),
                 "Can't both use Flexible GMRES and GMRESDeltaDirection "
                 "(LGMRES) at the same time");

        // Allocate array storage.
        if (!m_isModifiedGramSchmidt)
        {
            m_vExchange = MemoryRegion<TData>(m_LinSysMaxStorage, ePinned);
        }

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
        for (unsigned int dir = 0; dir < m_GMRESDeltaDirection; dir++)
        {
            m_delta.push_back(Field<TData, FieldState::Coeff>(
                "GMRESOp delta" + std::to_string(dir),
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<GMRESOpImpl<ExecSpace, TData>>(expansionList,
                                                               components);
    }

protected:
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_wk;
    Field<TData, FieldState::Coeff> m_r0;
    std::vector<Field<TData, FieldState::Coeff>> m_V;
    std::vector<Field<TData, FieldState::Coeff>> m_Z;
    std::deque<Field<TData, FieldState::Coeff>> m_delta;
    std::vector<std::vector<TData>> m_hes;
    std::vector<std::vector<TData>> m_upper;
    std::vector<unsigned int> m_id;
    std::vector<unsigned int> m_id_start;
    std::vector<unsigned int> m_id_end;

    MemoryRegion<TData> m_vExchange;

    TData rhsMagnitude = NekConstants::kNekUnsetDouble;
    bool m_verbose     = true;
    bool m_flexible;
    bool m_truncted;
    bool m_isModifiedGramSchmidt = true;
    bool m_GMRESCentralDifference;
    unsigned int m_GMRESDeltaDirection;
    unsigned int m_LinSysMaxStorage;
    unsigned int m_KrylovMaxHessMatBand;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
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
        TData prec_factor = 1.0, eps, eps0 = 1.0;

        // Calculate rhs magnitude.
        if (rhsMagnitude == NekConstants::kNekUnsetDouble)
        {
            this->m_assmbScatrOp->Apply(in, m_w);
            rhsMagnitude = this->m_math.ddot(in, m_w);
            this->m_rowComm->AllReduce(rhsMagnitude,
                                       Nektar::LibUtilities::ReduceSum);
            rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;
        }

        // Calculate prefactor.
        if (this->m_leftPreconditioner)
        {
            this->m_assmbScatrZeroDirOp->Apply(in, m_w);
            prec_factor = this->m_math.ddot(in, m_w);
            this->m_rowComm->AllReduce(prec_factor, LibUtilities::ReduceSum);
        }

        // Allocate memory, if necessary.
        if (m_V.size() == 0)
        {
            m_V.push_back(Field<TData, FieldState::Coeff>(
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));
            m_Z.push_back(Field<TData, FieldState::Coeff>(
                GetBlockAttributes<TData, FieldState::Coeff>(
                    this->m_expansionList),
                this->m_components, 1));
        }

        // GMRES with restart.
        while (true)
        {
            if (this->m_niter == this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << this->m_niter;
                WARNINGL0(false, msg.str());

                break;
            }

            std::fill_n(eta.begin(), m_LinSysMaxStorage + 1, 0.0);
            if (outerIterations == 0)
            {
                // Set the fields to zero.
                out.template Initialize<MemSpace>(0);

                // If not restarted, x0 should be zero
                m_r0.template Copy<MemSpace>(in);
            }
            else
            {
                // This is A*x
                this->m_lhs->Apply(out, m_r0);
                this->m_robBndCondOp->Apply(out, m_r0);

                // This is r0 = b-A*x
                sub<ExecSpace>(in, m_r0, m_r0);
            }

            // Apply preconditioner.
            if (this->m_leftPreconditioner)
            {
                this->m_assmbScatrZeroDirOp->Apply(m_r0);
                this->m_precon->Apply(m_r0, m_r0);
            }

            // Norm of (r0)
            if (m_isModifiedGramSchmidt)
            {
                this->m_assmbScatrZeroDirOp->Apply(m_r0, m_wk);
                eps = this->m_math.ddot(m_r0, m_wk);
            }
            else
            {
                this->m_assmbScatrZeroDirOp->Apply(m_r0);
                eps = this->m_math.ddot(m_r0, m_r0);
            }
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
                mul<ExecSpace>(std::sqrt(prec_factor / eps0), m_r0, m_r0);
                eta[0] = std::sqrt(prec_factor * eps / eps0);
            }
            else
            {
                eta[0] = std::sqrt(eps);
            }

            // Initial search vector.
            mul<ExecSpace>((TData)1.0 / eta[0], m_r0, m_V[0]);

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
                auto &V1 = (cond) ? m_delta[index] : m_V[ii];

                auto &Z1      = (this->m_rightPreconditioner)
                                    ? m_Z[(m_flexible) ? ii : 0]
                                    : V1;
                auto &h1      = m_hes[ii];
                auto &h2      = m_upper[ii];
                auto idtem    = m_id[ii];
                auto starttem = m_id_start[idtem];
                auto endtem   = m_id_end[idtem];

                // Apply preconditioner.
                if (this->m_rightPreconditioner)
                {
                    this->m_assmbScatrZeroDirOp->Apply(V1, Z1);
                    this->m_precon->Apply(Z1, Z1);
                }

                // -- Begin Arnoldi --
                // Apply lhs.
                this->m_lhs->Apply(Z1, m_w);
                this->m_robBndCondOp->Apply(Z1, m_w);
                if (!m_isModifiedGramSchmidt)
                {
                    this->m_assmbScatrZeroDirOp->Apply(m_w);
                }

                // Apply preconditioner.
                if (this->m_leftPreconditioner)
                {
                    this->m_assmbScatrZeroDirOp->Apply(m_w);
                    this->m_precon->Apply(m_w, m_w);
                    mul<ExecSpace>(std::sqrt(prec_factor / eps0), m_w, m_w);
                }

                if (m_isModifiedGramSchmidt)
                {
                    // Modified Gram-Schmidt.
                    for (unsigned int i = starttem; i < endtem; ++i)
                    {
                        this->m_assmbScatrZeroDirOp->Apply(m_V[i], m_wk);
                        h1[i] = this->m_math.ddot(m_w, m_wk);
                        this->m_rowComm->AllReduce(h1[i],
                                                   LibUtilities::ReduceSum);
                        daxpy<ExecSpace>(-h1[i], m_V[i], m_w, m_w);
                    }

                    // Calculate the L2 norm and normalize.
                    this->m_assmbScatrZeroDirOp->Apply(m_w, m_wk);
                    h1[endtem] = this->m_math.ddot(m_w, m_wk);
                    this->m_rowComm->AllReduce(h1[endtem],
                                               LibUtilities::ReduceSum);
                    h1[endtem] = std::sqrt(h1[endtem]);
                }
                else
                {
                    // Reset device memory.
                    auto exchange =
                        m_vExchange.template GetPtr<MemSpace, WriteOnly>();

                    // Classical Gram-Schmidt.
                    for (unsigned int i = starttem; i < endtem; ++i)
                    {
                        ddot<ExecSpace>(m_w, m_V[i], exchange + i);
                    }
                    this->m_rowComm->template AllReduce<MemSpace>(
                        m_vExchange, LibUtilities::ReduceSum);

                    // Device-to-host copy.
                    auto exchangeHost =
                        m_vExchange.template GetPtr<NektarSpaces::HostSpace,
                                                    ReadOnly>();
                    for (unsigned int i = starttem; i < endtem; ++i)
                    {
                        h1[i] = exchangeHost[i];
                        daxpy<ExecSpace>(-h1[i], m_V[i], m_w, m_w);
                    }

                    // Calculate the L2 norm and normalize.
                    h1[endtem] = this->m_math.ddot(m_w, m_w);
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
                if (m_V.size() == ii)
                {
                    m_V.push_back(Field<TData, FieldState::Coeff>(
                        GetBlockAttributes<TData, FieldState::Coeff>(
                            this->m_expansionList),
                        this->m_components, 1));
                    if (m_flexible)
                    {
                        m_Z.push_back(Field<TData, FieldState::Coeff>(
                            GetBlockAttributes<TData, FieldState::Coeff>(
                                this->m_expansionList),
                            this->m_components, 1));
                    }
                }

                // Compute new search vector.
                mul<ExecSpace>((TData)1.0 / h1[endtem], m_w, m_V[ii]);
            }

            // Do backward substitution.
            this->DoBackward(ii, m_upper, eta, yn);

            // Calculate solution delta.
            auto &Z = (m_flexible) ? m_Z : m_V;
            mul<ExecSpace>(yn[0], Z[0], m_w);
            for (unsigned int i = 1; i < ii; ++i)
            {
                // For LGMRES use m_delta for the last m_GMRESDeltaDirection
                // iterations.
                bool cond = i >= (m_LinSysMaxStorage - m_GMRESDeltaDirection) &&
                            outerIterations >= m_GMRESDeltaDirection;
                if (cond)
                {
                    unsigned int index =
                        i - (m_LinSysMaxStorage - m_GMRESDeltaDirection);
                    daxpy<ExecSpace>(yn[i], m_delta[index], m_w, m_w);
                }
                else
                {
                    daxpy<ExecSpace>(yn[i], Z[i], m_w, m_w);
                }
            }

            // Store last m_GMRESDeltaDirection delta for LGMRES.
            if (m_GMRESDeltaDirection)
            {
                auto last = std::move(m_delta.back());
                last.template Copy<MemSpace>(m_w);
                m_delta.pop_back();
                m_delta.push_front(std::move(last));
            }

            // Apply preconditioner.
            if (!m_flexible && this->m_rightPreconditioner)
            {
                this->m_assmbScatrZeroDirOp->Apply(m_w);
                this->m_precon->Apply(m_w, m_w);
            }

            // Update solution.
            add<ExecSpace>(m_w, out, out);

            ii = 0;
            outerIterations++;

            if (converged)
            {
                break;
            }
        }

        // Print output.
        if (m_verbose)
        {
            TData eps_real;

            // Calculate difference in residual of solution.
            this->m_lhs->Apply(out, m_r0);
            this->m_robBndCondOp->Apply(out, m_r0);
            sub<ExecSpace>(in, m_r0, m_r0);
            this->m_assmbScatrZeroDirOp->Apply(m_r0, m_w);
            eps_real = this->m_math.ddot(m_w, m_r0);
            this->m_rowComm->AllReduce(eps_real, LibUtilities::ReduceSum);

            if (this->m_root)
            {
                std::cout << this->name
                          << " iterations made = " << this->m_niter
                          << " using tolerance of " << this->m_tol
                          << " error = "
                          << std::sqrt(eps / eps0 * prec_factor / rhsMagnitude)
                          << " rhs_mag = " << std::sqrt(rhsMagnitude)
                          << " WITH (GMRES eps = " << eps
                          << " REAL eps= " << eps_real << ")";

                if (converged)
                {
                    std::cout << " CONVERGED" << std::endl;
                }
                else
                {
                    std::cout << " WARNING: Exceeded maxIt" << std::endl;
                }
            }
        }

        WARNINGL1(converged, "GMRES did not converge.");
    }
};

} // namespace Nektar::Operators::detail
