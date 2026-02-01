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

        auto session = expansionList->GetSession();
        session->LoadParameter("LinSysMaxStorage", m_LinSysMaxStorage, 50);
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

        // Allocate array storage.
        if (!m_isModifiedGramSchmidt)
        {
            m_vExchange = MemoryRegion<TData>(m_LinSysMaxStorage, ePinned);
        }
        m_hes   = std::vector<std::vector<TData>>(m_LinSysMaxStorage);
        m_upper = std::vector<std::vector<TData>>(m_LinSysMaxStorage);
        for (unsigned int nd = 0; nd < m_LinSysMaxStorage; nd++)
        {
            m_hes[nd]   = std::vector<TData>(m_LinSysMaxStorage + 1, 0.0);
            m_upper[nd] = std::vector<TData>(m_LinSysMaxStorage + 1, 0.0);
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
    std::vector<std::vector<TData>> m_hes;
    std::vector<std::vector<TData>> m_upper;

    MemoryRegion<TData> m_vExchange;

    TData m_rhs_magnitude = NekConstants::kNekUnsetDouble;
    bool m_flexible;
    bool m_isModifiedGramSchmidt = true;
    bool m_GMRESCentralDifference;
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
        std::vector<unsigned int> id(m_LinSysMaxStorage);
        std::vector<unsigned int> id_start(m_LinSysMaxStorage);
        std::vector<unsigned int> id_end(m_LinSysMaxStorage);
        const bool truncted          = (m_KrylovMaxHessMatBand > 0);
        unsigned int totalIterations = 0, ii = 0, outerIterations = 0;
        bool converged    = false;
        TData prec_factor = 1.0, eps, eps0 = 1.0;

        // Give an order for the entries in Hessenburg matrix.
        for (unsigned int nd = 0; nd < m_LinSysMaxStorage; ++nd)
        {
            id[nd]     = nd;
            id_end[nd] = nd + 1;
            if (truncted && id_end[nd] > m_KrylovMaxHessMatBand)
            {
                id_start[nd] = id_end[nd] - m_KrylovMaxHessMatBand;
            }
            else
            {
                id_start[nd] = 0;
            }
        }

        // Calculate rhs magnitude.
        if (m_rhs_magnitude == NekConstants::kNekUnsetDouble)
        {
            this->m_assmbScatrOp->Apply(in, m_w);
            m_rhs_magnitude = this->m_math.ddot(in, m_w);
            this->m_rowComm->AllReduce(m_rhs_magnitude,
                                       Nektar::LibUtilities::ReduceSum);
            m_rhs_magnitude =
                (m_rhs_magnitude > 1.0e-6) ? m_rhs_magnitude : 1.0;
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
            if (totalIterations == this->m_maxIter)
            {
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
            if (eps < this->m_tol * this->m_tol * m_rhs_magnitude)
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
                    (totalIterations == this->m_maxIter))
                {
                    break;
                }

                unsigned int znd = m_flexible ? ii : 0;
                auto &Z1 = this->m_rightPreconditioner ? m_Z[znd] : m_V[ii];
                auto &V1 = m_V[ii];
                auto &h1 = m_hes[ii];
                auto &h2 = m_upper[ii];

                // Apply preconditioner.
                if (this->m_rightPreconditioner)
                {
                    this->m_assmbScatrZeroDirOp->Apply(V1, Z1);
                    this->m_precon->Apply(Z1, Z1);
                }

                auto idtem    = id[ii];
                auto starttem = id_start[idtem];
                auto endtem   = id_end[idtem];

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
                DoGivensRotation(starttem, endtem, cs, sn, h2, eta);

                eps = eta[ii + 1] * eta[ii + 1];

                ii++;
                totalIterations++;

                // This Gmres merge truncted Gmres to accelerate.
                // If truncted, cannot jump out because
                // the last term of eta is not residual
                if ((!truncted) || (ii <= m_KrylovMaxHessMatBand))
                {
                    if (eps < this->m_tol * this->m_tol * m_rhs_magnitude)
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
            DoBackward(ii, m_upper, eta, yn);

            if (m_flexible)
            {
                // Calculate output yn*m_Z.
                for (unsigned int i = 0; i < ii; ++i)
                {
                    daxpy<ExecSpace>(yn[i], m_Z[i], out, out);
                }
            }
            else
            {
                // Calculate output yn*m_V.
                mul<ExecSpace>(yn[0], m_V[0], m_w);
                for (unsigned int i = 1; i < ii; ++i)
                {
                    daxpy<ExecSpace>(yn[i], m_V[i], m_w, m_w);
                }

                // Apply preconditioner.
                if (this->m_rightPreconditioner)
                {
                    this->m_assmbScatrZeroDirOp->Apply(m_w);
                    this->m_precon->Apply(m_w, m_w);
                }

                // Update output.
                add<ExecSpace>(m_w, out, out);
            }

            ii = 0;
            outerIterations++;

            if (converged)
            {
                break;
            }
        }

        // Print output.
        // if (m_verbose)
        {
            TData eps1;

            // Calculate difference in residual of solution.
            this->m_lhs->Apply(out, m_r0);
            this->m_robBndCondOp->Apply(out, m_r0);
            sub<ExecSpace>(in, m_r0, m_r0);
            this->m_assmbScatrZeroDirOp->Apply(m_r0, m_w);
            eps1 = this->m_math.ddot(m_w, m_r0);
            this->m_rowComm->AllReduce(eps1, LibUtilities::ReduceSum);

            if (this->m_root)
            {
                int nwidthcolm = 13;

                std::cout << std::scientific << std::setw(nwidthcolm)
                          << std::setprecision(nwidthcolm - 8)
                          << "       GMRES iterations made = "
                          << totalIterations << " using tolerance of "
                          << this->m_tol << " (error = "
                          << std::sqrt(eps / eps0 * prec_factor /
                                       m_rhs_magnitude)
                          << ")";

                std::cout << " WITH (GMRES eps = " << eps
                          << " REAL eps= " << eps1 << ")";

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

    // QR factorization through Givens rotation -> Put into a helper class
    void DoGivensRotation(const unsigned int starttem,
                          const unsigned int endtem, std::vector<TData> &c,
                          std::vector<TData> &s, std::vector<TData> &h,
                          std::vector<TData> &eta)
    {
        TData dbl;
        TData dd;
        TData hh;
        unsigned int idtem = endtem - 1;

        // The starttem and endtem are beginning and ending order of Givens
        // rotation They usually equal to the beginning position and ending
        // position of Hessenburg matrix But sometimes starttem will change,
        // like if it is initial 0 and becomes nonzero because previous Givens
        // rotation See Yu Pan's User Guide
        for (unsigned int i = starttem; i < idtem; ++i)
        {
            dbl      = c[i] * h[i] - s[i] * h[i + 1];
            h[i + 1] = s[i] * h[i] + c[i] * h[i + 1];
            h[i]     = dbl;
        }
        dd = h[idtem];
        hh = h[endtem];
        if (hh == 0.0)
        {
            c[idtem] = 1.0;
            s[idtem] = 0.0;
        }
        else if (std::abs(hh) > std::abs(dd))
        {
            dbl      = -dd / hh;
            s[idtem] = 1.0 / std::sqrt(1.0 + dbl * dbl);
            c[idtem] = dbl * s[idtem];
        }
        else
        {
            dbl      = -hh / dd;
            c[idtem] = 1.0 / std::sqrt(1.0 + dbl * dbl);
            s[idtem] = dbl * c[idtem];
        }

        h[idtem]  = c[idtem] * h[idtem] - s[idtem] * h[endtem];
        h[endtem] = 0.0;

        dbl         = c[idtem] * eta[idtem] - s[idtem] * eta[endtem];
        eta[endtem] = s[idtem] * eta[idtem] + c[idtem] * eta[endtem];
        eta[idtem]  = dbl;
    }

    void DoBackward(const unsigned int n,
                    const std::vector<std::vector<TData>> &A,
                    const std::vector<TData> &b, std::vector<TData> &y)
    {
        TData sum;
        y[n - 1] = b[n - 1] / A[n - 1][n - 1];
        for (unsigned int i = n - 2; i + 1 > 0; --i)
        {
            sum = b[i];
            for (unsigned int j = i + 1; j < n; ++j)
            {
                sum -= y[j] * A[j][i];
            }
            y[i] = sum / A[i][i];
        }
    }
};

} // namespace Nektar::Operators::detail
