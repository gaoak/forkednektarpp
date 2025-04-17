///////////////////////////////////////////////////////////////////////////////
//
// File: GMRESImpl.hpp
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

#include <MultiRegions/ContField.h>

#include "Operators/GlobalLinSysOps/GMRES/OperatorGMRES.hpp"

#include "Operators/AssmbScatr/OperatorAssmbScatr.hpp"
#include "Operators/BndCondOps/RobBndCond/OperatorRobBndCond.hpp"
#include "Operators/MathKernels/MathKernels.hpp"

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <memory>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class OperatorGMRESImpl : public OperatorGMRES<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorGMRESImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorGMRES<TData>(expansionList),
          m_w(Field<TData, FieldState::Coeff>::Create(
              "GMRES w",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              ExecSpace::alignment, eDeviceOnly)),
          m_wk(Field<TData, FieldState::Coeff>::Create(
              "GMRES wk",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              ExecSpace::alignment, eDeviceOnly)),
          m_r0(Field<TData, FieldState::Coeff>::Create(
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              ExecSpace::alignment, eDeviceOnly)),
          m_solution(Field<TData, FieldState::Coeff>::Create(
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              ExecSpace::alignment, eDeviceOnly))
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);

        // Set parameters.
        contfield->GetSession()->LoadParameter("NekLinSysMaxIterations",
                                               m_NekLinSysMaxIterations, 5000);
        contfield->GetSession()->LoadParameter("LinSysMaxStorage",
                                               m_LinSysMaxStorage, 5000);
        contfield->GetSession()->LoadParameter("IterativeSolverTolerance",
                                               m_tol, 1.0E-09);
        contfield->GetSession()->LoadParameter("GMRESMaxHessMatBand",
                                               m_KrylovMaxHessMatBand,
                                               m_LinSysMaxStorage + 1);
        contfield->GetSession()->MatchSolverInfo("GMRESLeftPrecon", "True",
                                                 m_NekLinSysLeftPrecon, false);
        contfield->GetSession()->MatchSolverInfo("GMRESRightPrecon", "True",
                                                 m_NekLinSysRightPrecon, true);
        contfield->GetSession()->MatchSolverInfo(
            "GMRESCentralDifference", "True", m_GMRESCentralDifference, false);

        // Set operators.
        m_assmbScatrOp = OperatorAssmbScatr<TData>::Create(
            this->m_expansionList, ExecSpace::name);
        m_robBndCondOp = OperatorRobBndCond<TData>::Create(
            this->m_expansionList, ExecSpace::name);
        m_rowComm = contfield->GetSession()->GetComm()->GetRowComm();

        // Allocate array storage.
        m_hes   = Array<OneD, Array<OneD, TData>>(m_LinSysMaxStorage);
        m_upper = Array<OneD, Array<OneD, TData>>(m_LinSysMaxStorage);
        for (size_t nd = 0; nd < m_LinSysMaxStorage; nd++)
        {
            m_hes[nd]   = Array<OneD, TData>(m_LinSysMaxStorage + 1, 0.0);
            m_upper[nd] = Array<OneD, TData>(m_LinSysMaxStorage + 1, 0.0);
        }

        // Restarted Gmres(m) process.
        if (m_NekLinSysRightPrecon)
        {
            m_V1 = Field<TData, FieldState::Coeff>::Create(
                GetBlockAttributes<TData>(FieldState::Coeff,
                                          this->m_expansionList),
                1, ExecSpace::alignment, eDeviceOnly);
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Initialize precond factor.
        m_prec_factor = NekConstants::kNekUnsetDouble;

        // Calculate rhs magnitude.
        if (m_rhs_magnitude == NekConstants::kNekUnsetDouble)
        {
            m_assmbScatrOp->apply(in, m_wk);
            ddot<ExecSpace>(in, m_wk, &m_rhs_magnitude);
            m_rowComm->AllReduce(m_rhs_magnitude,
                                 Nektar::LibUtilities::ReduceSum);
            m_rhs_magnitude =
                (m_rhs_magnitude > 1.0e-6) ? m_rhs_magnitude : 1.0;
        }

        // GMRES with restart.
        m_totalIterations    = 0;
        bool converged       = false;
        TData eps            = 0.0;
        const bool truncted  = (m_KrylovMaxHessMatBand > 0);
        const int maxrestart = m_NekLinSysMaxIterations / m_LinSysMaxStorage;
        for (int nrestart = 0; nrestart < maxrestart; ++nrestart)
        {
            const bool restart = (nrestart > 0);
            auto conv          = DoGmresRestart(restart, truncted, in, out);
            eps                = conv.first;
            converged          = conv.second;

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
            this->m_lhs->apply(out, m_r0);
            m_robBndCondOp->apply(out, m_r0);
            sub<ExecSpace>(in, m_r0, m_r0);
            m_assmbScatrOp->apply(m_r0, m_wk, true);
            ddot<ExecSpace>(m_wk, m_r0, &eps1);
            m_rowComm->AllReduce(eps1, LibUtilities::ReduceSum);

            // if (m_root)
            {
                int nwidthcolm = 13;

                std::cout << std::scientific << std::setw(nwidthcolm)
                          << std::setprecision(nwidthcolm - 8)
                          << "       GMRES iterations made = "
                          << m_totalIterations << " using tolerance of "
                          << m_tol << " (error = "
                          << sqrt(eps * m_prec_factor / m_rhs_magnitude) << ")";

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

    std::pair<TData, bool> DoGmresRestart(const bool restarted,
                                          const bool truncted,
                                          Field<TData, FieldState::Coeff> &in,
                                          Field<TData, FieldState::Coeff> &out)
    {
        // Allocate array storage.
        // Residual
        Array<OneD, TData> eta(m_LinSysMaxStorage + 1, 0.0);
        // Givens rotation c
        Array<OneD, TData> cs(m_LinSysMaxStorage, 0.0);
        // Givens rotation s
        Array<OneD, TData> sn(m_LinSysMaxStorage, 0.0);
        // Total coefficients, just for check
        Array<OneD, TData> y_total(m_LinSysMaxStorage, 0.0);
        // Search direction order
        Array<OneD, int> id(m_LinSysMaxStorage, 0);
        Array<OneD, int> id_start(m_LinSysMaxStorage, 0);
        Array<OneD, int> id_end(m_LinSysMaxStorage, 0);

        // Set the fields to zero.
        out.template Initialize<MemSpace>(0);

        if (restarted)
        {
            // This is A*x
            this->m_lhs->apply(out, m_r0);
            m_robBndCondOp->apply(out, m_r0);

            // This is r0 = b-A*x
            sub<ExecSpace>(in, m_r0, m_r0);
        }
        else
        {
            // If not restarted, x0 should be zero
            m_r0.template Copy<MemSpace>(in);
        }

        // Apply preconditioner.
        if (m_NekLinSysLeftPrecon)
        {
            this->m_precon->apply(m_r0, m_r0);
        }

        // Norm of (r0)
        TData eps;
        m_assmbScatrOp->apply(m_r0, m_wk, true);
        ddot<ExecSpace>(m_r0, m_wk, &eps);
        m_rowComm->AllReduce(eps, LibUtilities::ReduceSum);

        if (!restarted)
        {
            if (m_prec_factor == NekConstants::kNekUnsetDouble)
            {
                if (m_NekLinSysLeftPrecon)
                {
                    m_assmbScatrOp->apply(in, m_wk, true);
                    ddot<ExecSpace>(in, m_wk, &m_prec_factor);
                    m_rowComm->AllReduce(m_prec_factor,
                                         LibUtilities::ReduceSum);
                }

                m_prec_factor =
                    m_NekLinSysLeftPrecon ? m_prec_factor / eps : 1.0;
            }
        }

        mul<ExecSpace>(sqrt(m_prec_factor), m_r0, m_r0);
        eps *= m_prec_factor;
        eta[0] = sqrt(eps);

        // Give an order for the entries in Hessenburg matrix.
        for (int nd = 0; nd < m_LinSysMaxStorage; ++nd)
        {
            id[nd]       = nd;
            id_end[nd]   = nd + 1;
            int starttem = id_end[nd] - m_KrylovMaxHessMatBand;
            if (truncted && starttem > 0)
            {
                id_start[nd] = starttem;
            }
            else
            {
                id_start[nd] = 0;
            }
        }

        // Scalar multiplication.
        if (m_Vtotal.size() == 0)
        {
            m_Vtotal.push_back(Field<TData, FieldState::Coeff>::Create(
                GetBlockAttributes<TData>(FieldState::Coeff,
                                          this->m_expansionList),
                1, ExecSpace::alignment, eDeviceOnly));
        }
        mul<ExecSpace>(1.0 / eta[0], m_r0, m_Vtotal[0]);

        // Restarted Gmres(m) process.
        if (m_NekLinSysRightPrecon)
        {
            m_V1.template Initialize<MemSpace>(0.0);
        }

        bool converged = false;
        int nswp       = 0;
        for (int nd = 0; nd < m_LinSysMaxStorage; ++nd)
        {
            if (m_Vtotal.size() == nd + 1)
            {
                m_Vtotal.push_back(Field<TData, FieldState::Coeff>::Create(
                    GetBlockAttributes<TData>(FieldState::Coeff,
                                              this->m_expansionList),
                    1, ExecSpace::alignment, eDeviceOnly));
            }
            m_Vtotal[nd + 1].template Initialize<MemSpace>(0);
            Vmath::Zero(m_LinSysMaxStorage + 1, m_hes[nd], 1);
            auto &V1 = m_NekLinSysRightPrecon ? m_V1 : m_Vtotal[nd];
            auto &V2 = m_Vtotal[nd + 1];
            auto &h1 = m_hes[nd];
            auto &h2 = m_upper[nd];

            // Apply preconditioner.
            if (m_NekLinSysRightPrecon)
            {
                this->m_precon->apply(m_Vtotal[nd], V1);
            }

            auto idtem    = id[nd];
            auto starttem = id_start[idtem];
            auto endtem   = id_end[idtem];

            DoArnoldi(starttem, endtem, m_w, m_wk, V1, V2, h1);

            if (starttem > 0)
            {
                starttem = starttem - 1;
            }

            Vmath::Vcopy(m_LinSysMaxStorage + 1, h1.data(), 1, h2.data(), 1);
            DoGivensRotation(starttem, endtem, cs, sn, h2, eta);

            eps = eta[nd + 1] * eta[nd + 1];

            nswp++;
            m_totalIterations++;

            // This Gmres merge truncted Gmres to accelerate.
            // If truncted, cannot jump out because
            // the last term of eta is not residual
            if ((!truncted) || (nd < m_KrylovMaxHessMatBand))
            {
                if (eps < m_tol * m_tol * m_rhs_magnitude) //&& nd > 0)
                {
                    converged = true;
                    break;
                }
            }
        }

        // Do backward substitution.
        DoBackward(nswp, m_upper, eta, y_total);

        // Calculate output y_total*V_total.
        m_solution.template Initialize<MemSpace>(0);
        for (int i = 0; i < nswp; ++i)
        {
            daxpy<ExecSpace>(y_total[i], m_Vtotal[i], m_solution, m_solution);
        }

        // Apply preconditioner.
        if (m_NekLinSysRightPrecon)
        {
            this->m_precon->apply(m_solution, m_solution);
        }

        // Update output.
        add<ExecSpace>(m_solution, out, out);

        return {eps, converged};
    }

    // Arnoldi Subroutine
    void DoArnoldi(const int starttem, const int endtem,
                   Field<TData, FieldState::Coeff> &w,
                   Field<TData, FieldState::Coeff> &wk,
                   Field<TData, FieldState::Coeff> &V1,
                   Field<TData, FieldState::Coeff> &V2, Array<OneD, TData> &h)
    {
        // Apply lhs.
        this->m_lhs->apply(V1, w);
        m_robBndCondOp->apply(V1, w);

        // Apply preconditioner.
        if (m_NekLinSysLeftPrecon)
        {
            this->m_precon->apply(w, w);
        }

        mul<ExecSpace>(std::sqrt(m_prec_factor), w, w);

        // Modified Gram-Schmidt.
        for (int i = starttem; i < endtem; ++i)
        {
            m_assmbScatrOp->apply(m_Vtotal[i], wk, true);
            ddot<ExecSpace>(w, wk, &h[i]);
            m_rowComm->AllReduce(h[i], LibUtilities::ReduceSum);
            daxpy<ExecSpace>(-1.0 * h[i], m_Vtotal[i], w, w);
        }

        // Calculate the L2 norm and normalize.
        m_assmbScatrOp->apply(w, wk, true);
        ddot<ExecSpace>(w, wk, &h[endtem]);
        m_rowComm->AllReduce(h[endtem], LibUtilities::ReduceSum);
        h[endtem] = std::sqrt(h[endtem]);
        mul<ExecSpace>(1.0 / h[endtem], w, V2);
    }

    // QR factorization through Givens rotation -> Put into a helper class
    void DoGivensRotation(const int starttem, const int endtem,
                          Array<OneD, TData> &c, Array<OneD, TData> &s,
                          Array<OneD, TData> &h, Array<OneD, TData> &eta)
    {
        TData temp_dbl;
        TData dd;
        TData hh;
        int idtem = endtem - 1;
        // The starttem and endtem are beginning and ending order of Givens
        // rotation They usually equal to the beginning position and ending
        // position of Hessenburg matrix But sometimes starttem will change,
        // like if it is initial 0 and becomes nonzero because previous Givens
        // rotation See Yu Pan's User Guide
        for (int i = starttem; i < idtem; ++i)
        {
            temp_dbl = c[i] * h[i] - s[i] * h[i + 1];
            h[i + 1] = s[i] * h[i] + c[i] * h[i + 1];
            h[i]     = temp_dbl;
        }
        dd = h[idtem];
        hh = h[endtem];
        if (hh == 0.0)
        {
            c[idtem] = 1.0;
            s[idtem] = 0.0;
        }
        else if (abs(hh) > abs(dd))
        {
            temp_dbl = -dd / hh;
            s[idtem] = 1.0 / sqrt(1.0 + temp_dbl * temp_dbl);
            c[idtem] = temp_dbl * s[idtem];
        }
        else
        {
            temp_dbl = -hh / dd;
            c[idtem] = 1.0 / sqrt(1.0 + temp_dbl * temp_dbl);
            s[idtem] = temp_dbl * c[idtem];
        }

        h[idtem]  = c[idtem] * h[idtem] - s[idtem] * h[endtem];
        h[endtem] = 0.0;

        temp_dbl    = c[idtem] * eta[idtem] - s[idtem] * eta[endtem];
        eta[endtem] = s[idtem] * eta[idtem] + c[idtem] * eta[endtem];
        eta[idtem]  = temp_dbl;
    }

    void DoBackward(const int n, Array<OneD, Array<OneD, TData>> &A,
                    const Array<OneD, const TData> &b, Array<OneD, TData> &y)
    {
        TData sum;
        y[n - 1] = b[n - 1] / A[n - 1][n - 1];
        for (int i = n - 2; i > -1; --i)
        {
            sum = b[i];
            for (int j = i + 1; j < n; ++j)
            {
                sum -= y[j] * A[j][i];
            }
            y[i] = sum / A[i][i];
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorGMRESImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    LibUtilities::CommSharedPtr m_rowComm = nullptr;

    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatrOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_robBndCondOp;

    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_wk;
    Field<TData, FieldState::Coeff> m_r0;
    Field<TData, FieldState::Coeff> m_solution;
    Field<TData, FieldState::Coeff> m_V1;
    std::vector<Field<TData, FieldState::Coeff>> m_Vtotal;
    Array<OneD, Array<OneD, TData>> m_hes;
    Array<OneD, Array<OneD, TData>> m_upper;

    TData m_rhs_magnitude = NekConstants::kNekUnsetDouble;
    TData m_prec_factor;
    TData m_tol;
    bool m_NekLinSysLeftPrecon;
    bool m_NekLinSysRightPrecon;
    bool m_GMRESCentralDifference;
    size_t m_totalIterations;
    size_t m_NekLinSysMaxIterations;
    size_t m_LinSysMaxStorage;
    size_t m_KrylovMaxHessMatBand;
};

} // namespace Nektar::Operators::detail
