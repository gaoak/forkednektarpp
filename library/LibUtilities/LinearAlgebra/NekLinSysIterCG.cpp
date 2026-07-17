///////////////////////////////////////////////////////////////////////////////
//
// File: NekLinSysIterCG.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// License for the specific language governing rights and limitations under
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
// Description: NekLinSysIterCG definition
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Likwid.hpp>
#include <LibUtilities/LinearAlgebra/NekLinSysIterCG.h>
#include <LibUtilities/SimdLib/tinysimd.hpp>

using namespace std;

namespace Nektar::LibUtilities
{
/**
 * @class  NekLinSysIterCG
 *
 * Solves a linear system using iterative methods.
 */
string NekLinSysIterCG::className =
    LibUtilities::GetNekLinSysIterFactory().RegisterCreatorFunction(
        "ConjugateGradient", NekLinSysIterCG::create,
        "NekLinSysIterCG solver.");

NekLinSysIterCG::NekLinSysIterCG(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const LibUtilities::CommSharedPtr &vRowComm, const int nDimen,
    const NekSysKey &pKey)
    : NekLinSysIter(pSession, vRowComm, nDimen, pKey)
{
    m_flexible = pSession->DefinesParameter("FlexibleConjugateGradient")
                     ? bool(pSession->GetParameter("FlexibleConjugateGradient"))
                     : false;
}

void NekLinSysIterCG::v_InitObject()
{
    NekLinSysIter::v_InitObject();
}

/**
 *
 */
int NekLinSysIterCG::v_SolveSystem(const int nGlobal,
                                   const Array<OneD, const NekDouble> &pInput,
                                   Array<OneD, NekDouble> &pOutput,
                                   const int nDir)
{
    std::string tag = "DoConjugateGradient";
    LIKWID_MARKER_START(tag.c_str());
    DoConjugateGradient(nGlobal, pInput, pOutput, nDir);
    LIKWID_MARKER_STOP(tag.c_str());
    return m_totalIterations;
}

void NekLinSysIterCG::v_DoIterate(const int nGlobal,
                                  const Array<OneD, NekDouble> &rhs,
                                  Array<OneD, NekDouble> &x, const int nDir,
                                  NekDouble &err, int &iter)
{
    auto nNonDir = nGlobal - nDir;
    Array<OneD, NekDouble> r(nGlobal, 0.0);
    Array<OneD, NekDouble> dx(nGlobal, 0.0);
    // Get residual r = rhs - Ax
    m_operator.DoNekSysLhsEval(x, r);
    Vmath::Vsub(nNonDir, &rhs[0] + nDir, 1, &r[0] + nDir, 1, &r[0] + nDir, 1);
    // Get dx
    DoConjugateGradient(nGlobal, r, dx, nDir);
    // Update solution x = x + dx
    Vmath::Vadd(nNonDir, &x[0] + nDir, 1, &dx[0] + nDir, 1, &x[0] + nDir, 1);

    iter = m_totalIterations;
    err  = m_finalError;
}

/**  
 * Solve a global linear system using the conjugate gradient method.  
 * We solve only for the non-Dirichlet modes. The operator is evaluated  
 * using an auxiliary function m_operator.DoNekSysLhsEval defined by the  
 * specific solver. Distributed math routines are used to support  
 * parallel execution of the solver.  
 *  
 * The implemented algorithm uses a reduced-communication reordering of  
 * the standard PCG method (Demmel, Heath and Vorst, 1993)  
 *  
 * @param       pInput      Input residual  of all DOFs.  
 * @param       pOutput     Solution vector of all DOFs.  
 */
void NekLinSysIterCG::DoConjugateGradient(
    const int nGlobal, const Array<OneD, const NekDouble> &pInput,
    Array<OneD, NekDouble> &pOutput, const int nDir)
{
    // Get vector sizes
    int nNonDir = nGlobal - nDir;

    // Allocate array storage
    Array<OneD, NekDouble> w_A(nGlobal, 0.0);
    Array<OneD, NekDouble> s_A(nGlobal, 0.0);
    Array<OneD, NekDouble> p_A(nNonDir, 0.0);
    Array<OneD, NekDouble> r_A(nNonDir, 0.0);
    Array<OneD, NekDouble> q_A(nNonDir, 0.0);
    Array<OneD, NekDouble> tmp;

    NekDouble alpha;
    NekDouble beta;
    NekDouble rho;
    NekDouble rho_new;
    NekDouble rho_star;
    NekDouble mu;
    NekDouble eps;
    Array<OneD, NekDouble> vExchange(4, 0.0);

    // Arrays to store CG coefficients for EV estimation routine.
    Array<OneD, NekDouble> alpha_list;
    Array<OneD, NekDouble> beta_list;
    if (m_isComputeEigenvalues)
    {
        alpha_list = Array<OneD, NekDouble>(m_NekLinSysMaxIterations, 0.0);
        beta_list  = Array<OneD, NekDouble>(m_NekLinSysMaxIterations, 0.0);
    }

    // Copy initial residual from input
    Vmath::Vcopy(nNonDir, pInput + nDir, 1, r_A, 1);

    // Zero homogeneous out array ready for solution updates
    // Should not be earlier in case input vector is same as
    // output and above copy has been peformed
    Vmath::Zero(nNonDir, tmp = pOutput + nDir, 1);

    // Evaluate initial residual error for exit check
    vExchange[2] = Vmath::Dot2(nNonDir, r_A, r_A, m_map + nDir);

    m_rowComm->AllReduce(vExchange[2], Nektar::LibUtilities::ReduceSum);

    eps = vExchange[2];

    if (m_rhs_magnitude == NekConstants::kNekUnsetDouble)
    {
        Set_Rhs_Magnitude(pInput, nGlobal);
    }

    m_totalIterations = 0;
    m_eps_old         = sqrt(eps / m_rhs_magnitude);

    // If input residual is less than tolerance skip solve.
    if (eps < m_NekLinSysTolerance * m_NekLinSysTolerance * m_rhs_magnitude)
    {
        m_finalError = sqrt(eps / m_rhs_magnitude);
        if (m_verbose && m_root)
        {
            if (m_flexible)
            {
                cout << "Flexible ";
            }
            cout << "CG iterations made = " << m_totalIterations
                 << " using tolerance of " << m_NekLinSysTolerance
                 << " (error = " << m_finalError
                 << ", rhs_mag = " << sqrt(m_rhs_magnitude) << ")" << endl;
        }
        return;
    }

    m_operator.DoNekSysPrecon(r_A, tmp = w_A + nDir);
    m_operator.DoNekSysLhsEval(w_A, s_A);

    vExchange[0] = Vmath::Dot2(nNonDir, r_A, w_A + nDir, m_map + nDir);
    vExchange[1] = Vmath::Dot2(nNonDir, s_A + nDir, w_A + nDir, m_map + nDir);

    m_rowComm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

    rho_star          = 0.0;
    rho               = vExchange[0];
    mu                = vExchange[1];
    beta              = 0.0;
    alpha             = rho / mu;
    m_totalIterations = 1;

    std::string precon_tag        = "CGPrecondition";
    std::string lhs_tag           = "CGLhsEval";
    std::string before_precon_tag = "CGBeforePrecon";
    std::string after_lhs_tag     = "CGAfterLhsEval";

    // Continue until convergence
    while (true)
    {
        LIKWID_MARKER_START(before_precon_tag.c_str());
        if (m_totalIterations > m_NekLinSysMaxIterations)
        {
            m_finalError = sqrt(eps / m_rhs_magnitude);
            if (m_root)
            {
                if (m_flexible)
                {
                    cout << "Flexible ";
                }
                cout << "CG iterations made = " << m_totalIterations
                     << " using tolerance of " << m_NekLinSysTolerance
                     << " (error = " << m_finalError
                     << ", rhs_mag = " << sqrt(m_rhs_magnitude) << ")"
                     << " WARNING: Exceeded maxIt" << endl;
            }
            break;
        }

        if (m_isComputeEigenvalues)
        {
            // Store the current solution coefficient for EV estimation.
            alpha_list[m_totalIterations - 1] = alpha;
        }

        // Compute new search direction p_k, q_k
        Vmath::Svtvp(nNonDir, beta, &p_A[0], 1, &w_A[nDir], 1, &p_A[0], 1);
        Vmath::Svtvp(nNonDir, beta, &q_A[0], 1, &s_A[nDir], 1, &q_A[0], 1);

        // Update solution x_{k+1}
        Vmath::Svtvp(nNonDir, alpha, &p_A[0], 1, &pOutput[nDir], 1,
                     &pOutput[nDir], 1);

        // Update residual vector r_{k+1}
        Vmath::Svtvp(nNonDir, -alpha, &q_A[0], 1, &r_A[0], 1, &r_A[0], 1);

        if (m_flexible)
        {
            if (m_mapIsOnes)
            {
                // <r_{k+1}, w_{k}>
                vExchange[3] = Vmath::Dot(nNonDir, r_A, w_A + nDir);
            }
            else
            {
                // <r_{k+1}, w_{k}>
                vExchange[3] =
                    Vmath::Dot2(nNonDir, r_A, w_A + nDir, m_map + nDir);
            }
        }

        LIKWID_MARKER_STOP(before_precon_tag.c_str());

        LIKWID_MARKER_START(precon_tag.c_str());
        // Apply preconditioner
        m_operator.DoNekSysPrecon(r_A, tmp = w_A + nDir);
        LIKWID_MARKER_STOP(precon_tag.c_str());

        LIKWID_MARKER_START(lhs_tag.c_str());
        // Perform the method-specific matrix-vector multiply operation.
        m_operator.DoNekSysLhsEval(w_A, s_A);
        LIKWID_MARKER_STOP(lhs_tag.c_str());

        LIKWID_MARKER_START(after_lhs_tag.c_str());

        if (m_mapIsOnes)
        {
            // <r_{k+1}, w_{k+1}>
            vExchange[0] = Vmath::Dot(nNonDir, r_A, w_A + nDir);

            // <s_{k+1}, w_{k+1}>
            vExchange[1] = Vmath::Dot(nNonDir, s_A + nDir, w_A + nDir);

            if (m_totalIterations % m_errorCheckInterval == 0)
            {
                // <r_{k+1}, r_{k+1}>
                vExchange[2] = Vmath::Dot(nNonDir, r_A, r_A);
            }
        }
        else
        {
            // <r_{k+1}, w_{k+1}>
            vExchange[0] = Vmath::Dot2(nNonDir, r_A, w_A + nDir, m_map + nDir);

            // <s_{k+1}, w_{k+1}>
            vExchange[1] =
                Vmath::Dot2(nNonDir, s_A + nDir, w_A + nDir, m_map + nDir);

            if (m_totalIterations % m_errorCheckInterval == 0)
            {
                // <r_{k+1}, r_{k+1}>
                vExchange[2] = Vmath::Dot2(nNonDir, r_A, r_A, m_map + nDir);
            }
        }
        // Perform inner-product exchanges
        m_rowComm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        rho_new = vExchange[0];
        mu      = vExchange[1];
        eps     = vExchange[2];
        if (m_flexible)
        {
            rho_star = vExchange[3];
        }

        m_totalIterations++;

        // print iteration progress
        if (m_verbose && m_root)
        {
            NekDouble rel_err = sqrt(eps / m_rhs_magnitude);
            if (log(m_eps_old / rel_err) > log(10.0) * m_printThreshold)
            {
                if (m_flexible)
                {
                    cout << "Flexible ";
                }
                cout << "CG iteration = " << m_totalIterations
                     << ", error = " << rel_err
                     << ", rhs_mag = " << sqrt(m_rhs_magnitude) << endl;
                m_eps_old = rel_err;
            }
        }

        // Test if norm is within tolerance
        if (eps < m_NekLinSysTolerance * m_NekLinSysTolerance * m_rhs_magnitude)
        {
            m_finalError = sqrt(eps / m_rhs_magnitude);
            if (m_verbose && m_root)
            {
                if (m_flexible)
                {
                    cout << "Flexible ";
                }
                cout << "CG iterations made = " << m_totalIterations
                     << " using tolerance of " << m_NekLinSysTolerance
                     << " (error = " << m_finalError
                     << ", rhs_mag = " << sqrt(m_rhs_magnitude) << ")" << endl;
            }
            break;
        }

        // Compute search direction and solution coefficients
        beta = (rho_new - rho_star) / rho;
        if (m_isComputeEigenvalues)
        {
            beta_list[m_totalIterations - 2] = beta;
        }
        alpha = rho_new / (mu - rho_new * beta / alpha);
        rho   = rho_new;
        LIKWID_MARKER_STOP(after_lhs_tag.c_str());
    }

    if (m_isComputeEigenvalues)
    {
        ComputeEigenvalues(m_totalIterations - 1, alpha_list, beta_list);
    }
}

// Function to get EVs for a tridiagonal matrix using Lapack::Dsterf
// Input: Take list of alphas and betas from CG routine
// Step1: Create the TriDiag matrix using alpha, beta list
// Step2: Use the Lapack::Dsterf routine for calculating EVs
void NekLinSysIterCG::ComputeEigenvalues(
    const int length, const Array<OneD, NekDouble> &alpha_list,
    const Array<OneD, NekDouble> &beta_list)
{
    // Rank of the matrix for EV estimation
    m_eigenvalues = Array<OneD, NekDouble>(length, 0.0);
    Array<OneD, NekDouble> mainDiag(m_eigenvalues);
    Array<OneD, NekDouble> subDiag(length - 1, 0.0);

    // Create the Tridiagonal matrix neeeded for EV estimation
    // Fill subDiag,mainDiag with TriDiag information
    for (int i = 0; i < length; ++i)
    {
        // filling Diagonal information
        if (i == 0) // does not have the second term
        {
            // Handle the special case when i = 0
            mainDiag[i] = 1 / alpha_list[i];
        }
        else
        {
            // Perform element-wise division for i > 0
            mainDiag[i] =
                1.0 / alpha_list[i] + beta_list[i - 1] / alpha_list[i - 1];
        }
        // filling subDiagonal information, same on both sides
        if (i < length - 1) // Stop on the second last iteration
        {
            subDiag[i] = std::sqrt(beta_list[i]) / alpha_list[i];
        }
    }

    // use LAPACK:dsterf for calculating the EVs for TriDiag matrix
    // solves for symmetric TriDiag matrix
    int info = 0;
    Lapack::Dsterf(length, mainDiag.data(), subDiag.data(), info);

    if (m_root && m_verbose)
    {
        cout << "\nEigenvalues: " << endl;
        // print the last 10 values, which includes the max eigenv
        int m = min(length, 10);
        for (int j = 0; j < m; j++)
        {
            cout << mainDiag[length - 1 - j] << ' ';
        }
        cout << endl;
    }

    if (m_isOutputEigenvalues)
    {
        FILE *pFile;
        int myRank           = m_rowComm->GetRank();
        std::string filename = std::string("Eigenvalues_") +
                               std::to_string(myRank) + std::string(".txt");
        pFile = fopen(filename.c_str(), "w");
        for (int j = 0; j < length; j++)
        {
            fprintf(pFile, "%e\n", mainDiag[j]);
        }
        fclose(pFile);
    }
}

} // namespace Nektar::LibUtilities
