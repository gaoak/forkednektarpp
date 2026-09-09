///////////////////////////////////////////////////////////////////////////////
//
// File: PerforatedPlate.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2026 Nicolas de Jong
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
// Description: PerforatedPlate
//
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <AcousticSolver/LinerSolvers/PerforatedPlate.h>

#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <LibUtilities/TimeIntegration/TimeIntegrationSchemeOperators.h>

namespace Nektar
{

PerforatedPlate::PerforatedPlate(const int linerType, NekDouble w0,
                                 NekDouble wI, NekDouble u0, NekDouble uI)
    : m_modelType(linerType), m_w0(w0), m_wI(wI), m_u0(u0), m_uI(uI)
{
}

/**
 * @brief Initialization object for the AcousticSystem class.
 */
void PerforatedPlate::InitObject(Array<OneD, NekDouble> &A,
                                 Array<OneD, NekDouble> &B, short sizeF,
                                 int nBC, Array<OneD, int> &nPnt, int nExp)
{
    (void)m_modelType; // unused, for future development

    // Format rational fraction coefficients
    m_A = NekVector<NekDouble>(sizeF, 0.0);
    m_B = NekVector<NekDouble>(sizeF, 0.0);
    for (short i = 0; i < sizeF; ++i)
    {
        m_A[i] = A[i];
        m_B[i] = B[i];
    }
    if (m_A[sizeF - 1] != 1.0)
    {
        Vmath::Smul(sizeF, 1.0 / m_A[sizeF - 1], &m_B[0], 1, &m_B[0], 1);
        Vmath::Smul(sizeF, 1.0 / m_A[sizeF - 1], &m_A[0], 1, &m_A[0], 1);
    }

    // Liner initial state computation
    NekDouble iV_real = m_A[0], iV_imag = 0.0, iV_mod = 0.0, lastImag = 0.0;
    for (short k = 2; k < m_A.GetDimension(); k = k + 2)
    {
        iV_real = iV_real + m_A[k] * std::pow(m_w0, k) * std::pow(-1.0, k / 2);
        iV_imag = iV_imag + m_A[k] * std::pow(m_wI, k) * std::pow(-1.0, k / 2);
    }
    for (short k = 1; k < m_A.GetDimension(); k = k + 2)
    {
        iV_imag =
            iV_imag + m_A[k] * std::pow(m_w0, k) * std::pow(-1.0, (k - 1) / 2);
        iV_real =
            iV_real + m_A[k] * std::pow(m_wI, k) * std::pow(-1.0, (k + 1) / 2);
    }
    iV_mod  = 1.0 / (std::pow(iV_real, 2.0) + std::pow(iV_imag, 2.0));
    iV_real = iV_real * iV_mod;
    iV_imag = -iV_imag * iV_mod;
    if (m_u0 != 0.0)
    {
        iV_real = iV_real * m_u0 - iV_imag * m_uI;
        iV_imag = iV_real * m_uI + iV_imag * m_u0;
    }

    // ODE Parameter initialization
    m_U = Array<OneD, Array<OneD, Array<OneD, Array<OneD, NekDouble>>>>(nBC);
    m_y = Array<OneD, Array<OneD, Array<OneD, Array<OneD, NekDouble>>>>(nBC);
    for (short i = 0; i < nBC; ++i)
    {
        m_U[i] = Array<OneD, Array<OneD, Array<OneD, NekDouble>>>(nPnt[i]);
        m_y[i] = Array<OneD, Array<OneD, Array<OneD, NekDouble>>>(nPnt[i]);
        for (short j = 0; j < nPnt[i]; ++j)
        {
            m_U[i][j] = Array<OneD, Array<OneD, NekDouble>>(m_A.GetDimension());
            m_U[i][j][0] = Array<OneD, NekDouble>(nExp, 0.0);
            m_U[i][j][1] = Array<OneD, NekDouble>(nExp, iV_real);
            m_y[i][j] = Array<OneD, Array<OneD, NekDouble>>(m_A.GetDimension());
            m_y[i][j][0] = Array<OneD, NekDouble>(nExp, 0.0);
            m_y[i][j][1] = Array<OneD, NekDouble>(nExp, 0.0);
            lastImag     = iV_imag;
            for (short k = 2; k < m_A.GetDimension(); ++k)
            {
                m_U[i][j][k] = Array<OneD, NekDouble>(
                    nExp, -m_U[i][j][k - 1][0] * m_wI - lastImag * m_w0);
                lastImag     = m_U[i][j][k - 1][0] * m_w0 - lastImag * m_wI;
                m_y[i][j][k] = Array<OneD, NekDouble>(nExp, 0.0);
            }
        }
    }
    m_currentTime = 0.0;

    // ODE intialization
    m_timeScheme =
        LibUtilities::GetTimeIntegrationSchemeFactory().CreateInstance(
            "RungeKutta", "", 4, {});
    m_ODE.DefineOdeRhs(&PerforatedPlate::LinerOperator, this);
    m_ODE.DefineProjection(&PerforatedPlate::LinerProjection, this);
}

/**
 * @brief Function to compute the pressure jump across the perforated plate for
 * a given value of the acoustic velocity.
 */
Array<OneD, NekDouble> PerforatedPlate::LinerOutput(const int bound,
                                                    const int point,
                                                    const NekDouble *u,
                                                    const NekDouble intTime,
                                                    const NekDouble simdt)
{
    // Initialize ODE parameters
    Vmath::Vcopy(m_U[bound][point][0].size(), u, 1, &m_U[bound][point][0][0],
                 1);
    if (intTime == 0.0 && m_u0 == 0.0)
    {
        m_dt = 0.0;
        for (short i = 1; i < m_A.GetDimension(); ++i)
        {
            Vmath::Vmul(m_U[bound][point][i].size(), u, 1,
                        &m_U[bound][point][i][0], 1, &m_U[bound][point][i][0],
                        1);
        }
    }

    // Compute time step and update base state
    if (m_lastTime < intTime)
    {
        m_dt = intTime - m_currentTime;
        if (m_dt > simdt)
        {
            m_currentTime = m_lastTime;
            m_dt          = intTime - m_lastTime;
            for (int j = 0; j < m_U.size(); ++j)
            {
                for (int k = 0; k < m_U[j].size(); ++k)
                {
                    for (int i = 1; i < m_y[j][k].size(); ++i)
                    {
                        Vmath::Vcopy(m_U[j][k][i].size(), &m_y[j][k][i][0], 1,
                                     &m_U[j][k][i][0], 1);
                    }
                }
            }
        }
        m_lastTime = intTime;
    }

    // ODE single step integration
    m_timeScheme->InitializeScheme(m_dt, m_U[bound][point], m_currentTime,
                                   m_ODE);
    m_y[bound][point] = m_timeScheme->TimeIntegrate(0, m_dt);

    // Compute output
    Array<OneD, NekDouble> bsum(m_y[bound][point][0].size(), 0.0);
    for (short i = 0; i < m_A.GetDimension() - 1; ++i)
    {
        Vmath::Svtvp(bsum.size(), m_B[i] - m_B[m_A.GetDimension() - 1] * m_A[i],
                     &m_y[bound][point][1 + i][0], 1, &bsum[0], 1, &bsum[0], 1);
    }
    Vmath::Svtvp(bsum.size(), m_B[m_A.GetDimension() - 1], &u[0], 1, &bsum[0],
                 1, &bsum[0], 1);

    return bsum;
}

/**
 * @brief Operator used for the ODE time integration.
 */
void PerforatedPlate::LinerOperator(
    const Array<OneD, const Array<OneD, NekDouble>> &inx,
    Array<OneD, Array<OneD, NekDouble>> &outx,
    [[maybe_unused]] const NekDouble time)
{
    // inx element 0 is u vector, element 1 to end are ss variables and its
    // derivatives; outx elements are the derivatives of x and u
    Array<OneD, NekDouble> asum(inx[0].size(), 0.0);
    for (short i = 1; i < m_A.GetDimension(); ++i)
    {
        if (i < m_A.GetDimension() - 1)
        {
            Vmath::Vcopy(inx[i].size(), &inx[i + 1][0], 1, &outx[i][0], 1);
        }
        Vmath::Svtvp(inx[i].size(), m_A[i - 1], &inx[i][0], 1, &asum[0], 1,
                     &asum[0], 1);
    }
    Vmath::Vsub(inx[0].size(), &inx[0][0], 1, &asum[0], 1,
                &outx[m_A.GetDimension() - 1][0], 1);
    Vmath::Zero(inx[0].size(), &outx[0][0], 1);
}

/**
 * @brief Operator used for the ODE time integration.
 */
void PerforatedPlate::LinerProjection(
    const Array<OneD, Array<OneD, NekDouble>> &inx,
    Array<OneD, Array<OneD, NekDouble>> &outx, [[maybe_unused]] NekDouble time)
{
    for (int i = 0; i < inx.size(); ++i)
    {
        Vmath::Vcopy(inx[i].size(), &inx[i][0], 1, &outx[i][0], 1);
    }
}

} // namespace Nektar
