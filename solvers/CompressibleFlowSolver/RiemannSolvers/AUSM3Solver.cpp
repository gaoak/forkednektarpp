///////////////////////////////////////////////////////////////////////////////
//
// File: AUSM3Solver.cpp
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
// Description: AUSM3 Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#include <CompressibleFlowSolver/RiemannSolvers/AUSM3Solver.h>

namespace Nektar
{

std::string AUSM3Solver::solverName =
    SolverUtils::GetRiemannSolverFactory().RegisterCreatorFunction(
        "AUSM3", AUSM3Solver::create, "AUSM3 Riemann solver");

AUSM3Solver::AUSM3Solver(const LibUtilities::SessionReaderSharedPtr &pSession)
    : AUSM0Solver(pSession)
{
    pSession->LoadParameter("Mco", m_Mco, 0.01);
}

/**
 * @brief AUSM3 Riemann solver
 *
 * Written directly in Cartesian components against the trace normal; see
 * CompressibleSolver::v_PointSolve for why no rotation is needed.
 *
 * @param rhoL      Density left state.
 * @param momL      Momentum vector (3 components) left state.
 * @param EL        Energy left state.
 * @param rhoR      Density right state.
 * @param momR      Momentum vector (3 components) right state.
 * @param ER        Energy right state.
 * @param normal    Unit trace normal (3 components).
 * @param rhof      Computed Riemann flux for density.
 * @param momf      Computed Riemann flux for momentum (3 components).
 * @param Ef        Computed Riemann flux for energy.
 */
void AUSM3Solver::v_PointSolve(NekDouble rhoL, const NekDouble *momL,
                               NekDouble EL, NekDouble rhoR,
                               const NekDouble *momR, NekDouble ER,
                               const NekDouble *normal, NekDouble &rhof,
                               NekDouble *momf, NekDouble &Ef)
{
    // Left and Right velocities and normal velocities
    NekDouble uL[3], uR[3];
    for (size_t d = 0; d < 3; ++d)
    {
        uL[d] = momL[d] / rhoL;
        uR[d] = momR[d] / rhoR;
    }
    NekDouble unL = RiemannDot(uL, normal);
    NekDouble unR = RiemannDot(uR, normal);

    // Internal energy (per unit mass)
    NekDouble eL = (EL - 0.5 * RiemannDot(momL, uL)) / rhoL;
    NekDouble eR = (ER - 0.5 * RiemannDot(momR, uR)) / rhoR;
    // Pressure
    NekDouble pL = m_eos->GetPressure(rhoL, eL);
    NekDouble pR = m_eos->GetPressure(rhoR, eR);
    // Speed of sound
    NekDouble cL = m_eos->GetSoundSpeed(rhoL, eL);
    NekDouble cR = m_eos->GetSoundSpeed(rhoR, eR);

    // Average speeds of sound
    NekDouble cA = 0.5 * (cL + cR);

    // Local Mach numbers, based on the velocity normal to the trace
    NekDouble ML = unL / cA;
    NekDouble MR = unR / cA;

    // Parameters for specify the upwinding
    // Note: if fa = 1 then AUSM3 = AUSM3
    NekDouble Mtilde = 0.5 * (ML * ML + MR * MR);
    NekDouble Mo    = std::sqrt(std::min(1.0, std::max(Mtilde, m_Mco * m_Mco)));
    NekDouble fa    = Mo * (2.0 - Mo);
    NekDouble beta  = 0.125;
    NekDouble alpha = 0.1875;
    NekDouble sigma = 1.0;
    NekDouble Kp    = 0.25;
    NekDouble Ku    = 0.75;
    NekDouble rhoA  = 0.5 * (rhoL + rhoR);
    NekDouble Mp    = -(Kp / fa) * ((pR - pL) / (rhoA * cA * cA)) *
                   std::max(1.0 - sigma * Mtilde, 0.0);

    NekDouble Mbar = M4Function(0, beta, ML) + M4Function(1, beta, MR) + Mp;

    NekDouble pu = -2.0 * Ku * rhoA * cA * cA * (MR - ML) *
                   P5Function(0, alpha, ML) * P5Function(1, alpha, MR);

    NekDouble pbar =
        pL * P5Function(0, alpha, ML) + pR * P5Function(1, alpha, MR) + pu;

    // The convective part is the mass flux times the upwinded velocity
    // vector, and the pressure part acts along the trace normal.
    if (Mbar >= 0.0)
    {
        rhof = cA * Mbar * rhoL;
        for (size_t d = 0; d < 3; ++d)
        {
            momf[d] = cA * Mbar * rhoL * uL[d] + pbar * normal[d];
        }
        Ef = cA * Mbar * (EL + pL);
    }
    else
    {
        rhof = cA * Mbar * rhoR;
        for (size_t d = 0; d < 3; ++d)
        {
            momf[d] = cA * Mbar * rhoR * uR[d] + pbar * normal[d];
        }
        Ef = cA * Mbar * (ER + pR);
    }
}

} // namespace Nektar
