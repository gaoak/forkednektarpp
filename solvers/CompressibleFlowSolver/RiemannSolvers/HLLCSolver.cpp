///////////////////////////////////////////////////////////////////////////////
//
// File: HLLCSolver.cpp
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
// Description: HLLC Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#include <CompressibleFlowSolver/RiemannSolvers/HLLCSolver.h>

namespace Nektar
{
std::string HLLCSolver::solverName =
    SolverUtils::GetRiemannSolverFactory().RegisterCreatorFunction(
        "HLLC", HLLCSolver::create, "HLLC Riemann solver");

HLLCSolver::HLLCSolver(const LibUtilities::SessionReaderSharedPtr &pSession)
    : CompressibleSolver(pSession)
{
}

/**
 * @brief HLLC Riemann solver
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
void HLLCSolver::v_PointSolve(NekDouble rhoL, const NekDouble *momL,
                              NekDouble EL, NekDouble rhoR,
                              const NekDouble *momR, NekDouble ER,
                              const NekDouble *normal, NekDouble &rhof,
                              NekDouble *momf, NekDouble &Ef)
{
    // Left and right velocities and normal velocities
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
    // Left and right total enthalpy
    NekDouble HL = (EL + pL) / rhoL;
    NekDouble HR = (ER + pR) / rhoR;

    // Square root of rhoL and rhoR.
    NekDouble srL  = sqrt(rhoL);
    NekDouble srR  = sqrt(rhoR);
    NekDouble srLR = srL + srR;

    // Roe average state
    NekDouble uRoe[3];
    for (size_t d = 0; d < 3; ++d)
    {
        uRoe[d] = (srL * uL[d] + srR * uR[d]) / srLR;
    }
    NekDouble unRoe = RiemannDot(uRoe, normal);
    NekDouble URoe2 = RiemannDot(uRoe, uRoe);
    NekDouble HRoe  = (srL * HL + srR * HR) / srLR;
    NekDouble cRoe  = GetRoeSoundSpeed(rhoL, pL, eL, HL, srL, rhoR, pR, eR, HR,
                                       srR, HRoe, URoe2, srLR);

    // Mass flux through the interface from each side
    NekDouble mnL = rhoL * unL;
    NekDouble mnR = rhoR * unR;

    // Maximum wave speeds
    NekDouble SL = std::min(unL - cL, unRoe - cRoe);
    NekDouble SR = std::max(unR + cR, unRoe + cRoe);

    // HLLC Riemann fluxes (positive case)
    if (SL >= 0)
    {
        rhof = mnL;
        for (size_t d = 0; d < 3; ++d)
        {
            momf[d] = mnL * uL[d] + pL * normal[d];
        }
        Ef = unL * (EL + pL);
    }
    // HLLC Riemann fluxes (negative case)
    else if (SR <= 0)
    {
        rhof = mnR;
        for (size_t d = 0; d < 3; ++d)
        {
            momf[d] = mnR * uR[d] + pR * normal[d];
        }
        Ef = unR * (ER + pR);
    }
    // HLLC Riemann fluxes (general case (SL < 0 | SR > 0)
    else
    {
        NekDouble SM = (pR - pL + mnL * (SL - unL) - mnR * (SR - unR)) /
                       (rhoL * (SL - unL) - rhoR * (SR - unR));

        // In the star region only the normal velocity changes, to SM; the
        // tangential velocity is advected unaltered. In Cartesian components
        // that is u + (SM - u.n) n, which needs no tangential basis.
        NekDouble rhoML = rhoL * (SL - unL) / (SL - SM);
        NekDouble EML =
            rhoML * (EL / rhoL + (SM - unL) * (SM + pL / (rhoL * (SL - unL))));

        NekDouble rhoMR = rhoR * (SR - unR) / (SR - SM);
        NekDouble EMR =
            rhoMR * (ER / rhoR + (SM - unR) * (SM + pR / (rhoR * (SR - unR))));

        if (SL < 0.0 && SM >= 0.0)
        {
            rhof = mnL + SL * (rhoML - rhoL);
            for (size_t d = 0; d < 3; ++d)
            {
                NekDouble momML = rhoML * (uL[d] + (SM - unL) * normal[d]);
                momf[d] = mnL * uL[d] + pL * normal[d] + SL * (momML - momL[d]);
            }
            Ef = unL * (EL + pL) + SL * (EML - EL);
        }
        else if (SM < 0.0 && SR > 0.0)
        {
            rhof = mnR + SR * (rhoMR - rhoR);
            for (size_t d = 0; d < 3; ++d)
            {
                NekDouble momMR = rhoMR * (uR[d] + (SM - unR) * normal[d]);
                momf[d] = mnR * uR[d] + pR * normal[d] + SR * (momMR - momR[d]);
            }
            Ef = unR * (ER + pR) + SR * (EMR - ER);
        }
    }
}
} // namespace Nektar
