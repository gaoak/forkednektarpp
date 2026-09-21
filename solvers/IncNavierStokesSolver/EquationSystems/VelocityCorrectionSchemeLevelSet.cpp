///////////////////////////////////////////////////////////////////////////////
//
// File: VelocityCorrectionSchemeLevelSet.cpp
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
// Description: Level Set Velocity Correction Scheme for the Incompressible
// Navier Stokes equations coupled with level-set method for two-phase flow
//
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/EquationSystems/VelocityCorrectionSchemeLevelSet.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <LocalRegions/Expansion2D.h>
#include <LocalRegions/Expansion3D.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/GJPStabilisation.h>
#include <SolverUtils/Core/Misc.h>

#include <boost/algorithm/string.hpp>

using namespace std;

namespace Nektar
{
using namespace MultiRegions;

std::string VCSLevelSet::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "VCSLevelSet", VCSLevelSet::create);

std::string VCSLevelSet::solverTypeLookupId =
    LibUtilities::SessionReader::RegisterEnumValue("SolverType", "VCSLevelSet",
                                                   eVCSLevelSet);

/**
 * Constructor. Creates ...
 *
 * \param
 * \param
 */
VCSLevelSet::VCSLevelSet(const LibUtilities::SessionReaderSharedPtr &pSession,
                         const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph),
      VelocityCorrectionScheme(pSession, pGraph)
{
}

void VCSLevelSet::v_InitObject(bool DeclareField)
{
    VelocityCorrectionScheme::v_InitObject(DeclareField);

    // Set up parameters for the Level set system
    m_session->LoadParameter("epsilon", m_epsilon, 0.0);
    m_session->LoadParameter("rhol", m_rhol, 1.0);
    m_session->LoadParameter("rhof", m_rhof, 1.0);
    m_session->LoadParameter("viscl", m_viscl, 1.0);
    m_session->LoadParameter("viscf", m_viscf, 1.0);
    m_session->LoadParameter("sigma", m_sigma, 0.0);
    m_session->LoadParameter("gravity", m_gravity, 0.0);
}

/**
 *
 */
void VCSLevelSet::v_GenerateSummary(SolverUtils::SummaryList &s)
{
    AdvectionSystem::v_GenerateSummary(s);
    SolverUtils::AddSummaryItem(s, "Splitting Scheme",
                                "Velocity correction Level Set");
}

/**
 * Implicit part of the method - Poisson + nConv*Helmholtz
 */
void VCSLevelSet::v_SolveUnsteadyStokesSystem(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray,
    [[maybe_unused]] const NekDouble time, const NekDouble aii_Dt)
{

    // Check if user tries to setup flow rate in a multiphase context
    if (m_flowrate > 0.0)
    {
        NEKERROR(ErrorUtil::efatal,
                 "Flow rate is not supported in multiphase flows. ");
    }

    int nvel = m_velocity.size();

    // Substep the pressure boundary condition if using substepping
    m_extrapolation->SubStepSetPressureBCs(inarray, aii_Dt, m_kinvis);

    // Set up scalar(phi) forcing term for LS equation
    LibUtilities::Timer timer;
    timer.Start();
    SetUpLevelSetForcing(inarray[nvel], m_F[nvel], aii_Dt);
    timer.Stop();
    timer.AccumulateRegion("Level Set Forcing");

    // Solve Level Set System
    timer.Start();
    SolveLevelSet(m_F[nvel], inarray[nvel], outarray[nvel], outarray[nvel + 1],
                  outarray[nvel + 2], aii_Dt);
    timer.Stop();
    timer.AccumulateRegion("Level Set Solve");

    // Set up forcing term for pressure Poisson equation
    timer.Start();
    SetUpPressureForcing(inarray, m_F, aii_Dt);
    timer.Stop();
    timer.AccumulateRegion("Pressure Forcing");

    // Solve Pressure System
    timer.Start();
    SolvePressure(m_F[0]);
    timer.Stop();
    timer.AccumulateRegion("Pressure Solve");

    // Set up forcing term for Helmholtz problems
    timer.Start();
    SetUpViscousForcing(inarray, m_F, aii_Dt);
    timer.Stop();
    timer.AccumulateRegion("Viscous Forcing");

    // Solve velocity system
    timer.Start();
    SolveViscous(m_F, inarray, outarray, aii_Dt);
    timer.Stop();
    timer.AccumulateRegion("Viscous Solve");
}

/**
 * Computes the forcing term that includes explicit artificial compression flux
 * for level set solve.
 */
void VCSLevelSet::SetUpLevelSetForcing(const Array<OneD, NekDouble> &fields,
                                       Array<OneD, NekDouble> &Forcing,
                                       const NekDouble aii_Dt)
{

    int i;
    int nvel    = m_velocity.size();
    int physTot = m_fields[nvel]->GetTotPoints();

    Vmath::Zero(physTot, Forcing, 1);

    Array<OneD, Array<OneD, NekDouble>> gradPhi(nvel);
    Array<OneD, Array<OneD, NekDouble>> IntNorm(nvel);
    for (i = 0; i < nvel; ++i)
    {
        gradPhi[i] = Array<OneD, NekDouble>(physTot, 0.0);
        IntNorm[i] = Array<OneD, NekDouble>(physTot, 0.0);
    }

    for (i = 0; i < nvel; ++i)
    {
        m_fields[nvel]->PhysDeriv(DirCartesianMap[i], fields, gradPhi[i]);
    }

    Array<OneD, NekDouble> gradPhiMag(physTot, 0.0);
    Array<OneD, NekDouble> tmp(physTot, 0.0);
    for (i = 0; i < nvel; ++i)
    {
        Vmath::Vpow(physTot, gradPhi[i], 1, 2.0, tmp, 1);
        Vmath::Vadd(physTot, gradPhiMag, 1, tmp, 1, gradPhiMag, 1);
    }
    Vmath::Vsqrt(physTot, gradPhiMag, 1, gradPhiMag, 1);
    Vmath::Vabs(physTot, gradPhiMag, 1, gradPhiMag, 1);

    for (i = 0; i < nvel; ++i)
    {
        Vmath::Vdiv(physTot, gradPhi[i], 1, gradPhiMag, 1, IntNorm[i], 1);

        for (int j = 0; j < physTot; ++j)
        {
            if (gradPhi[i][j] == 0 || gradPhiMag[j] == 0 ||
                !std::isfinite(IntNorm[i][j]))
            {
                IntNorm[i][j] = 0;
            }
        }
    }

    Array<OneD, NekDouble> divIntNorm(physTot, 0.0);
    Vmath::Zero(physTot, tmp, 1);
    for (i = 0; i < nvel; ++i)
    {
        m_fields[nvel]->PhysDeriv(DirCartesianMap[i], IntNorm[i], tmp);
        Vmath::Vadd(physTot, divIntNorm, 1, tmp, 1, divIntNorm, 1);
    }

    // phiOnePhiTwo = (phi)* (1-phi)
    Array<OneD, NekDouble> oneMinusPhi(physTot, 0.0);
    Array<OneD, NekDouble> phiOnephiTwo(physTot, 0.0);
    Vmath::Ssub(physTot, 1.0, fields, 1, oneMinusPhi, 1);
    Vmath::Vmul(physTot, fields, 1, oneMinusPhi, 1, phiOnephiTwo, 1);

    // OneMinusTwoPhi = 1-2phi
    Array<OneD, NekDouble> twoPhi(physTot, 0.0);
    Array<OneD, NekDouble> oneMinusTwoPhi(physTot, 0.0);
    Vmath::Smul(physTot, 2.0, fields, 1, twoPhi, 1);
    Vmath::Ssub(physTot, 1.0, twoPhi, 1, oneMinusTwoPhi, 1);

    // gDI = gradPhi(Dot)IntNorm
    Array<OneD, NekDouble> gDI(physTot, 0.0);
    Vmath::Zero(physTot, tmp, 1);
    for (i = 0; i < nvel; ++i)
    {
        Vmath::Vmul(physTot, gradPhi[i], 1, IntNorm[i], 1, tmp, 1);
        Vmath::Vadd(physTot, tmp, 1, gDI, 1, gDI, 1);
    }

    Array<OneD, NekDouble> Fc(physTot, 0.0);
    Vmath::Vvtvvtp(physTot, phiOnephiTwo, 1, divIntNorm, 1, oneMinusTwoPhi, 1,
                   gDI, 1, Fc, 1);

    NekDouble aii_dtinv = 1.0 / aii_Dt / m_epsilon;
    Vmath::Svtsvtp(physTot, -aii_dtinv, fields, 1, 1.0 / m_epsilon, Fc, 1,
                   Forcing, 1);
}

/**
 * Computes the forcing term that includes surface tension forces for the
 * pressure poisson solve.
 */
void VCSLevelSet::v_SetUpPressureForcing(
    const Array<OneD, const Array<OneD, NekDouble>> &fields,
    Array<OneD, Array<OneD, NekDouble>> &Forcing, const NekDouble aii_Dt)
{
    size_t i;
    size_t physTot = m_fields[0]->GetTotPoints();
    size_t nvel    = m_velocity.size();

    m_fields[nvel]->BwdTrans(m_fields[nvel]->GetCoeffs(),
                             m_fields[nvel]->UpdatePhys());
    // Surface tension components
    Array<OneD, Array<OneD, NekDouble>> gradPhi(nvel);
    Array<OneD, Array<OneD, NekDouble>> IntNorm(nvel);
    Array<OneD, Array<OneD, NekDouble>> SurfTen(nvel);
    m_SurfTen = Array<OneD, Array<OneD, NekDouble>>(nvel);
    for (int i = 0; i < nvel; ++i)
    {
        m_SurfTen[i] = Array<OneD, NekDouble>(physTot, 0.0);
        gradPhi[i]   = Array<OneD, NekDouble>(physTot, 0.0);
        IntNorm[i]   = Array<OneD, NekDouble>(physTot, 0.0);
        SurfTen[i]   = Array<OneD, NekDouble>(physTot, 0.0);
    }

    for (int i = 0; i < nvel; ++i)
    {
        m_fields[nvel]->PhysDeriv(Nektar::MultiRegions::DirCartesianMap[i],
                                  m_fields[nvel]->GetPhys(), gradPhi[i]);
    }

    Array<OneD, NekDouble> gradPhiMag(physTot, 0.0);
    Array<OneD, NekDouble> tmp(physTot, 0.0);
    for (int i = 0; i < nvel; ++i)
    {
        Vmath::Vpow(physTot, gradPhi[i], 1, 2.0, tmp, 1);
        Vmath::Vadd(physTot, gradPhiMag, 1, tmp, 1, gradPhiMag, 1);
    }

    Vmath::Vsqrt(physTot, gradPhiMag, 1, gradPhiMag, 1);
    Vmath::Vabs(physTot, gradPhiMag, 1, gradPhiMag, 1);
    for (int i = 0; i < nvel; ++i)
    {
        Vmath::Vdiv(physTot, gradPhi[i], 1, gradPhiMag, 1, IntNorm[i], 1);

        for (int j = 0; j < physTot; ++j)
        {
            if (gradPhi[i][j] == 0 || gradPhiMag[j] == 0 ||
                !std::isfinite(IntNorm[i][j]))
            {
                IntNorm[i][j] = 0;
            }
        }
    }
    Array<OneD, NekDouble> divIntNorm(physTot, 0.0);
    Vmath::Zero(physTot, tmp, 1);
    for (int i = 0; i < nvel; ++i)
    {
        m_fields[nvel]->PhysDeriv(Nektar::MultiRegions::DirCartesianMap[i],
                                  IntNorm[i], tmp);
        Vmath::Vadd(physTot, divIntNorm, 1, tmp, 1, divIntNorm, 1);
    }
    m_fields[nvel + 1]->BwdTrans(m_fields[nvel + 1]->GetCoeffs(),
                                 m_fields[nvel + 1]->UpdatePhys());

    // Calculate Surface tension components
    for (int i = 0; i < nvel; ++i)
    {
        Vmath::Vmul(physTot, gradPhi[i], 1, divIntNorm, 1, SurfTen[i], 1);
        Vmath::Smul(physTot, m_sigma, SurfTen[i], 1, SurfTen[i], 1);
        Vmath::Vdiv(physTot, SurfTen[i], 1, m_fields[nvel + 1]->GetPhys(), 1,
                    SurfTen[i], 1);
    }

    // Fwd and Bwd transfer to make the function continuous
    int ncoeffs = m_fields[0]->GetNcoeffs();
    for (int i = 0; i < nvel; ++i)
    {
        Array<OneD, NekDouble> coeffs(ncoeffs, 0.0);
        m_fields[nvel]->FwdTrans(SurfTen[i], coeffs);
        m_fields[nvel]->BwdTrans(coeffs, SurfTen[i]);
        Vmath::Vcopy(physTot, SurfTen[i], 1, m_SurfTen[i], 1);
    }

    // div(u*[i])/dt - div(ST[i])
    Vmath::Zero(physTot, tmp, 1);
    Vmath::Zero(physTot, Forcing[1], 1);
    for (int i = 0; i < nvel; ++i)
    {
        m_fields[nvel]->PhysDeriv(Nektar::MultiRegions::DirCartesianMap[i],
                                  SurfTen[i], tmp);
        Vmath::Vadd(physTot, Forcing[1], 1, tmp, 1, Forcing[1], 1);
    }

    m_fields[0]->PhysDeriv(eX, fields[0], Forcing[0]);
    Vmath::Zero(physTot, tmp, 1);
    for (i = 1; i < nvel; ++i)
    {
        m_fields[i]->PhysDeriv(DirCartesianMap[i], fields[i], tmp);
        Vmath::Vadd(physTot, tmp, 1, Forcing[0], 1, Forcing[0], 1);
    }
    Vmath::Smul(physTot, 1.0 / aii_Dt, Forcing[0], 1, Forcing[0], 1);

    Vmath::Vsub(physTot, Forcing[0], 1, Forcing[1], 1, Forcing[0], 1);
}

/**
 * Computes the forcing term that includes the updated pressure gradient, and
 * surface tension forces.
 */
void VCSLevelSet::v_SetUpViscousForcing(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &Forcing, const NekDouble aii_Dt)
{
    NekDouble aii_dtinv = 1.0 / aii_Dt;
    size_t physTot      = m_fields[0]->GetTotPoints();

    // Grad p
    m_pressure->BwdTrans(m_pressure->GetCoeffs(), m_pressure->UpdatePhys());

    int nvel = m_velocity.size();
    m_fields[nvel + 1]->BwdTrans(m_fields[nvel + 1]->GetCoeffs(),
                                 m_fields[nvel + 1]->UpdatePhys());
    if (nvel == 2)
    {
        m_pressure->PhysDeriv(m_pressure->GetPhys(), Forcing[m_velocity[0]],
                              Forcing[m_velocity[1]]);
        Vmath::Vdiv(physTot, Forcing[m_velocity[0]], 1,
                    m_fields[nvel + 1]->GetPhys(), 1, Forcing[m_velocity[0]],
                    1);
        Vmath::Vdiv(physTot, Forcing[m_velocity[1]], 1,
                    m_fields[nvel + 1]->GetPhys(), 1, Forcing[m_velocity[1]],
                    1);
    }
    else
    {
        m_pressure->PhysDeriv(m_pressure->GetPhys(), Forcing[m_velocity[0]],
                              Forcing[m_velocity[1]], Forcing[m_velocity[2]]);
        Vmath::Vdiv(physTot, Forcing[m_velocity[0]], 1,
                    m_fields[nvel + 1]->GetPhys(), 1, Forcing[m_velocity[0]],
                    1);
        Vmath::Vdiv(physTot, Forcing[m_velocity[1]], 1,
                    m_fields[nvel + 1]->GetPhys(), 1, Forcing[m_velocity[1]],
                    1);
        Vmath::Vdiv(physTot, Forcing[m_velocity[2]], 1,
                    m_fields[nvel + 1]->GetPhys(), 1, Forcing[m_velocity[2]],
                    1);
    }

    // Add Surface tension
    for (int i = 0; i < nvel; ++i)
    {
        Vmath::Vadd(physTot, Forcing[i], 1, m_SurfTen[i], 1, Forcing[i], 1);
    }

    // zero convective fields.
    for (int i = nvel; i < m_nConvectiveFields; ++i)
    {
        Vmath::Zero(physTot, Forcing[i], 1);
    }

    // calculate 1/nudt and 1/nu to multiply with inarray and forcing
    // respectively
    Array<OneD, NekDouble> aiiDtInvVisc(physTot, 0.0);
    Array<OneD, NekDouble> InvVisc(physTot, 0.0);
    m_fields[nvel + 2]->BwdTrans(m_fields[nvel + 2]->GetCoeffs(),
                                 m_fields[nvel + 2]->UpdatePhys());
    Vmath::Sdiv(physTot, -aii_dtinv, m_fields[nvel + 2]->GetPhys(), 1,
                aiiDtInvVisc, 1);
    Vmath::Sdiv(physTot, 1.0, m_fields[nvel + 2]->GetPhys(), 1, InvVisc, 1);

    for (int i = 0; i < nvel; ++i)
    {
        Vmath::Vvtvvtp(physTot, aiiDtInvVisc, 1, inarray[i], 1, InvVisc, 1,
                       Forcing[i], 1, Forcing[i], 1);
    }
}

/**
 * Solve level set system using a Helmholtz-type operator.
 */
void VCSLevelSet::SolveLevelSet(
    const Array<OneD, NekDouble> &Forcing,
    [[maybe_unused]] const Array<OneD, NekDouble> &inarray,
    Array<OneD, NekDouble> &outarrayPhi, Array<OneD, NekDouble> &outarrayRho,
    Array<OneD, NekDouble> &outarrayVisc, const NekDouble aii_Dt)
{
    int nvel    = m_velocity.size();
    int physTot = m_fields[nvel]->GetTotPoints();
    StdRegions::ConstFactorMap factors;
    StdRegions::VarCoeffMap varCoeffMap     = StdRegions::NullVarCoeffMap;
    StdRegions::VarFactorsMap varFactorsMap = StdRegions::NullVarFactorsMap;
    factors[StdRegions::eFactorLambda]      = 1.0 / aii_Dt / m_epsilon;

    m_fields[nvel]->HelmSolve(Forcing, m_fields[nvel]->UpdateCoeffs(), factors,
                              varCoeffMap, varFactorsMap);
    m_fields[nvel]->BwdTrans(m_fields[nvel]->GetCoeffs(), outarrayPhi);

    // f= fluid(gas) l=liquid
    // Rho = (rhol-rhof)*phi + rhof = rhol*phi +rhof*(1-phi)
    Vmath::Smul(physTot, m_rhol - m_rhof, outarrayPhi, 1, outarrayRho, 1);
    Vmath::Sadd(physTot, m_rhof, outarrayRho, 1, outarrayRho, 1);
    m_fields[nvel + 1]->FwdTransLocalElmt(outarrayRho,
                                          m_fields[nvel + 1]->UpdateCoeffs());

    // DynVisc = (viscl/rhol - viscf/rhof)*phi + viscf/rhof
    Vmath::Smul(physTot, m_viscl / m_rhol - m_viscf / m_rhof, outarrayPhi, 1,
                outarrayVisc, 1);
    Vmath::Sadd(physTot, m_viscf / m_rhof, outarrayVisc, 1, outarrayVisc, 1);
    m_fields[nvel + 2]->FwdTransLocalElmt(outarrayVisc,
                                          m_fields[nvel + 2]->UpdateCoeffs());
}

/**
 * Solve pressure system via a variable-coefficient Poisson problem.
 */
void VCSLevelSet::v_SolvePressure(const Array<OneD, NekDouble> &Forcing)
{
    StdRegions::ConstFactorMap factors;
    StdRegions::VarCoeffMap varCoeffMap     = StdRegions::NullVarCoeffMap;
    StdRegions::VarFactorsMap varFactorsMap = StdRegions::NullVarFactorsMap;

    // Setup coefficient for equation
    factors[StdRegions::eFactorLambda] = 0.0;

    int nvel    = m_velocity.size();
    int physTot = m_fields[nvel + 1]->GetTotPoints();

    // Calculate 1/rho to use as laplacian coefficients
    Array<OneD, NekDouble> InvRho(physTot, 0.0);
    m_fields[nvel + 1]->BwdTrans(m_fields[nvel + 1]->GetCoeffs(),
                                 m_fields[nvel + 1]->UpdatePhys());
    Vmath::Sdiv(physTot, 1.0, m_fields[nvel + 1]->GetPhys(), 1, InvRho, 1);
    varCoeffMap[StdRegions::eVarCoeffD00] = InvRho;
    varCoeffMap[StdRegions::eVarCoeffD11] = InvRho;
    if (nvel == 3)
    {
        varCoeffMap[StdRegions::eVarCoeffD22] = InvRho;
    }

    auto gkey = m_pressure->HelmSolve(Forcing, m_pressure->UpdateCoeffs(),
                                      factors, varCoeffMap, varFactorsMap);
    m_pressure->UnsetGlobalLinSys(gkey, true);

    // Add presure to outflow bc if using convective like BCs
    m_extrapolation->AddPressureToOutflowBCs(m_kinvis);
}

/**
 * Solve velocity system velocity system via variable-viscosity Helmholtz
 * problems for each directional velocity component using a mass matrix variable
 * coefficient.
 */
void VCSLevelSet::v_SolveViscous(
    const Array<OneD, const Array<OneD, NekDouble>> &Forcing,
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble aii_Dt)
{
    StdRegions::ConstFactorMap factors;
    StdRegions::VarCoeffMap varCoeffMap     = StdRegions::NullVarCoeffMap;
    StdRegions::VarFactorsMap varFactorsMap = StdRegions::NullVarFactorsMap;

    AppendSVVFactors(factors, varFactorsMap);
    ComputeGJPNormalVelocity(inarray, varCoeffMap);

    size_t nvel = m_velocity.size();
    int physTot = m_fields[nvel + 2]->GetTotPoints();

    // Calculate 1/nu to use as mass matrix coefficient
    Array<OneD, NekDouble> InvVisc(physTot, 0.0);
    m_fields[nvel + 2]->BwdTrans(m_fields[nvel + 2]->GetCoeffs(),
                                 m_fields[nvel + 2]->UpdatePhys());
    Vmath::Sdiv(physTot, 1.0, m_fields[nvel + 2]->GetPhys(), 1, InvVisc, 1);
    varCoeffMap[StdRegions::eVarCoeffMass] = InvVisc;

    // Solve Helmholtz system and put in Physical space
    for (int i = 0; i < nvel; ++i)
    {
        // Add diffusion coefficient to GJP matrix operator (Implicit part)
        if (m_useGJPStabilisation)
        {
            if (m_useGJPNormalVel)
            {
                factors[StdRegions::eFactorGJP] = m_GJPJumpScale;
            }
            else
            {
                factors[StdRegions::eFactorGJP] =
                    m_GJPJumpScale / m_diffCoeff[i];
            }
        }

        // Setup coefficients for equation
        factors[StdRegions::eFactorLambda] = 1.0 / aii_Dt;
        auto gkey =
            m_fields[i]->HelmSolve(Forcing[i], m_fields[i]->UpdateCoeffs(),
                                   factors, varCoeffMap, varFactorsMap);
        m_fields[i]->UnsetGlobalLinSys(gkey, true);
        m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(), outarray[i]);
    }
}

} // namespace Nektar
