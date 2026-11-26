///////////////////////////////////////////////////////////////////////////////
//
// File: ForcingLorentz.cpp
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
// Description: Add the Lorentz force, j x B
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/Forcing/ForcingLorentz.h>
#include <LibUtilities/BasicUtils/StringUtils.hpp>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ExpList.h>
#include <SolverUtils/Filters/FilterInterfaces.hpp>
#include <boost/lexical_cast.hpp>
#include <cmath>
#include <iostream>

namespace Nektar::SolverUtils
{

std::string ForcingLorentz::classNameBody =
    GetForcingFactory().RegisterCreatorFunction(
        "ForcingLorentz", ForcingLorentz::create, "Quasi-static MHD forcing");

/**
 * @brief
 * @param pSession
 * @param pEquation
 */
ForcingLorentz::ForcingLorentz(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const std::weak_ptr<EquationSystem> &pEquation)
    : Forcing(pSession, pEquation)
{
}

ForcingLorentz::~ForcingLorentz(void)
{
}

/**
 * @brief Initialise the forcing module
 * @param pFields
 * @param pNumForcingFields
 * @param pForce
 */
void ForcingLorentz::v_InitObject(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
    [[maybe_unused]] const unsigned int &pNumForcingFields,
    const LibUtilities::SessionNode &pForce)
{
    // read space dimention
    bool isH1d, isH2d;
    m_session->MatchSolverInfo("Homogeneous", "1D", isH1d, false);
    m_session->MatchSolverInfo("Homogeneous", "2D", isH2d, false);
    int expdim = isH2d ? 1 : pFields[0]->GetGraph()->GetMeshDimension();
    m_spacedim = expdim + (isH1d ? 1 : 0) + (isH2d ? 2 : 0);
    auto equ   = m_equ.lock();
    ASSERTL0(equ, "Weak pointer to the equation system is expired");
    m_FluidEq = std::dynamic_pointer_cast<VelocityCorrectionScheme>(equ);
    // read electric conductivity
    m_session->LoadParameter("ElectricConductivity", m_sigma, -1.);
    ASSERTL0(m_sigma > 0,
             "ElectricConductivity should be defined and be positive.");
    // Read the constant external electric and magnetic fields.
    m_E0                              = Array<OneD, NekDouble>(3, 0.);
    m_B0                              = Array<OneD, NekDouble>(3, 0.);
    std::vector<std::string> elecVars = {"Ex", "Ey", "Ez"};
    std::vector<std::string> magVars  = {"Bx", "By", "Bz"};
    ASSERTL0(pForce.DefinesChild("ELECTRICMAGNETICFIELDS"),
             "Requires ELECTRICMAGNETICFIELDS tag specifying the function "
             "that prescribes the external electric and magnetic fields.");
    std::string electromagneticFields =
        pForce.GetValue("ELECTRICMAGNETICFIELDS");
    ASSERTL0(m_session->DefinesFunction(electromagneticFields),
             "Function '" + electromagneticFields +
                 "' is not defined in the session.");
    for (size_t i = 0; i < 3; ++i)
    {
        if (m_session->DefinesFunction(electromagneticFields, elecVars[i]))
        {
            LibUtilities::EquationSharedPtr fieldEquation =
                m_session->GetFunction(electromagneticFields, elecVars[i]);
            m_E0[i] = fieldEquation->Evaluate(0., 0., 0., 0.);
        }
        if (m_session->DefinesFunction(electromagneticFields, magVars[i]))
        {
            LibUtilities::EquationSharedPtr fieldEquation =
                m_session->GetFunction(electromagneticFields, magVars[i]);
            m_B0[i] = fieldEquation->Evaluate(0., 0., 0., 0.);
        }
    }
    m_damping =
        m_sigma * (m_B0[0] * m_B0[0] + m_B0[1] * m_B0[1] + m_B0[2] * m_B0[2]);

    m_implicitDamping = false;
    if (pForce.DefinesChild("IMPLICITDAMPING"))
    {
        const std::string &text = pForce.GetValue("IMPLICITDAMPING");
        ASSERTL0(StringUtils::IEquals(text, "true") ||
                     StringUtils::IEquals(text, "false"),
                 "IMPLICITDAMPING must be either True or False.");
        m_implicitDamping = StringUtils::IEquals(text, "true");
    }
    // ------------------------------------------------------------------
    // Hartmann braking (Sommeria & Moreau 1982, J. Fluid Mech. 118, eq. 18)
    //
    // The wall Hartmann layers are not resolved; their net braking on the
    // core flow is modelled as a linear friction -chi_i u_i with
    //
    //   chi   = n_w sqrt(sigma |B0|^2 nu / rho) / a = n_w Ha nu / a^2,
    //   chi_i = chi (1 - b_hat_i^2),  b_hat = B0/|B0|.
    //
    // Here rho = 1 (kinematic viscosity convention), a is the gap between
    // the two Hartmann walls and n_w the number of active Hartmann walls.
    // chi_i is added to the implicit diagonal of the velocity Helmholtz
    // solve, next to the existing sigma |B0|^2 term; it never appears in the
    // explicit force, so v_Apply is unchanged.
    // ------------------------------------------------------------------
    bool hartmannBraking = false;
    if (pForce.DefinesChild("HARTMANNBRAKING"))
    {
        const std::string &text = pForce.GetValue("HARTMANNBRAKING");
        ASSERTL0(StringUtils::IEquals(text, "true") ||
                     StringUtils::IEquals(text, "false"),
                 "HARTMANNBRAKING must be either True or False.");
        hartmannBraking = StringUtils::IEquals(text, "true");
    }

    if (hartmannBraking)
    {
        NekDouble bmag = std::sqrt(m_damping / m_sigma);
        ASSERTL0(bmag > 0.0,
                 "Hartmann braking requires a non-zero external magnetic "
                 "field.");
        // The diagonal mask is exact only for an axis-aligned B0; a general
        // orientation couples the components through chi b_hat (b_hat . u).
        for (size_t i = 0; i < 3; ++i)
        {
            NekDouble bhat = m_B0[i] / bmag;
            ASSERTL0(std::abs(bhat) < 1e-10 ||
                         std::abs(std::abs(bhat) - 1.0) < 1e-10,
                     "Hartmann braking currently supports only a magnetic "
                     "field aligned with a coordinate axis.");
        }

        // Hartmann-wall gap a: either a number or the name of a <P>
        // parameter (the latter is preferred, since <P> entries are
        // expression-evaluated by the session reader).
        ASSERTL0(pForce.DefinesChild("HARTMANNGAP"),
                 "HARTMANNBRAKING requires HARTMANNGAP, the distance between "
                 "the two Hartmann walls.");
        const std::string &gapText = pForce.GetValue("HARTMANNGAP");
        NekDouble gap             = -1.0;
        if (m_session->DefinesParameter(gapText))
        {
            m_session->LoadParameter(gapText, gap);
        }
        else
        {
            bool parsed = true;
            try
            {
                gap = boost::lexical_cast<NekDouble>(gapText);
            }
            catch (const boost::bad_lexical_cast &)
            {
                parsed = false;
            }
            ASSERTL0(parsed, "HARTMANNGAP must be a positive number or the "
                             "name of a session parameter.");
        }
        ASSERTL0(gap > 0.0, "HARTMANNGAP must be positive.");

        int nWalls = 2;
        if (pForce.DefinesChild("HARTMANNWALLS"))
        {
            bool parsed = true;
            try
            {
                nWalls =
                    boost::lexical_cast<int>(pForce.GetValue("HARTMANNWALLS"));
            }
            catch (const boost::bad_lexical_cast &)
            {
                parsed = false;
            }
            ASSERTL0(parsed,
                     "HARTMANNWALLS must be either 1 or 2.");
            ASSERTL0(nWalls == 1 || nWalls == 2,
                     "HARTMANNWALLS must be either 1 or 2.");
        }

        const NekDouble kinvis = m_FluidEq->GetKinvis();
        const NekDouble chi =
            nWalls * std::sqrt(m_damping * kinvis) / gap;
        // Hartmann layer thickness delta_H = sqrt(rho nu / (sigma B0^2))
        // = a / Ha(a), with Ha(a) = |B0| a sqrt(sigma/(rho nu)).
        const NekDouble haGap =
            bmag * gap * std::sqrt(m_sigma / kinvis);
        const NekDouble deltaH = gap / haGap;

        // Total implicit damping: sigma |B0|^2 (only if it is treated
        // implicitly) plus the anisotropic braking term.
        Array<OneD, NekDouble> damping(3, 0.0);
        Array<OneD, NekDouble> chiComps(3, 0.0);
        for (size_t i = 0; i < 3; ++i)
        {
            NekDouble bhat = m_B0[i] / bmag;
            chiComps[i]    = chi * (1.0 - bhat * bhat);
            damping[i] =
                (m_implicitDamping ? m_damping : 0.0) + chiComps[i];
        }
        m_FluidEq->SetImplicitDamping(damping);

        if (m_session->GetComm()->GetRank() == 0)
        {
            std::cout << "Hartmann braking enabled (Sommeria & Moreau 1982, "
                         "eq. 18):"
                      << std::endl
                      << "  |B0| = " << bmag << ", sigma = " << m_sigma
                      << ", nu = " << kinvis << ", rho = 1" << std::endl
                      << "  gap a = " << gap << ", Hartmann walls n_w = "
                      << nWalls << std::endl
                      << "  Ha(a) = |B0| a sqrt(sigma/nu) = " << haGap
                      << ", delta_H = a/Ha = " << deltaH << std::endl
                      << "  chi = n_w sqrt(sigma |B0|^2 nu)/a = " << chi
                      << ", chi/(sigma |B0|^2) = " << chi / m_damping
                      << std::endl
                      << "  chi_i = (" << chiComps[0] << ", "
                      << chiComps[1] << ", " << chiComps[2] << ")"
                      << std::endl;
            NekDouble uref = -1.0;
            m_session->LoadParameter("Umax", uref, -1.0);
            if (uref <= 0.0)
            {
                m_session->LoadParameter("U", uref, -1.0);
            }
            if (uref > 0.0)
            {
                std::cout << "  N = sigma |B0|^2 a/(rho U) = "
                          << m_damping * gap / uref
                          << ", Da = chi a/U = " << chi * gap / uref
                          << " (U = " << uref << ")" << std::endl;
            }
            std::cout << "  model assumptions: insulating Hartmann walls "
                         "(c = 0); chi is valid only where delta_H is "
                         "unresolved by the near-wall mesh (compare delta_H "
                         "with the wall-normal spacing); braking acts only "
                         "on components perpendicular to B0."
                      << std::endl;
        }
    }
    else if (m_implicitDamping)
    {
        m_FluidEq->SetImplicitLorentzDamping(m_damping);
    }

    // Keep all three components of u x B and J, including the out-of-plane
    // component required by a two-dimensional flow with an in-plane field.
    m_Efield = Array<OneD, Array<OneD, NekDouble>>(3);
    for (size_t i = 0; i < 3; ++i)
    {
        m_Efield[i] = Array<OneD, NekDouble>(pFields[0]->GetTotPoints(), 0.);
    }
}

/**
 * @brief Adds the body force, -Omega X u.
 * @param fields
 * @param inarray
 * @param outarray
 * @param time
 */
void ForcingLorentz::v_Apply(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    [[maybe_unused]] const Array<OneD, Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray,
    [[maybe_unused]] const NekDouble &time)
{
    size_t physTot = fields[0]->GetTotPoints();
    if (m_spacedim == 2)
    {
        Vmath::Smul(physTot, m_B0[2], inarray[1], 1, m_Efield[0], 1);
        Vmath::Smul(physTot, -m_B0[2], inarray[0], 1, m_Efield[1], 1);
        Vmath::Svtsvtp(physTot, m_B0[1], inarray[0], 1, -m_B0[0], inarray[1], 1,
                       m_Efield[2], 1);
    }
    else if (m_spacedim == 3)
    {
        Vmath::Svtsvtp(physTot, m_B0[2], inarray[1], 1, -m_B0[1], inarray[2], 1,
                       m_Efield[0], 1);
        Vmath::Svtsvtp(physTot, -m_B0[2], inarray[0], 1, m_B0[0], inarray[2], 1,
                       m_Efield[1], 1);
        Vmath::Svtsvtp(physTot, m_B0[1], inarray[0], 1, -m_B0[0], inarray[1], 1,
                       m_Efield[2], 1);
    }
    for (size_t i = 0; i < 3; ++i)
    {
        Vmath::Sadd(physTot, m_E0[i], m_Efield[i], 1, m_Efield[i], 1);
    }
    m_FluidEq->SolveEfield(m_Efield, m_Efield);
    NekDouble SB0[3];
    SB0[0] = m_sigma * m_B0[0];
    SB0[1] = m_sigma * m_B0[1];
    SB0[2] = m_sigma * m_B0[2];
    if (m_spacedim == 2)
    {
        Vmath::Svtvp(physTot, SB0[2], m_Efield[1], 1, outarray[0], 1,
                     outarray[0], 1);
        Vmath::Svtvp(physTot, -SB0[1], m_Efield[2], 1, outarray[0], 1,
                     outarray[0], 1);
        Vmath::Svtvp(physTot, -SB0[2], m_Efield[0], 1, outarray[1], 1,
                     outarray[1], 1);
        Vmath::Svtvp(physTot, SB0[0], m_Efield[2], 1, outarray[1], 1,
                     outarray[1], 1);
    }
    else if (m_spacedim == 3)
    {
        Vmath::Svtvp(physTot, SB0[2], m_Efield[1], 1, outarray[0], 1,
                     outarray[0], 1);
        Vmath::Svtvp(physTot, -SB0[1], m_Efield[2], 1, outarray[0], 1,
                     outarray[0], 1);
        Vmath::Svtvp(physTot, -SB0[2], m_Efield[0], 1, outarray[1], 1,
                     outarray[1], 1);
        Vmath::Svtvp(physTot, SB0[0], m_Efield[2], 1, outarray[1], 1,
                     outarray[1], 1);
        Vmath::Svtvp(physTot, SB0[1], m_Efield[0], 1, outarray[2], 1,
                     outarray[2], 1);
        Vmath::Svtvp(physTot, -SB0[0], m_Efield[1], 1, outarray[2], 1,
                     outarray[2], 1);
    }

    if (m_implicitDamping)
    {
        for (int i = 0; i < m_spacedim; ++i)
        {
            // Remove the -sigma |B|^2 u contribution from (u x B) x B.
            // This term is instead included in the velocity Helmholtz solve.
            Vmath::Svtvp(physTot, m_damping, inarray[i], 1, outarray[i], 1,
                         outarray[i], 1);
        }
    }
}

} // namespace Nektar::SolverUtils
