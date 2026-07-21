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
// Description: Solving the absolute flow in a moving body frame,
// by adding (U0 + Omega X (x - x0)) . grad u - Omega X u
// as the body force.
// U0 is the translational velocity of the body frame.
// Omega is the angular velocity.
// x0 is the rotation pivot in the body frame.
// All vectors use the basis of the body frame.
// Translational motion is allowed for all dimensions.
// Rotation is not allowed for 1D, 2DH1D, 3DH2D.
// Rotation in z direction is allowed for 2D and 3DH1D.
// Rotation in 3 directions are allowed for 3D.
// TODO: add suport for 3D rotation using Quaternion
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/Forcing/ForcingLorentz.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ExpList.h>
#include <SolverUtils/Filters/FilterInterfaces.hpp>

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
    [[maybe_unused]] const TiXmlElement *pForce)
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
    std::string electromagneticFields = "ElectricMagneticFields";
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
}

} // namespace Nektar::SolverUtils
