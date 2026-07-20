///////////////////////////////////////////////////////////////////////////////
//
// File: VCSQuasiStaticMHD.cpp
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
// Description: Velocity Correction Scheme for fluid-structure interaction
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/EquationSystems/VCSQuasiStaticMHD.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <SolverUtils/Core/Misc.h>
#include <boost/algorithm/string.hpp>

namespace Nektar
{
std::string VCSQuasiStaticMHD::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "VCSQuasiStaticMHD", VCSQuasiStaticMHD::create);

std::string VCSQuasiStaticMHD::solverTypeLookupId =
    LibUtilities::SessionReader::RegisterEnumValue(
        "SolverType", "VCSQuasiStaticMHD", eVCSQuasiStaticMHD);

/**
 * Constructor. Creates ...
 *
 * \param
 * \param
 */
VCSQuasiStaticMHD::VCSQuasiStaticMHD(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph),
      VelocityCorrectionScheme(pSession, pGraph)
{
}

void VCSQuasiStaticMHD::v_InitObject(bool DeclareField)
{
    VelocityCorrectionScheme::v_InitObject(DeclareField);
    // Set m_pressure to point to last field of m_fields;
    if (m_session->DefinesParameter("numConvectiveFields"))
    {
        ASSERTL0(m_nConvectiveFields == m_fields.size() - 2,
                 "Need to set up electric potential  and pressure fields "
                 "definition");
        m_potential = m_fields[m_fields.size() - 2];
    }
    else
    {
        ASSERTL0(false, "Need to define numConvectiveFields.");
    }
}

/**
 * Solve total electric field
 */
void VCSQuasiStaticMHD::SolveEfield(
    const Array<OneD, Array<OneD, NekDouble>> &movEfield,
    Array<OneD, Array<OneD, NekDouble>> &totEfield)
{
    int physTot = m_fields[0]->GetTotPoints();
    m_fields[0]->PhysDeriv(MultiRegions::eX, movEfield[0], m_F[0]);
    for (size_t i = 1; i < m_spacedim; ++i)
    {
        // Use Forcing[1] as storage since it is not needed for the pressure
        m_fields[i]->PhysDeriv(MultiRegions::DirCartesianMap[i], movEfield[i],
                               m_F[1]);
        Vmath::Vadd(physTot, m_F[1], 1, m_F[0], 1, m_F[0], 1);
    }
    StdRegions::ConstFactorMap factors;
    // Setup coefficient for equation
    factors[StdRegions::eFactorLambda] = 0.0;

    // Solver electric potential Poisson Equation
    m_potential->HelmSolve(m_F[0], m_potential->UpdateCoeffs(), factors);
    m_potential->BwdTrans(m_potential->GetCoeffs(), m_potential->UpdatePhys());
    if (m_spacedim == 2)
    {
        m_potential->PhysDeriv(m_potential->GetPhys(), m_F[0], m_F[1]);
    }
    else if (m_spacedim == 3)
    {
        m_potential->PhysDeriv(m_potential->GetPhys(), m_F[0], m_F[1], m_F[2]);
    }
    for (size_t i = 0; i < m_spacedim; ++i)
    {
        Vmath::Vsub(physTot, movEfield[i], 1, m_F[i], 1, totEfield[i], 1);
    }
}

/**
 * Destructor
 */
VCSQuasiStaticMHD::~VCSQuasiStaticMHD(void)
{
}

} // namespace Nektar
