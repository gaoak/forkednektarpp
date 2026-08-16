///////////////////////////////////////////////////////////////////////////////
//
// File: NavierStokesCFE.cpp
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
// Description: Navier-Stokes equations in conservative variables without
// artificial diffusion
//
///////////////////////////////////////////////////////////////////////////////

#include <CompressibleFlowSolverRedesign/EquationSystems/NavierStokesCFE.h>

namespace Nektar
{
using namespace Operators;

std::string NavierStokesCFE::className =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "NavierStokesCFE", NavierStokesCFE::create,
        "Navier-Stokes equations in conservative variables.");

NavierStokesCFE::NavierStokesCFE(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), m_gamma(1.4)
{
    ASSERTL0(m_projectionType == MultiRegions::eDiscontinuous,
             "The NavierStokesCFE is only implemented for projectionType "
             "Discontinuous");
}

/**
 * @brief Initialisation object for the Navier-Stokes equations in conservative
 * variables.
 */
void NavierStokesCFE::v_InitObject(bool declareExpansionLists)
{
    // Call to the initialisation object of EquationSystem
    UnsteadySystem::v_InitObject(declareExpansionLists);

    ASSERTL0(m_session->DefinesSolverInfo("UPWINDTYPE"),
             "No UPWINDTYPE defined in session.");

    /// Create Field for solution m_fields and others
    InitialiseFields();

    // Load physical parameters
    InitialiseParameters();

    // Create and initialise all operators
    InitialiseOperators();

    // Configure Time-integration
    InitialiseTimeOp();
    m_timeOp->DefineExplicitRhs(&NavierStokesCFE::DoOdeRhs, this);
}

void NavierStokesCFE::v_GenerateSummary(SummaryList &s)
{
    UnsteadySystem::v_GenerateSummary(s);

    AddSummaryItem(s, "Equations", "Compressible Navier-Stokes");
    AddSummaryItem(s, "Formulation", "Explicit");

    std::stringstream ss;
    ss << R"(
 _   _      _    _
| \ | |    | |  | |              _     _
|  \| | ___| | _| |_ __ _ _ __ _| |_ _| |_
| . ` |/ _ \ |/ / __/ _` | '__|_   _|_   _|
| |\  |  __/   <| || (_| | |    |_|   |_|
\_| \_/\___|_|\_\\__\__,_|_|
              _           _                      _
             | |         (_)                    | |
 _ __ ___  __| | ___  ___ _  __ _ _ __   ___  __| |
| '__/ _ \/ _` |/ _ \/ __| |/ _` | '_ \ / _ \/ _` |
| | |  __/ (_| |  __/\__ \ | (_| | | | |  __/ (_| |
|_|  \___|\__,_|\___||___/_|\__, |_| |_|\___|\__,_|
                             __/ |
                            |___/
)";
    AddSummaryItem(s, "Redesign disclaimer", ss.str());
}

/**
 * @brief Assemble the explicit Navier-Stokes increment.
 *
 * The field-based TimeOp stores stage increments, so this routine returns
 * dt * RHS rather than the unscaled RHS used by the legacy time-integration
 * interface.  The explicit conservative Navier-Stokes RHS is
 *
 *     RHS = -advection + diffusion.
 *
 * The diffusion operator is applied in append mode so its final backward
 * transform accumulates directly into the advective contribution.
 */
void NavierStokesCFE::DoOdeRhs(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out,
    [[maybe_unused]] const double &time, const double &dt)
{
    // out = -dt * advection.
    m_advectionWeakDGOp->SetScale(-dt);
    m_advectionWeakDGOp->Apply(in, out);

    // out += dt * diffusion.
    m_diffusionIPOp->SetScale(dt);
    m_diffusionIPOp->SetAppend(true);
    m_diffusionIPOp->Apply(in, out);
}

/*
 *  Create and initialise all operators for this solver
 */
void NavierStokesCFE::v_InitialiseOperators()
{
    EquationSystem::v_InitialiseOperators();

    // Create advection operators
    m_advectionWeakDGOp =
        AdvectionWeakDGOp<double>::Create(m_expansionLists[0], m_variables);
    std::string execName      = Operator<double>::GetOpExecSpace(m_session);
    std::string riemannMethod = m_session->GetSolverInfo("UpwindType");

    // Euler trace and volume ops
    m_riemannSolverOp = CompressibleSolverOp<double>::Create(
        m_expansionLists[0], m_variables, riemannMethod, execName);
    m_eulerVolFluxOp =
        EulerVolumeFluxOp<double>::Create(m_expansionLists[0], m_variables);

    // Set volume flux and Riemann solver for advection operator
    m_advectionWeakDGOp->SetVolumeFluxOp(m_eulerVolFluxOp);
    m_advectionWeakDGOp->SetRiemannSolver(m_riemannSolverOp);

    // Diffusion volume and trace ops
    m_diffusionVolFluxOp =
        DiffusionCFEVolFluxOp<double>::Create(m_expansionLists[0], m_variables);
    m_diffusionTraceFluxOp = DiffusionCFETraceFluxOp<double>::Create(
        m_expansionLists[0], m_variables);

    // Create diffusion operator
    m_diffusionIPOp =
        DiffusionIPOp<double>::Create(m_expansionLists[0], m_variables);
    m_diffusionIPOp->SetVolumeFluxOp(m_diffusionVolFluxOp);
    m_diffusionIPOp->SetTraceFluxOp(m_diffusionTraceFluxOp);
}

/**
 * @brief Load CFS parameters from the session file.
 */
void NavierStokesCFE::InitialiseParameters()
{
    // Get gamma parameter from session file.
    m_session->LoadParameter("Gamma", m_gamma, 1.4);
}

} // namespace Nektar
