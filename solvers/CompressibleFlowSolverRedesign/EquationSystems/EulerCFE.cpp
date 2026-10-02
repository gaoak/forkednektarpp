///////////////////////////////////////////////////////////////////////////////
//
// File: EulerCFE.cpp
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
// Description: Euler equations in consƒervative variables without artificial
// diffusion
//
///////////////////////////////////////////////////////////////////////////////

#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondEnforceEntropyPressureCFE/BndCondEnforceEntropyPressureCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondEnforceEntropyTotalEnthalpyCFE/BndCondEnforceEntropyTotalEnthalpyCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondEnforceEntropyVelocityCFE/BndCondEnforceEntropyVelocityCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondExtrapOrder0CFE/BndCondExtrapOrder0CFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondPressureOutflowCFE/BndCondPressureOutflowCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondRiemannInvariantCFE/BndCondRiemannInvariantCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondSlipWallCFE/BndCondSlipWallCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondStagnationInflowCFE/BndCondStagnationInflowCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/EquationSystems/EulerCFE.h>
#include <CompressibleFlowSolverRedesign/EulerVolumeFlux/EulerVolumeFluxOp.hpp>
#include <iomanip>

namespace Nektar
{
using namespace Operators;

std::string EulerCFE::className =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "EulerCFE", EulerCFE::create,
        "Euler equations in conservative variables.");

EulerCFE::EulerCFE(const LibUtilities::SessionReaderSharedPtr &pSession,
                   const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph)
{
    ASSERTL0(m_projectionType == MultiRegions::eDiscontinuous,
             "The NavierStokesCFE is only implemented for projectionType "
             "Discontinuous");
}

/**
 * @brief Initialisation object for the Euler equations in conservative
 * variables.
 */
void EulerCFE::v_InitObject(bool declareExpansionLists)
{
    // Call to the initialisation object of EquationSystem
    UnsteadySystem::v_InitObject(declareExpansionLists);

    ASSERTL0(m_session->DefinesSolverInfo("UPWINDTYPE"),
             "No UPWINDTYPE defined in session.");

    /// Create Field for solution m_fields and others
    InitialiseFields();

    // Create and initialise all operators
    InitialiseOperators();

    // Configure Time-integration
    InitialiseTimeOp();
    m_timeOp->DefineExplicitRhs(&EulerCFE::DoAdvection, this);
}

void EulerCFE::v_GenerateSummary(SummaryList &s)
{
    UnsteadySystem::v_GenerateSummary(s);

    AddSummaryItem(s, "Equations", "Compressible Euler");
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

/*
 *  Setup rhs advection term.
 *
 *  Upon input
 *  param in: = u^{n}
 *  param out: = param in
 *  param time: = t^{n+1}
 *  param dt: = \Delta t
 *
 *  Upon output
 *  param in: = u^{n+1}
 *  param out: = param in
 */
void EulerCFE::DoAdvection(LibUtilities::Field<double, FieldState::Phys> &in,
                           LibUtilities::Field<double, FieldState::Phys> &out,
                           [[maybe_unused]] const double &time,
                           const double &dt)
{
    // Solve advection problem
    m_advectionWeakDGOp->SetScale(-dt);
    m_advectionWeakDGOp->Apply(in, out);
}

/*
 *  Create and initialise all operators for this solver
 */
void EulerCFE::v_InitialiseOperators()
{
    EquationSystem::v_InitialiseOperators();

    // Create operators
    m_advectionWeakDGOp =
        AdvectionWeakDGOp<double>::Create(m_expansionLists[0], m_variables);
    std::string execName =
        Operators::Operator<double>::GetOpExecSpace(m_session);
    std::string riemannMethod = m_session->GetSolverInfo("UpwindType");
    auto volumeFluxOp =
        EulerVolumeFluxOp<double>::Create(m_expansionLists[0], m_variables);

    // Set volume flux and for advection operator
    m_advectionWeakDGOp->SetVolumeFluxOp(volumeFluxOp);

    std::string EqnOfState =
        boost::to_upper_copy(m_session->GetEquationOfState().type);
    riemannMethod    = "AdvTraceFluxCFE" + riemannMethod + EqnOfState;
    auto traceFluxOp = SolverCore::TraceFluxOp<double>::Create(
        m_expansionLists[0], m_variables, riemannMethod, execName);

    // Set trace flux and Riemann solver for advection operator
    m_advectionWeakDGOp->SetTraceFlux(traceFluxOp);

    SetUpBoundaryConditions();
}

/**
 * @brief Attach the boundary conditions this system supports.
 *
 * Each operator claims only the regions carrying its tag and costs nothing
 * when there are none, so all of them are attached unconditionally.
 */
void EulerCFE::SetUpBoundaryConditions()
{
    // A pressure outflow extrapolates the interior state and imposes only the
    // static pressure, so it is a condition on the inviscid state and belongs
    // here as much as on the viscous path. Attached unconditionally: the
    // operator claims only the regions tagged PressureOutflow and costs nothing
    // when there are none.
    auto bndCondPressureOutflowOp = BndCondPressureOutflowCFEOp<double>::Create(
        m_expansionLists[0], m_variables);
    m_advectionWeakDGOp->AddBndCondUpdateOp(bndCondPressureOutflowOp);

    // A subsonic entropy inflow is likewise a condition on the inviscid state.
    auto bndCondEntropyVelocityOp =
        BndCondEnforceEntropyVelocityCFEOp<double>::Create(m_expansionLists[0],
                                                           m_variables);
    m_advectionWeakDGOp->AddBndCondUpdateOp(bndCondEntropyVelocityOp);

    // Its two siblings, which fill the same degree of freedom with the
    // pressure or the total enthalpy instead.
    auto bndCondEntropyPressureOp =
        BndCondEnforceEntropyPressureCFEOp<double>::Create(m_expansionLists[0],
                                                           m_variables);
    m_advectionWeakDGOp->AddBndCondUpdateOp(bndCondEntropyPressureOp);

    auto bndCondEntropyTotalEnthalpyOp =
        BndCondEnforceEntropyTotalEnthalpyCFEOp<double>::Create(
            m_expansionLists[0], m_variables);
    m_advectionWeakDGOp->AddBndCondUpdateOp(bndCondEntropyTotalEnthalpyOp);

    // An inviscid wall and a symmetry plane, which mirror the momentum, and a
    // zeroth order extrapolation, which claims its regions so they are seeded
    // with the interior state and then leaves them alone.
    auto bndCondSlipWallOp =
        BndCondSlipWallCFEOp<double>::Create(m_expansionLists[0], m_variables);
    m_advectionWeakDGOp->AddBndCondUpdateOp(bndCondSlipWallOp);

    auto bndCondExtrapOrder0Op = BndCondExtrapOrder0CFEOp<double>::Create(
        m_expansionLists[0], m_variables);
    m_advectionWeakDGOp->AddBndCondUpdateOp(bndCondExtrapOrder0Op);

    // The characteristic farfield, which takes its freestream from the session
    // parameters rather than from the region's boundary values.
    auto bndCondRiemannInvariantOp =
        BndCondRiemannInvariantCFEOp<double>::Create(m_expansionLists[0],
                                                     m_variables);
    m_advectionWeakDGOp->AddBndCondUpdateOp(bndCondRiemannInvariantOp);

    // And the reservoir inflow, whose session values are a stagnation state
    // and a flow direction rather than a conserved state.
    auto bndCondStagnationInflowOp =
        BndCondStagnationInflowCFEOp<double>::Create(m_expansionLists[0],
                                                     m_variables);
    m_advectionWeakDGOp->AddBndCondUpdateOp(bndCondStagnationInflowOp);
}

} // namespace Nektar
