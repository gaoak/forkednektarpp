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

#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondEnforceEntropyPressureCFE/BndCondEnforceEntropyPressureCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondEnforceEntropyTotalEnthalpyCFE/BndCondEnforceEntropyTotalEnthalpyCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondEnforceEntropyVelocityCFE/BndCondEnforceEntropyVelocityCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondExtrapOrder0CFE/BndCondExtrapOrder0CFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondPressureOutflowCFE/BndCondPressureOutflowCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondRiemannInvariantCFE/BndCondRiemannInvariantCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondSlipWallCFE/BndCondSlipWallCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondStagnationInflowCFE/BndCondStagnationInflowCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/BndCondOps/BndCondWallCFE/BndCondWallCFEOp.hpp>
#include <CompressibleFlowSolverRedesign/DiffusionCFEVolFlux/DiffusionCFEVolFluxOp.hpp>
#include <CompressibleFlowSolverRedesign/EquationSystems/NavierStokesCFE.h>
#include <CompressibleFlowSolverRedesign/EulerVolumeFlux/EulerVolumeFluxOp.hpp>

namespace Nektar
{
using namespace MultiRegions;

std::string NavierStokesCFE::className =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "NavierStokesCFE", NavierStokesCFE::create,
        "Navier-Stokes equations in conservative variables.");

NavierStokesCFE::NavierStokesCFE(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph)
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
 * dt * RHS rather than the unscaled RHS. The explicit conservative
 * Navier-Stokes RHS is
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
    // out = dt * (-advection + diffusion)
    m_advDiffusionOp->SetScale(dt);
    m_advDiffusionOp->Apply(in, out);
}

/*
 *  Create and initialise all operators for this solver
 */
void NavierStokesCFE::v_InitialiseOperators()
{
    EquationSystem::v_InitialiseOperators();

    std::string execName =
        MultiRegions::Operator<double>::GetOpExecSpace(m_session);

    // Create advection operators
    m_advDiffusionOp = AdvWeakDGDiffusionIPOp<double>::Create(
        m_expansionLists[0], m_variables);

    std::string riemannMethod = m_session->GetSolverInfo("UpwindType");

    std::string EqnOfState =
        boost::to_upper_copy(m_session->GetEquationOfState().type);

    std::string AdvDiffMethod = "AdvDiffTraceFluxCFE" + riemannMethod +
                                EqnOfState; // add prefix for local version
    auto advDiffusionTraceFluxOp = SolverCore::TraceFluxOp<double>::Create(
        m_expansionLists[0], m_variables, AdvDiffMethod);
    m_advDiffusionOp->SetAdvDiffTraceFlux(advDiffusionTraceFluxOp);

    auto eulerVolFluxOp =
        EulerVolumeFluxOp<double>::Create(m_expansionLists[0], m_variables);
    m_advDiffusionOp->SetAdvVolFlux(eulerVolFluxOp);

    auto diffusionVolFluxOp =
        DiffusionCFEVolFluxOp<double>::Create(m_expansionLists[0], m_variables);
    m_advDiffusionOp->SetDiffVolFlux(diffusionVolFluxOp);

    SetUpBoundaryConditions();
}

/**
 * @brief Attach the boundary conditions this system supports.
 *
 * Each operator claims only the regions carrying its tag and costs nothing
 * when there are none, so all of them are attached unconditionally.
 */
void NavierStokesCFE::SetUpBoundaryConditions()
{
    // Viscous wall states are functions of the interior trace rather than of
    // the session, so they are recomputed each apply and written into the same
    // boundary storage the trace flux gathers from. Attached unconditionally:
    // the operator claims only the regions tagged as walls and costs nothing
    // when there are none.
    auto bndCondWallOp =
        BndCondWallCFEOp<double>::Create(m_expansionLists[0], m_variables);
    m_advDiffusionOp->AddBndCondUpdateOp(bndCondWallOp);

    // Likewise for a pressure outflow, which extrapolates the interior state
    // and imposes only the static pressure. Each of these claims its own
    // regions by USERDEFINEDTYPE, so attaching both is not a conflict.
    auto bndCondPressureOutflowOp = BndCondPressureOutflowCFEOp<double>::Create(
        m_expansionLists[0], m_variables);
    m_advDiffusionOp->AddBndCondUpdateOp(bndCondPressureOutflowOp);

    // And for the entropy inflow, which is a condition on the inviscid state
    // and so belongs on both this path and the Euler one.
    auto bndCondEntropyVelocityOp =
        BndCondEnforceEntropyVelocityCFEOp<double>::Create(m_expansionLists[0],
                                                           m_variables);
    m_advDiffusionOp->AddBndCondUpdateOp(bndCondEntropyVelocityOp);

    // Its two siblings, which fill the same degree of freedom with the
    // pressure or the total enthalpy instead.
    auto bndCondEntropyPressureOp =
        BndCondEnforceEntropyPressureCFEOp<double>::Create(m_expansionLists[0],
                                                           m_variables);
    m_advDiffusionOp->AddBndCondUpdateOp(bndCondEntropyPressureOp);

    auto bndCondEntropyTotalEnthalpyOp =
        BndCondEnforceEntropyTotalEnthalpyCFEOp<double>::Create(
            m_expansionLists[0], m_variables);
    m_advDiffusionOp->AddBndCondUpdateOp(bndCondEntropyTotalEnthalpyOp);

    // An inviscid wall and a symmetry plane, which mirror the momentum, and a
    // zeroth order extrapolation, which claims its regions so they are seeded
    // with the interior state and then leaves them alone.
    auto bndCondSlipWallOp =
        BndCondSlipWallCFEOp<double>::Create(m_expansionLists[0], m_variables);
    m_advDiffusionOp->AddBndCondUpdateOp(bndCondSlipWallOp);

    auto bndCondExtrapOrder0Op = BndCondExtrapOrder0CFEOp<double>::Create(
        m_expansionLists[0], m_variables);
    m_advDiffusionOp->AddBndCondUpdateOp(bndCondExtrapOrder0Op);

    // The characteristic farfield, which takes its freestream from the session
    // parameters rather than from the region's boundary values.
    auto bndCondRiemannInvariantOp =
        BndCondRiemannInvariantCFEOp<double>::Create(m_expansionLists[0],
                                                     m_variables);
    m_advDiffusionOp->AddBndCondUpdateOp(bndCondRiemannInvariantOp);

    // And the reservoir inflow, whose session values are a stagnation state
    // and a flow direction rather than a conserved state.
    auto bndCondStagnationInflowOp =
        BndCondStagnationInflowCFEOp<double>::Create(m_expansionLists[0],
                                                     m_variables);
    m_advDiffusionOp->AddBndCondUpdateOp(bndCondStagnationInflowOp);
}

/**
 * @brief Supply the Courant estimate with the compressible wave speed.
 *
 * The conserved state is converted to velocity and sound speed on demand,
 * so a run that asks for no Courant estimate builds neither the operator
 * nor the field it writes into. The estimate this feeds is advective; the
 * diffusive time scale of the viscous terms does not enter it.
 */
LibUtilities::Field<double, FieldState::Phys> &NavierStokesCFE::
    v_GetCFLVelocityField()
{
    if (!m_cflVelocityOp)
    {
        // The kernel works in the shape dimension while the reduction reads
        // the coordinate dimension; they part company only on a manifold,
        // which the compressible solver does not support.
        ASSERTL0(m_expansionLists[0]->GetExp(0)->GetShapeDimension() ==
                     m_coordim,
                 "The Courant estimate expects the shape and coordinate "
                 "dimensions to agree.");

        std::vector<std::string> components;
        for (unsigned int i = 0; i < m_coordim; ++i)
        {
            components.push_back("u" + std::to_string(i));
        }
        components.push_back("c");

        m_cflVelocityOp =
            CFLVelocityCFEOp<double>::Create(m_expansionLists[0], components);

        auto bAtr_phys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[0]);

        m_cflVelocity = LibUtilities::Field<double, FieldState::Phys>(
            "cflVelocity", bAtr_phys, m_coordim + 1, m_npointsZ);
    }

    m_cflVelocityOp->Apply(m_fields, m_cflVelocity);

    return m_cflVelocity;
}

double NavierStokesCFE::v_GetSoundSpeedFactor()
{
    // The last component is a sound speed, so it counts in full.
    return 1.0;
}

} // namespace Nektar
