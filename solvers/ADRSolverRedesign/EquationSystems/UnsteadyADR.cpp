///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyADR.cpp
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
// Description: UnsteadyADR problem solve routines
//
///////////////////////////////////////////////////////////////////////////////

#include <ADRSolverRedesign/DiffusionScalarIPTraceFlux/DiffusionScalarIPTraceFluxOp.hpp>
#include <ADRSolverRedesign/DiffusionScalarVolFlux/DiffusionScalarVolFluxOp.hpp>
#include <ADRSolverRedesign/ScalarTraceFlux/ScalarTraceFluxOp.hpp>
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp>
#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/PoissonSolve/PoissonSolveOp.hpp>
#include <SolverCore/TraceFlux/TraceFluxOp.hpp>

#include <ADRSolverRedesign/EquationSystems/UnsteadyADR.h>

namespace Nektar
{
using namespace MultiRegions;

std::string UnsteadyADR::className0 =
    GetEquationSystemFactory().RegisterCreatorFunction("UnsteadyAdvection",
                                                       UnsteadyADR::create);
std::string UnsteadyADR::className1 =
    GetEquationSystemFactory().RegisterCreatorFunction("UnsteadyDiffusion",
                                                       UnsteadyADR::create);
std::string UnsteadyADR::className2 =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "UnsteadyAdvectionDiffusion", UnsteadyADR::create);

UnsteadyADR::UnsteadyADR(const LibUtilities::SessionReaderSharedPtr &pSession,
                         const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph)
{
}

/**
 * @brief Initialisation object for the UnsteadyADR problem.
 */
void UnsteadyADR::v_InitObject(bool declareExpansionLists)
{
    // Call to the initialisation object of EquationSystem
    UnsteadySystem::v_InitObject(declareExpansionLists);

    /// Load parameters from session
    LoadSessionParameters();

    /// Create Fields for solution m_phys, m_coeff and others
    v_InitialiseFields();

    // Set advection velocity
    SetAdvectionVel();

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Create and initialise all operators
    v_InitialiseOperators();

    // Configure Time-integration
    m_timeOp = TimeOp<double>::Create(m_expansionLists[0], m_variables[0]);
    m_timeOp->DefineProjection(&UnsteadyADR::DoProjection, this);
    m_timeOp->DefineImplicit(&UnsteadyADR::DoImplicit, this);
    m_timeOp->DefineExplicitRhs(&UnsteadyADR::DoExplicitRhs, this);
}

void UnsteadyADR::v_GenerateSummary(SummaryList &s)
{
    UnsteadySystem::v_GenerateSummary(s);

    AddSummaryItem(s, "Equations", "Advection-Diffusion-Reaction");
    if (m_advection)
    {
        AddSummaryItem(s, "Advect. advancement",
                       m_explicitAdvection ? "explicit" : "implicit");
    }
    if (m_diffusion)
    {
        AddSummaryItem(s, "Diffusion advancement",
                       m_explicitDiffusion ? "explicit" : "implicit");
    }

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
 * @brief Project onto the solution space: a copy for DG, a forward and
 * backward transform with the Dirichlet values at time for CG.
 */
void UnsteadyADR::DoProjection(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out, const double &time)
{
    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            if (&in != &out)
            {
                m_math.copy(in, out);
            }
            break;
        }
        case MultiRegions::eGalerkin:
        {
            m_fwdTransOp->UpdateBndCoeffs(time);
            m_fwdTransOp->Apply(in, m_coeff[0]);
            m_bwdTransOp->Apply(m_coeff[0], out);
            break;
        }
        default:
        {
            ASSERTL0(false, "Unsupported projection type.");
            break;
        }
    }
}

void UnsteadyADR::DoImplicit(
    LibUtilities::Field<double, FieldState::Phys> &in,
    [[maybe_unused]] LibUtilities::Field<double, FieldState::Phys> &out,
    const double &time, const double &dt_inv_gamma)
{
    if (m_session->GetSolverInfo("EQTYPE") == "UnsteadyAdvection")
    {
        ASSERTL0(false,
                 "Implicit time-stepping not supported for pure advection");
        return;
    }

    if (m_isTimeDependentDiffCoeff)
    {
        m_varDiffCoeffFunction->Evaluate(m_varDiffCoeff, time);
        m_preconOp.clear();
    }

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        // Discontinuous projection
        case MultiRegions::eDiscontinuous:
        {
            ASSERTL0(false, "Implicit/IMEX time-stepping not supported for DG");
            if (m_implicitAdvection)
            {
                // To be completed
            }
            if (m_implicitDiffusion)
            {
                // To be completed
            }
            break;
        }
        // Continuous field
        case MultiRegions::eGalerkin:
        {
            // Update \lambda = \gamma / \Delta t
            m_lambda = 1.0 / dt_inv_gamma;

            // Update LinearSystem
            if (m_implicitAdvection && m_implicitDiffusion)
            {
                std::dynamic_pointer_cast<LinearADRSolveOp<double>>(
                    m_linearSystemOp)
                    ->SetLambda(m_lambda);
            }
            else if (m_implicitDiffusion)
            {
                std::dynamic_pointer_cast<HelmSolveOp<double>>(m_linearSystemOp)
                    ->SetLambda(m_lambda);
            }

            if (m_preconOp.find(dt_inv_gamma) == m_preconOp.end())
            {
                // Configure and cache preconditioner
                m_preconOp.insert(
                    {dt_inv_gamma,
                     PreconOp<double>::Create(m_expansionLists[0],
                                              m_session->GetVariables())});
                m_linearSystemOp->SetPrecon(m_preconOp[dt_inv_gamma]);
                m_linearSystemOp->UpdatePrecon();
            }
            else
            {
                // Re-use preconditioner
                m_linearSystemOp->SetPrecon(m_preconOp[dt_inv_gamma]);
            }

            // Multiply by negative lambda
            m_math.mul(-m_lambda, in, out);

            // Refresh time-dependent Dirichlet and Neumann coefficients at the
            // current implicit stage time before applying the linear system.
            m_linearSystemOp->UpdateBndCoeffs(time);

            // Solve implicit ADR problem
            m_linearSystemOp->Apply(out, m_coeff[0]);

            // Transform to physical space
            m_bwdTransOp->Apply(m_coeff[0], out);
            break;
        }
        default:
        {
            ASSERTL0(false, "Unsupported projection type.");
            break;
        }
    }
}

/*
 *  @brief Evaluate reaction and/or forcing terms.
 *
 *  Upon input
 *  param in: = u^{n}
 *
 *  Upon output
 *  param out: = \kappa u^{n}
 */
void UnsteadyADR::DoExplicitRhs(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out, const double &time,
    const double &dt)
{
    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        // Discontinuous projection
        case MultiRegions::eDiscontinuous:
        {
            // Add advection term
            if (m_explicitAdvection)
            {
                // Refresh time-dependent boundary values at the current stage
                // time, as the continuous branch does through the linear
                // system operator.
                m_advectionWeakDGOp->UpdateBndPhys(time);
                m_advectionWeakDGOp->SetScale(-dt);
                m_advectionWeakDGOp->Apply(in, out);
            }
            if (m_explicitDiffusion)
            {
                m_diffusionIPOp->UpdateBndPhys(time);
                m_diffusionIPOp->SetScale(dt);
                m_diffusionIPOp->SetAppend(m_explicitAdvection);
                m_diffusionIPOp->Apply(in, out);
            }
            break;
        }
        // Continuous field
        case MultiRegions::eGalerkin:
        {
            // Add advection term
            if (m_explicitAdvection)
            {
                m_advectionCGOp->SetScale(-dt);
                m_advectionCGOp->Apply(in, out);
            }
            if (m_explicitDiffusion)
            {
                // To be completed
            }
            break;
        }
        default:
        {
            ASSERTL0(false, "Unsupported projection type.");
            break;
        }
    }

    // Add forcing terms, if defined. The first forcing assigns when there is
    // no preceding explicit contribution; subsequent forcings append.
    const bool hasExplicitTerm = m_explicitAdvection || m_explicitDiffusion;
    for (unsigned int i = 0; i < m_forcing.size(); ++i)
    {
        m_forcing[i]->SetAppend(hasExplicitTerm || i > 0);
        m_forcing[i]->Apply(in, out, time, dt);
    }

    // Zero output if no explicit term.
    if (!hasExplicitTerm && m_forcing.empty())
    {
        m_math.zero(out);
    }
}

/**
 * @brief Offer the advection velocity to the Courant estimate.
 *
 * Without advection this field is never constructed, and
 * GetCFLTimeStep() skips the estimate when the returned field is not
 * instantiated.
 */
LibUtilities::Field<double, FieldState::Phys> &UnsteadyADR::
    v_GetCFLVelocityField()
{
    return m_advectionVel;
}

void UnsteadyADR::LoadSessionParameters()
{
    // Get variable strings and number of variables.
    if ((m_session->GetSolverInfo("EQTYPE") == "UnsteadyAdvection") ||
        (m_session->GetSolverInfo("EQTYPE") == "UnsteadyAdvectionDiffusion"))
    {
        m_advection = true;
        m_session->MatchSolverInfo("ADVECTIONADVANCEMENT", "Explicit",
                                   m_explicitAdvection, true);
        m_implicitAdvection = !m_explicitAdvection;
    }

    if ((m_session->GetSolverInfo("EQTYPE") == "UnsteadyDiffusion") ||
        (m_session->GetSolverInfo("EQTYPE") == "UnsteadyAdvectionDiffusion"))
    {
        m_diffusion = true;
        m_session->MatchSolverInfo("DIFFUSIONADVANCEMENT", "Explicit",
                                   m_explicitDiffusion, true);
        m_implicitDiffusion = !m_explicitDiffusion;
    }

    // Get diffusion parameters.
    if (m_diffusion)
    {
        // Resize diffCoeff vector.
        const auto nDiffCoeff = m_coordim * (m_coordim + 1) / 2;
        m_diffCoeff.resize(nDiffCoeff);

        m_session->LoadParameter("epsilon", m_epsilon, 1.0);
        if (m_coordim == 1)
        {
            m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
        }
        else if (m_coordim == 2)
        {
            m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
            m_session->LoadParameter("D01", m_diffCoeff[1], 0.0);
            m_session->LoadParameter("D11", m_diffCoeff[2], 1.0);
        }
        else if (m_coordim == 3)
        {
            m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
            m_session->LoadParameter("D01", m_diffCoeff[1], 0.0);
            m_session->LoadParameter("D11", m_diffCoeff[2], 1.0);
            m_session->LoadParameter("D02", m_diffCoeff[3], 0.0);
            m_session->LoadParameter("D12", m_diffCoeff[4], 0.0);
            m_session->LoadParameter("D22", m_diffCoeff[5], 1.0);
        }
    }
}

/*
 *  @brief Sets the Advection Veclocity
 */
void UnsteadyADR::SetAdvectionVel()
{
    if (m_advection)
    {
        // Read advection velocity from session.
        if (m_session->DefinesFunction("AdvectionVelocity"))
        {
            auto bAtr_phys =
                MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                    m_expansionLists[0]);
            m_advectionVel = LibUtilities::Field<double, FieldState::Phys>(
                "advVel", bAtr_phys, m_coordim, m_npointsZ);

            // Reads the Session File Velocity defined as function.
            std::vector<std::string> vel = {"Vx", "Vy", "Vz"};
            vel.resize(m_coordim);
            SessionFunction advectionVelocity(m_session, m_expansionLists[0],
                                              vel, "AdvectionVelocity");
            advectionVelocity.Evaluate(m_advectionVel, m_time);
        }
        else
        {
            NEKERROR(ErrorUtil::efatal, "Function 'AdvectionVelocity' was "
                                        "not defined in session file.")
        }
    }
}

/**
 * @brief Set the diffusion Coeff.
 * the diff. coeff., diffCoeff_{ij} := \epslion * D_{ij}
 * where,
 *  \epslion is scalar part
 *   D_{ij}  is a 2D tensor for enforcing anisotropy
 * Note, it is assumed that,  diffCoeff_{ij}  =  diffCoeff_{ji}
 *
 * Let, Vec be a Vector<double> storing a flatten \diffCoeff_{ij}, then
 * Case 1: For 1D
 *      Vec[0] = D_{00}
 * Case 2: For 2D
 *      Vec[0] = D_{00}
 *      Vec[1] = D_{01}
 *      Vec[2] = D_{11}
 * Case 3: For 3D
 *      Vec[0] = D_{00}
 *      Vec[1] = D_{01}
 *      Vec[2] = D_{11}
 *      Vec[3] = D_{02}
 *      Vec[4] = D_{12}
 *      Vec[5] = D_{22}
 */
void UnsteadyADR::SetDiffusionCoeff()
{
    // Optional physical tensor, in the same D00,D01,D11,D02,D12,D22 order
    // as the constant parameters. epsilon scales every tensor component.
    if (m_session->DefinesFunction("DiffusionCoefficient"))
    {
        ASSERTL0(m_implicitDiffusion,
                 "Variable DiffusionCoefficient requires implicit diffusion");

        std::vector<std::string> names = {"D00", "D01", "D11",
                                          "D02", "D12", "D22"};
        names.resize(m_coordim * (m_coordim + 1) / 2);
        auto bAtr_phys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[0]);
        m_varDiffCoeff = LibUtilities::Field<double, FieldState::Phys>(
            "DiffusionCoefficient", bAtr_phys, names.size(), m_npointsZ);
        m_varDiffCoeffFunction = std::make_shared<SessionFunction>(
            m_session, m_expansionLists[0], names, "DiffusionCoefficient");
        m_isTimeDependentDiffCoeff = !m_varDiffCoeffFunction->IsConstant();
        m_varDiffCoeffFunction->SetScale(m_epsilon);
        m_varDiffCoeffFunction->Evaluate(m_varDiffCoeff, m_time);
    }
    else if (m_diffusion)
    {
        if (m_coordim == 1)
        {
            // Multiply by Scalar Part
            m_diffCoeff[0] *= m_epsilon;
        }
        else if (m_coordim == 2)
        {
            // Multiply by Scalar Part
            m_diffCoeff[0] *= m_epsilon;
            m_diffCoeff[1] *= m_epsilon;
            m_diffCoeff[2] *= m_epsilon;
        }
        else if (m_coordim == 3)
        {
            // Multiply by Scalar Part
            m_diffCoeff[0] *= m_epsilon;
            m_diffCoeff[1] *= m_epsilon;
            m_diffCoeff[2] *= m_epsilon;
            m_diffCoeff[3] *= m_epsilon;
            m_diffCoeff[4] *= m_epsilon;
            m_diffCoeff[5] *= m_epsilon;
        }
    }
}

/*
 *  Create and initialise all operators for this solver
 */
void UnsteadyADR::v_InitialiseOperators()
{
    // Transforms, and the projection of the solution to coefficients for
    // output: a global mass matrix solve for CG, element local for DG.
    m_bwdTransOp =
        BwdTransOp<double>::Create(m_expansionLists[0], m_variables[0]);
    if (m_projectionType == eGalerkin)
    {
        m_fwdTransOp = CreateFwdTransOp(m_expansionLists[0], m_variables[0]);
    }
    else
    {
        m_iProductWRTBaseOp = IProductWRTBaseOp<double>::Create(
            m_expansionLists[0], m_variables[0]);
        m_multiplyByElmtInvMassOp = MultiplyByElmtInvMassOp<double>::Create(
            m_expansionLists[0], m_variables[0]);
    }

    // Switch on the projection type (Discontinuous or Continuous).
    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            // Create operators.
            if (m_advection)
            {
                m_advectionWeakDGOp = AdvectionWeakDGOp<double>::Create(
                    m_expansionLists[0], m_variables[0]);
                m_volumeFluxOp = LinearAdvVolumeFluxOp<double>::Create(
                    m_expansionLists[0], m_variables[0]);

                // Set volume flux and Riemann solver for advection operator
                m_advectionWeakDGOp->SetVolumeFluxOp(m_volumeFluxOp);

                m_traceFluxOp = ScalarTraceFluxOp<double>::Create(
                    m_expansionLists[0], m_variables[0]);
                m_advectionWeakDGOp->SetTraceFlux(m_traceFluxOp);

                // Set up trace advection velocity on trace for upwind
                // solver
                m_traceAdvVel = LibUtilities::Field<double, FieldState::Phys>(
                    "traceAdvVel",
                    MultiRegions::GetLocTraceBlockAttributes<double,
                                                             FieldState::Phys>(
                        m_expansionLists[0]),
                    m_coordim, m_npointsZ);

                // extract advection trace
                auto physTraceExtractOp = PhysTraceExtractOp<double>::Create(
                    m_expansionLists[0], m_advectionVel.GetComponentNames());
                physTraceExtractOp->Apply(m_advectionVel, m_traceAdvVel);
                // the trace flux reads it through a pointer from here on
                m_traceFluxOp->SetTraceAdvVel(m_traceAdvVel);

                // Set advection velocity in volume flux
                m_volumeFluxOp->SetAdvVel(m_advectionVel);
            }
            if (m_diffusion)
            {
                m_diffusionIPOp = DiffusionIPOp<double>::Create(
                    m_expansionLists[0], m_variables[0]);
                auto diffusionScalarVolFluxOp =
                    DiffusionScalarVolFluxOp<double>::Create(
                        m_expansionLists[0], m_variables[0]);

                auto diffusionScalarIPTraceFluxOp =
                    DiffusionScalarIPTraceFluxOp<double>::Create(
                        m_expansionLists[0], m_variables[0]);

                diffusionScalarVolFluxOp->SetDiffCoeff(m_diffCoeff);
                diffusionScalarIPTraceFluxOp->SetDiffCoeff(m_diffCoeff);

                m_diffusionIPOp->SetVolumeFluxOp(diffusionScalarVolFluxOp);
                m_diffusionIPOp->SetTraceFluxOp(diffusionScalarIPTraceFluxOp);
            }
            break;
        }
        case MultiRegions::eGalerkin:
        {
            // Explicit operators.
            if (m_explicitAdvection)
            {
                m_advectionCGOp = AdvectionOp<double>::Create(
                    m_expansionLists[0], m_variables[0]);
                m_advectionCGOp->SetAdvVel(m_advectionVel);
            }
            if (m_explicitDiffusion)
            {
                // To be completed
            }

            // Implicit operators.
            if (m_implicitAdvection && m_implicitDiffusion)
            {
                m_linearSolverOp = LinearSolverOp<double>::Create(
                    m_expansionLists[0], m_session->GetVariables());
                auto linearADRSolveOp = LinearADRSolveOp<double>::Create(
                    m_expansionLists[0], m_session->GetVariables());
                linearADRSolveOp->SetLinearSolver(m_linearSolverOp);
                if (m_varDiffCoeffFunction)
                {
                    linearADRSolveOp->SetVarDiffCoeff(m_varDiffCoeff);
                }
                else
                {
                    linearADRSolveOp->SetDiffCoeff(m_diffCoeff);
                }
                linearADRSolveOp->SetAdvVel(m_advectionVel);
                m_linearSystemOp = linearADRSolveOp;
            }
            else if (m_implicitDiffusion)
            {
                if (m_explicitAdvection)
                {
                    m_linearSolverOp = LinearSolverOp<double>::Create(
                        m_expansionLists[0], m_session->GetVariables(),
                        "ConjGrad");
                }
                else
                {
                    m_linearSolverOp = LinearSolverOp<double>::Create(
                        m_expansionLists[0], m_session->GetVariables());
                }
                auto helmSolveOp = HelmSolveOp<double>::Create(
                    m_expansionLists[0], m_session->GetVariables());
                helmSolveOp->SetLinearSolver(m_linearSolverOp);
                if (m_varDiffCoeffFunction)
                {
                    helmSolveOp->SetVarDiffCoeff(m_varDiffCoeff);
                }
                else
                {
                    helmSolveOp->SetDiffCoeff(m_diffCoeff);
                }
                m_linearSystemOp = helmSolveOp;
            }
            break;
        }
        default:
        {
            ASSERTL0(false, "Unsupported projection type.");
            break;
        }
    }

    // Load forcing terms, if defined in the session file.
    m_forcing = Forcing::Load(m_session, m_expansionLists[0], m_variables[0]);
}

CoefficientFieldViews UnsteadyADR::v_GetOutputFieldViews()
{
    // The solution is advanced in physical space, so project back to
    // coefficients before exposing it to output filters. CG uses its global
    // transform with boundary coefficients; DG uses the element-local L2
    // projection required by the restart path.
    if (m_projectionType == MultiRegions::eGalerkin)
    {
        m_fwdTransOp->UpdateBndCoeffs(m_time);
        m_math.zero(m_coeff[0]);
        m_fwdTransOp->Apply(m_phys[0], m_coeff[0]);
    }
    else if (m_projectionType == MultiRegions::eDiscontinuous)
    {
        m_iProductWRTBaseOp->Apply(m_phys[0], m_wsp_coeff[0]);
        m_multiplyByElmtInvMassOp->Apply(m_wsp_coeff[0], m_coeff[0]);
    }

    return UnsteadySystem::v_GetOutputFieldViews();
}

} // namespace Nektar
