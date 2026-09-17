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
#include <ADRSolverRedesign/DiffusionScalarIPVolFlux/DiffusionScalarIPVolFluxOp.hpp>
#include <Operators/ElmtOps/Expression/ExpressionOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/PoissonSolve/PoissonSolveOp.hpp>

#include <ADRSolverRedesign/EquationSystems/UnsteadyADR.h>

namespace Nektar
{
using namespace Operators;

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
    InitialiseParameters();

    /// Create Field for solution m_fields and others
    InitialiseFields();

    // Set advection velocity
    SetAdvectionVel();

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Create and initialise all operators
    InitialiseOperators();

    // Configure Time-integration
    InitialiseTimeOp();
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
            m_linearSystemOp->Apply(out, m_fields_coeff);

            // Transform to physical space
            m_bwdTransOp->Apply(m_fields_coeff, out);
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
                m_advectionWeakDGOp->SetScale(-dt);
                m_advectionWeakDGOp->Apply(in, out);
            }
            if (m_explicitDiffusion)
            {
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
    if (m_diffusion)
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
        else
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
    EquationSystem::v_InitialiseOperators();

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            // Create operators
            if (m_advection)
            {
                m_advectionWeakDGOp = AdvectionWeakDGOp<double>::Create(
                    m_expansionLists[0], m_variables);
                m_volumeFluxOp = LinearAdvVolumeFluxOp<double>::Create(
                    m_expansionLists[0], m_variables);
                m_riemannSolverOp = RiemannSolverOp<double>::Create(
                    m_expansionLists[0], m_variables);

                // Set volume flux and Riemann solver for advection operator
                m_advectionWeakDGOp->SetVolumeFluxOp(m_volumeFluxOp);
                m_advectionWeakDGOp->SetRiemannSolver(m_riemannSolverOp);

                // Check if forcing is defined
                if (m_session->DefinesFunction("AdvectionVelocity"))
                {
                    // Reads the Session File Velocity defined as function
                    std::vector<std::string> vel;
                    vel.push_back("Vx");
                    vel.push_back("Vy");
                    vel.push_back("Vz");
                    vel.resize(m_coordim);

                    // Create operator
                    m_getFwdBwdTracePhysOp =
                        GetFwdBwdTracePhysOp<double>::Create(
                            m_expansionLists[0], vel);
                    m_getFwdBwdTracePhysOp->SetFwdOnly(true);

                    // Extract trace advection velocity for upwind solver
                    m_getFwdBwdTracePhysOp->Apply(m_advectionVel,
                                                  m_traceAdvectionVel,
                                                  m_traceAdvectionVel);
                }

                // Set advection velocity
                m_volumeFluxOp->SetAdvVel(m_advectionVel);

                // Set trace advection velocity for upwind solver
                m_riemannSolverOp->SetTraceAdvVel(m_traceAdvectionVel);
            }
            if (m_diffusion)
            {
                m_diffusionIPOp = DiffusionIPOp<double>::Create(
                    m_expansionLists[0], m_variables);
                auto diffusionScalarIPVolFluxOp =
                    DiffusionScalarIPVolFluxOp<double>::Create(
                        m_expansionLists[0], m_variables);
                auto diffusionScalarIPTraceFluxOp =
                    DiffusionScalarIPTraceFluxOp<double>::Create(
                        m_expansionLists[0], m_variables);

                diffusionScalarIPVolFluxOp->SetDiffCoeff(m_diffCoeff);
                diffusionScalarIPTraceFluxOp->SetDiffCoeff(m_diffCoeff);

                m_diffusionIPOp->SetVolumeFluxOp(diffusionScalarIPVolFluxOp);
                m_diffusionIPOp->SetTraceFluxOp(diffusionScalarIPTraceFluxOp);
            }
            break;
        }
        case MultiRegions::eGalerkin:
        {
            // Explicit operators
            if (m_explicitAdvection)
            {
                m_advectionCGOp = AdvectionOp<double>::Create(
                    m_expansionLists[0], m_variables);
                m_advectionCGOp->SetAdvVel(m_advectionVel);
            }
            if (m_explicitDiffusion)
            {
                // To be completed
            }

            // Implicit operators
            if (m_implicitAdvection && m_implicitDiffusion)
            {
                m_linearSolverOp = LinearSolverOp<double>::Create(
                    m_expansionLists[0], m_session->GetVariables());
                auto linearADRSolveOp = LinearADRSolveOp<double>::Create(
                    m_expansionLists[0], m_session->GetVariables());
                linearADRSolveOp->SetLinearSolver(m_linearSolverOp);
                linearADRSolveOp->SetDiffCoeff(m_diffCoeff);
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
                helmSolveOp->SetDiffCoeff(m_diffCoeff);
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
    m_forcing = Forcing::Load(m_session, m_expansionLists[0], m_variables);
}

/*
 *  Create and initialise Fields.
 */
void UnsteadyADR::v_InitialiseFields()
{
    // Initialise solution "m_fields" via EquationSystem routine
    EquationSystem::v_InitialiseFields();

    unsigned int numHomoModes = m_npointsZ; // Note read in EquationSystem.cpp

    // Get interleave width
    auto interleaveWidth =
        Operator<double>::GetDefaultInterleaveWidth(m_session);

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        // Discontinuous projection
        case MultiRegions::eDiscontinuous:
        {
            // Create blocks.
            auto bAtr_phys =
                MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                    m_expansionLists[0], interleaveWidth);
            auto bAtr_phys_trace =
                MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                    m_expansionLists[0]->GetTrace(), interleaveWidth);

            if (m_advection)
            {
                m_advectionVel = LibUtilities::Field<double, FieldState::Phys>(
                    "advectionVel", bAtr_phys, m_coordim, numHomoModes);
                m_traceAdvectionVel =
                    LibUtilities::Field<double, FieldState::Phys>(
                        "traceAdvectVel", bAtr_phys_trace, m_coordim,
                        numHomoModes);
            }
            break;
        }
        // Continuous field
        case MultiRegions::eGalerkin:
        {
            // Get block Attributes.
            auto bAtr_phys =
                MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                    m_expansionLists[0], interleaveWidth);
            auto bAtr_coeff =
                MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
                    m_expansionLists[0], interleaveWidth);

            // Create fields.
            if (m_advection)
            {
                m_advectionVel = LibUtilities::Field<double, FieldState::Phys>(
                    "advVel", bAtr_phys, m_coordim, numHomoModes);
            }
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
 *  @brief Sets the Advection Veclocity
 */
void UnsteadyADR::SetAdvectionVel()
{
    if (m_advection)
    {
        // Read advection velocity from session
        if (m_session->DefinesFunction("AdvectionVelocity"))
        {
            // Reads the Session File Velocity defined as function
            std::vector<std::string> vel;
            vel.push_back("Vx");
            vel.push_back("Vy");
            vel.push_back("Vz");
            vel.resize(m_coordim);

            // Initialise operators
            auto expressionOp =
                ExpressionOp<double>::Create(m_expansionLists[0], vel);

            // Read advection velocity expressions and configure operator
            std::vector<LibUtilities::EquationSharedPtr> advectionVelocities;
            for (unsigned int i = 0; i < m_coordim; ++i)
            {
                advectionVelocities.push_back(
                    m_session->GetFunction("AdvectionVelocity", vel[i]));
            }
            expressionOp->SetExpressions(advectionVelocities);
            expressionOp->SetTime(m_time);

            // Initialise m_advectionVel, evaluate all expressions and
            // transform to array
            m_math.zero(m_advectionVel);
            expressionOp->Apply(m_advectionVel, m_advectionVel);
        }
        else
        {
            NEKERROR(ErrorUtil::efatal, "Function 'AdvectionVelocity' was "
                                        "not defined in session file.")
        }
    }
}

void UnsteadyADR::InitialiseParameters()
{
    // Get variable strings and number of variables
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

    // Get diffusion parameters
    if (m_diffusion)
    {
        // Resize diffCoeff vector
        const auto diffCoeffSize = m_coordim * (m_coordim + 1) / 2;
        m_diffCoeff.resize(diffCoeffSize);

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
        else
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

} // namespace Nektar
