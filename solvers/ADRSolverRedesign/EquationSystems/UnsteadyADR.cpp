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

#include "Operators/TimeOps/TimeOp.hpp"
#include <Operators/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/PoissonSolve/PoissonSolveOp.hpp>

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
    : EquationSystem(pSession, pGraph)
{
}

/**
 * @brief Initialisation object for the UnsteadyADR problem.
 */
void UnsteadyADR::v_InitObject(bool DeclareFields)
{
    // Call to the initialisation object of EquationSystem
    EquationSystem::v_InitObject(DeclareFields);

    // Initialise boundary conditions
    SetBoundaryConditions(m_time);

    // Get variable strings and number of variables
    m_variables  = m_session->GetVariables();
    m_nVariables = m_variables.size();
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

    // Initialise Time-stepping operator
    m_timeOp = TimeOp<double>::Create(m_fields[0], m_variables);
    m_timeOp->DefineImplicit(&UnsteadyADR::DoImplicit, this);
    m_timeOp->DefineExplicitRhs(&UnsteadyADR::DoExplicitRhs, this);
    m_timeOp->DefineProjection(&UnsteadyADR::DoProjection, this);

    // Create and initialise all fields
    InitialiseFields();

    // Set advection velocity
    SetAdvectionVel();

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Create and initialise all operators
    InitialiseOperators();
}

/**
 * @brief Solves UnsteadyADR problem.
 */
void UnsteadyADR::v_DoSolve()
{
    // Set InitialConditions
    SetInitialConditionsField(m_in);

    // Time-stepping loop
    while (m_timeOp->GetStep() < m_steps ||
           m_timeOp->GetTime() < m_fintime - NekConstants::kNekZeroTol)
    {
        // Do time integration
        m_timeOp->Apply(m_in);
        m_time = m_timeOp->GetTime();

        // Verbose print
        if (m_infosteps && !(m_timeOp->GetStep() % m_infosteps))
        {
            std::cout
                // << std::scientific
                << "Steps: " << std::setw(8) << std::left << m_timeOp->GetStep()
                << " Time: " << std::setw(12) << std::left
                << m_timeOp->GetTime() << std::endl;
        }
    }

    // Keep EquationSystem time consistent with the time integrator state for
    // exact-solution evaluation and output metadata.
    m_time = m_timeOp->GetTime();

    // TODO : Remove the below code, when updated with Redesign solverUtils
    //  ----------------------------------------------------------------------
    // Write result into m_fields.m_phys for correct output to Fld file and
    // check against exact solution
    // Note outcoeffs has data arranged as [comp0, comp1, comp2] in a single
    // array
    Array<OneD, double> out = m_in.ToArray<double>();
    auto size               = 0;
    for (unsigned int i = 0; i < m_nVariables; i++)
    {
        // Get physical size for this field
        auto nPhys = m_fields[i]->GetTotPoints();

        // Copy result into m_fields.m_phys
        Array<OneD, double> out_var(nPhys);
        for (unsigned int j = 0; j < nPhys; j++)
        {
            out_var[j] = out[size + j];
        }

        m_fields[i]->SetPhys(out_var);
        m_fields[i]->SetPhysState(true);

        // Increment
        size += nPhys;
    }
    //--------------------------------------------------------------------------
}

void UnsteadyADR::v_GenerateSummary(SummaryList &s)
{
    SessionSummary(s);

    AddSummaryItem(s, "Integration Scheme",
                   m_session->GetTimeIntScheme().method);
    AddSummaryItem(s, "Integration Order",
                   std::to_string(m_session->GetTimeIntScheme().order));
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
    Field<double, FieldState::Phys> &in,
    [[maybe_unused]] Field<double, FieldState::Phys> &out,
    [[maybe_unused]] const double &time, const double &dt_inv_gamma)
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
                     PreconOp<double>::Create(m_fields[0],
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

            // Solve implicit ADR problem
            m_linearSystemOp->Apply(out, m_wsp_coeff);

            // Transform to physical space
            m_bwdTransOp->Apply(m_wsp_coeff, out);
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
void UnsteadyADR::DoExplicitRhs(Field<double, FieldState::Phys> &in,
                                Field<double, FieldState::Phys> &out,
                                const double &time, const double &dt)
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
                // To be completed
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

    // Add the forcing term, if defined.
    if (m_session->DefinesFunction("BodyForce"))
    {
        // Set forcing operator and evaluate
        m_forcingOp->SetTime(time);
        m_forcingOp->SetScale(dt);
        m_forcingOp->Apply(in, out);
    }

    // Zero output if no explicit term.
    if (!m_explicitAdvection && !m_explicitDiffusion &&
        !m_session->DefinesFunction("BodyForce"))
    {
        m_math.zero(out);
    }
}

/**
 * @brief Compute the projection for the unsteady ADR problem.
 *
 * @param in    Given fields.
 * @param out   CG-projected fields.
 * @param time  Time.
 */
void UnsteadyADR::DoProjection(Field<double, FieldState::Phys> &in,
                               Field<double, FieldState::Phys> &out,
                               [[maybe_unused]] const double time)
{
    // Update time-varying boundary conditions
    // SetBoundaryConditions(time);

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            // Discontinuous projection
            if (&in != &out)
            {
                m_math.copy(in, out);
            }
            break;
        }
        case MultiRegions::eGalerkin:
        {
            // Continuous projection
            // Note we could use the cheaper operators: AvgAssemble or
            // GlobalToLocal
            m_fwdTransOp->Apply(in, m_wsp_coeff);
            m_bwdTransOp->Apply(m_wsp_coeff, out);
            break;
        }
        default:
        {
            ASSERTL0(false, "Unsupported projection type.");
            break;
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
    if (m_diffusion)
    {
        // Get the scalar part
        m_session->LoadParameter("epsilon", m_epsilon, 1.0);

        // Set-up anisotropic diffusion coefficient
        // Default value for D_ij = 1, if i = j
        // Default value for D_ij = 0, if i != j
        const auto coordDim      = m_fields[0]->GetCoordim(0);
        const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
        m_diffCoeff.resize(diffCoeffSize);

        if (coordDim == 1)
        {
            m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
            // Multiply by Scalar Part
            m_diffCoeff[0] *= m_epsilon;
        }
        else if (coordDim == 2)
        {
            m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
            m_session->LoadParameter("D01", m_diffCoeff[1], 0.0);
            m_session->LoadParameter("D11", m_diffCoeff[2], 1.0);
            // Multiply by Scalar Part
            m_diffCoeff[0] *= m_epsilon;
            m_diffCoeff[1] *= m_epsilon;
            m_diffCoeff[2] *= m_epsilon;
        }
        else
        {
            m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
            m_session->LoadParameter("D01", m_diffCoeff[1], 0.0);
            m_session->LoadParameter("D11", m_diffCoeff[2], 1.0);
            m_session->LoadParameter("D02", m_diffCoeff[3], 0.0);
            m_session->LoadParameter("D12", m_diffCoeff[4], 0.0);
            m_session->LoadParameter("D22", m_diffCoeff[5], 1.0);
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
void UnsteadyADR::InitialiseOperators()
{
    // Initialise math helper
    std::string execName = Operator<double>::GetOpExecSpace(m_session);
    m_math               = Math(execName);

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            // Discontinuous projection
            m_fields[0]->GetTrace()->SetDataWarehouse();

            // Create operators
            if (m_advection)
            {
                m_advectionWeakDGOp =
                    AdvectionWeakDGOp<double>::Create(m_fields[0], m_variables);
                m_volumeFluxOp =
                    VolumeFluxOp<double>::Create(m_fields[0], m_variables);
                m_riemannSolverOp =
                    RiemannSolverOp<double>::Create(m_fields[0], m_variables);

                // Set volume flux and Riemann solver for advection operator
                m_advectionWeakDGOp->SetVolumeFluxOp(m_volumeFluxOp);
                m_advectionWeakDGOp->SetRiemannSolver(m_riemannSolverOp);

                // Check if forcing is defined
                if (m_session->DefinesFunction("AdvectionVelocity"))
                {
                    unsigned int coordDim = m_fields[0]->GetCoordim(0);

                    // Reads the Session File Velocity defined as function
                    std::vector<std::string> vel;
                    vel.push_back("Vx");
                    vel.push_back("Vy");
                    vel.push_back("Vz");
                    vel.resize(coordDim);

                    // Create operator
                    m_getFwdBwdTracePhysOp =
                        GetFwdBwdTracePhysOp<double>::Create(m_fields[0], vel);
                    m_getFwdBwdTracePhysOp->SetFwdOnly(true);

                    // Extract trace advection velocity for upwind solver
                    m_getFwdBwdTracePhysOp->Apply(m_advectionVel,
                                                  m_traceAdvectionVel,
                                                  m_traceAdvectionVel);
                }

                // Set advection velocity
                m_volumeFluxOp->SetAdvectVel(m_advectionVel);

                // Set trace advection velocity for upwind solver
                m_riemannSolverOp->SetTraceAdvVel(m_traceAdvectionVel);
            }
            if (m_diffusion)
            {
                // To be completed
            }
            break;
        }
        case MultiRegions::eGalerkin:
        {
            // Continuous projection
            m_bwdTransOp = BwdTransOp<double>::Create(
                m_fields[0], m_session->GetVariables());

            // Explicit operators
            if (m_explicitAdvection)
            {
                m_advectionCGOp =
                    AdvectionOp<double>::Create(m_fields[0], m_variables);
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
                    m_fields[0], m_session->GetVariables());
                auto linearADRSolveOp = LinearADRSolveOp<double>::Create(
                    m_fields[0], m_session->GetVariables());
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
                        m_fields[0], m_session->GetVariables(), "ConjGrad");
                }
                else
                {
                    m_linearSolverOp = LinearSolverOp<double>::Create(
                        m_fields[0], m_session->GetVariables());
                }
                auto helmSolveOp = HelmSolveOp<double>::Create(
                    m_fields[0], m_session->GetVariables());
                helmSolveOp->SetLinearSolver(m_linearSolverOp);
                helmSolveOp->SetDiffCoeff(m_diffCoeff);
                m_linearSystemOp = helmSolveOp;
            }

            // Projection operators
            m_fwdTransOp = FwdTransOp<double>::Create(
                m_fields[0], m_session->GetVariables());
            auto preconOp = PreconOp<double>::Create(m_fields[0],
                                                     m_session->GetVariables());
            // The projection solve is a mass-matrix problem, so use
            // ConjGrad regardless of the ADR system solver configured in
            // the session.
            auto linsolverOp = LinearSolverOp<double>::Create(
                m_fields[0], m_session->GetVariables(), "ConjGrad");
            m_fwdTransOp->SetLinearSolver(linsolverOp);
            m_fwdTransOp->SetPrecon(preconOp);
            m_fwdTransOp->UpdatePrecon();
            break;
        }
        default:
        {
            ASSERTL0(false, "Unsupported projection type.");
            break;
        }
    }

    // Check if forcing is defined
    if (m_session->DefinesFunction("BodyForce"))
    {
        // Create operator
        m_forcingOp = ExpressionOp<double>::Create(m_fields[0],
                                                   m_session->GetVariables());

        // Read initial conditions and configure operator
        std::vector<LibUtilities::EquationSharedPtr> forcingEquations;
        for (unsigned int i = 0; i < m_nVariables; ++i)
        {
            forcingEquations.push_back(m_session->GetFunction("BodyForce", i));
        }
        m_forcingOp->SetExpressions(forcingEquations);
        if (m_explicitAdvection)
        {
            m_forcingOp->SetAppend(true);
        }
        m_forcingOp->SetTime(m_time);
    }
}

/*
 *  Create and initialise Fields.
 */
void UnsteadyADR::InitialiseFields()
{
    unsigned int coordDim     = m_fields[0]->GetCoordim(0);
    unsigned int numHomoModes = m_npointsZ; // Note read in EquationSystem.cpp

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        // Discontinuous projection
        case MultiRegions::eDiscontinuous:
        {
            // Create blocks.
            auto block_attr_phys =
                GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
            auto block_attr_trace_phys =
                GetBlockAttributes<double, FieldState::Phys>(
                    m_fields[0]->GetTrace());

            // Create fields.
            m_in = Field<double, FieldState::Phys>("solution", block_attr_phys,
                                                   m_nVariables, numHomoModes);
            if (m_advection)
            {
                m_advectionVel = Field<double, FieldState::Phys>(
                    "advectionVel", block_attr_phys, coordDim, numHomoModes);
                m_traceAdvectionVel = Field<double, FieldState::Phys>(
                    "traceAdvectVel", block_attr_trace_phys, coordDim,
                    numHomoModes);
            }

            // Initialise fields
            break;
        }
        // Continuous field
        case MultiRegions::eGalerkin:
        {
            // Get block Attributes.
            auto block_attr_phys =
                GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
            auto block_attr_coeff =
                GetBlockAttributes<double, FieldState::Coeff>(m_fields[0]);

            // Create fields.
            m_in = Field<double, FieldState::Phys>("solution", block_attr_phys,
                                                   m_nVariables, numHomoModes);
            m_wsp_coeff = Field<double, FieldState::Coeff>(
                "m_wsp_coeff", block_attr_coeff, m_nVariables, numHomoModes);
            if (m_advection)
            {
                m_advectionVel = Field<double, FieldState::Phys>(
                    "advVel", block_attr_phys, coordDim, numHomoModes);
            }
            m_math.zero(m_wsp_coeff);
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
            unsigned int coordDim = m_fields[0]->GetCoordim(0);

            // Reads the Session File Velocity defined as function
            std::vector<std::string> vel;
            vel.push_back("Vx");
            vel.push_back("Vy");
            vel.push_back("Vz");
            vel.resize(coordDim);

            // Initialise operators
            auto expressionOp = ExpressionOp<double>::Create(m_fields[0], vel);

            // Read advection velocity expressions and configure operator
            std::vector<LibUtilities::EquationSharedPtr> advectionVelocities;
            for (unsigned int i = 0; i < coordDim; ++i)
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

void UnsteadyADR::SetInitialConditionsField(
    Field<double, FieldState::Phys> &field)
{
    // Print to log/console
    if (m_session->GetComm()->GetRank() == 0)
    {
        std::cout << "Initial Conditions:" << std::endl;
    }

    // Set initial conditions from session file
    if (m_session->DefinesFunction("InitialConditions"))
    {
        // Initialise operators
        auto initialOp = ExpressionOp<double>::Create(
            m_fields[0], m_session->GetVariables());

        // Read initial conditions and configure operator
        std::vector<LibUtilities::EquationSharedPtr> initialConditons;
        for (unsigned int i = 0; i < m_nVariables; ++i)
        {
            initialConditons.push_back(
                m_session->GetFunction("InitialConditions", i));
        }
        initialOp->SetExpressions(initialConditons);
        initialOp->SetTime(m_time);

        // Set initial conditions defined in session and update coefficients
        m_math.zero(field);
        initialOp->Apply(field, field);

        // Project initial conditions
        DoProjection(field, field, 0.0);

        // Print for initial conditions
        if (m_session->GetComm()->GetRank() == 0)
        {
            for (unsigned int i = 0; i < m_nVariables; ++i)
            {
                std::string varName = m_variables[i];
                std::cout << "  - Field " << varName << ": "
                          << GetFunction("InitialConditions")->Describe(varName)
                          << std::endl;
            }
        }
    }
    else
    {
        m_math.zero(field);

        for (unsigned int i = 0; i < m_nVariables; i++)
        {
            if (m_session->GetComm()->GetRank() == 0)
            {
                std::cout << "  - Field " << m_variables[i] << ": 0 (default)"
                          << std::endl;
            }
        }
    }
}

} // namespace Nektar
