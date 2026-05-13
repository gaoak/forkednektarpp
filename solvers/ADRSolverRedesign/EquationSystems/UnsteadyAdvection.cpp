///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyAdvection.cpp
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
// Description: Unsteady linear advection solve routines
//
///////////////////////////////////////////////////////////////////////////////

#include "Operators/TimeOps/TimeOp.hpp"

#include <ADRSolverRedesign/EquationSystems/UnsteadyAdvection.h>
#include <LibUtilities/BasicUtils/Timer.h>

namespace Nektar
{
using namespace Operators;

std::string UnsteadyAdvection::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "UnsteadyAdvection", UnsteadyAdvection::create,
        "Unsteady Advection equation.");

UnsteadyAdvection::UnsteadyAdvection(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph), m_nVariables(1)
{
}

/**
 * @brief Initialisation object for the unsteady linear advection equation.
 */
void UnsteadyAdvection::v_InitObject(bool DeclareFields)
{
    // Call to the initialisation object of EquationSystem
    EquationSystem::v_InitObject(DeclareFields);

    // Initialise boundary conditions
    SetBoundaryConditions(m_time);

    // Get variable strings and number of variables
    m_variables  = m_session->GetVariables();
    m_nVariables = m_variables.size();

    // Initialise Time-stepping operator
    m_timeOp = TimeOp<double>::Create(m_fields[0], m_variables);
    m_timeOp->DefineExplicitRhs(&UnsteadyAdvection::DoAdvection, this);
    m_timeOp->DefineProjection(&UnsteadyAdvection::DoProjection, this);

    // Create and initialise all fields
    InitialiseFields();

    // Set Advection Velocity
    SetAdvectionVel();

    // Create and initialise all operators
    InitialiseOperators();
}

/**
 * @brief Explicit solution of the unsteady advection problem.
 * Using a standalone time stepping loop.
 */
void UnsteadyAdvection::v_DoSolve()
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
    // Write result into m_fields.m_coeffs for correct output to Fld file and
    // check against exact solution
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

void UnsteadyAdvection::v_GenerateSummary(SummaryList &s)
{
    SessionSummary(s);

    AddSummaryItem(s, "Integration Scheme",
                   m_session->GetTimeIntScheme().method);
    AddSummaryItem(s, "Integration Order",
                   std::to_string(m_session->GetTimeIntScheme().order));

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
void UnsteadyAdvection::DoAdvection(Field<double, FieldState::Phys> &in,
                                    Field<double, FieldState::Phys> &out,
                                    [[maybe_unused]] const double &time,
                                    [[maybe_unused]] const double &dt)
{
    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        // Discontinuous projection
        case MultiRegions::eDiscontinuous:
        {
            // Solve advection problem
            m_advectionWeakDGOp->SetScale(-dt);
            m_advectionWeakDGOp->Apply(in, out);
            break;
        }
        // Continuous field
        case MultiRegions::eGalerkin:
        {
            m_advectionCGOp->SetScale(-dt);
            m_advectionCGOp->Apply(in, out);
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
 * @brief Compute the projection for the unsteady advection problem.
 *
 * @param in    Given fields.
 * @param out   CG-projected fields.
 * @param time  Time.
 */
void UnsteadyAdvection::DoProjection(Field<double, FieldState::Phys> &in,
                                     Field<double, FieldState::Phys> &out,
                                     const double time)
{
    // Update time-varying boundary conditions
    SetBoundaryConditions(time);

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

/*
 *  Create and initialise all operators for this solver
 */
void UnsteadyAdvection::InitialiseOperators()
{
    // Initialise Math
    std::string execName = Operator<double>::GetOpExecSpace(m_session);
    m_math               = Math(execName);

    m_fields[0]->SetDataWarehouse();

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            // Discontinuous projection
            m_fields[0]->GetTrace()->SetDataWarehouse();

            // Create operators
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
                m_getFwdBwdTracePhysOp->Apply(
                    m_advectionVel, m_traceAdvectionVel, m_traceAdvectionVel);
            }

            // Set advection velocity
            m_volumeFluxOp->SetAdvectVel(m_advectionVel);

            // Set trace advection velocity for upwind solver
            m_riemannSolverOp->SetTraceAdvVel(m_traceAdvectionVel);
            break;
        }
        case MultiRegions::eGalerkin:
        {
            // Continuous projection
            m_advectionCGOp =
                AdvectionOp<double>::Create(m_fields[0], m_variables);
            m_advectionCGOp->SetAdvVel(m_advectionVel);

            m_bwdTransOp = BwdTransOp<double>::Create(
                m_fields[0], m_session->GetVariables());

            // Initialise forward transform operator to evaluate initial
            // condition in coefficient space.
            m_fwdTransOp = FwdTransOp<double>::Create(
                m_fields[0], m_session->GetVariables());
            auto preconOp = PreconOp<double>::Create(m_fields[0],
                                                     m_session->GetVariables());
            // The projection solve is a mass-matrix problem, so use ConjGrad
            // regardless of the ADR system solver configured in the session.
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
}

/*
 *  Create and initialise Fields.
 */
void UnsteadyAdvection::InitialiseFields()
{
    unsigned int coordDim     = m_fields[0]->GetCoordim(0);
    unsigned int numHomoModes = 1;

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        // Discontinuous projection
        case MultiRegions::eDiscontinuous:
        {
            // Create blocks.
            auto blocks_in =
                GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
            auto blocks_trace = GetBlockAttributes<double, FieldState::Phys>(
                m_fields[0]->GetTrace());

            // Create fields.
            m_in = Field<double, FieldState::Phys>("solution", blocks_in,
                                                   m_nVariables, numHomoModes);
            m_advectionVel = Field<double, FieldState::Phys>(
                "advectionVel", blocks_in, coordDim, numHomoModes);
            m_traceAdvectionVel = Field<double, FieldState::Phys>(
                "traceAdvectVel", blocks_trace, coordDim, numHomoModes);

            // Initialise fields
            m_math.zero(m_in);
            m_math.zero(m_advectionVel);
            m_math.zero(m_traceAdvectionVel);
            break;
        }
        // Continuous field
        case MultiRegions::eGalerkin:
        {
            // Create blocks.
            auto blocks_in =
                GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
            auto blocks_coeff =
                GetBlockAttributes<double, FieldState::Coeff>(m_fields[0]);

            // Create fields.
            m_in = Field<double, FieldState::Phys>("solution", blocks_in,
                                                   m_nVariables, numHomoModes);
            m_advectionVel = Field<double, FieldState::Phys>(
                "advectionVel", blocks_in, coordDim, numHomoModes);
            m_wsp_coeff = Field<double, FieldState::Coeff>(
                "wsp coeff", blocks_coeff, m_nVariables, numHomoModes);

            // Initialise fields
            m_math.zero(m_in);
            m_math.zero(m_advectionVel);
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
void UnsteadyAdvection::SetAdvectionVel()
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
        NEKERROR(
            ErrorUtil::efatal,
            "Function 'AdvectionVelocity' was not defined in session file.")
    }
}

void UnsteadyAdvection::SetInitialConditionsField(
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
        for (unsigned int i = 0; i < m_nVariables; i++)
        {
            m_math.zero(field);

            if (m_session->GetComm()->GetRank() == 0)
            {
                std::cout << "  - Field " << m_variables[i] << ": 0 (default)"
                          << std::endl;
            }
        }
    }
}

} // namespace Nektar
