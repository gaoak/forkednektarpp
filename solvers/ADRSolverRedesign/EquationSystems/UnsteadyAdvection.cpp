/////////////////////////////////////////////////////////////////////////////
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

    // Load output/verbose parameters
    m_session->LoadParameter("IO_InfoSteps", m_infosteps, 0);

    // Initialise boundary conditions
    SetBoundaryConditions(m_time);

    // Get variable strings and number of variables
    m_variables  = m_session->GetVariables();
    m_nVariables = m_variables.size();
    m_ndim       = m_fields[0]->GetExp(0)->GetCoordim();

    m_timeOp = TimeOp<double>::Create(m_fields[0], m_variables);
    m_timeOp->DefineExplicitRhs(&UnsteadyAdvection::DoAdvection, this);
    m_timeOp->DefineProjection(&UnsteadyAdvection::DoProjection, this);

    // Create and initialise all operators
    InitialiseOperators();
    InitialiseFields();
}

/**
 * @brief Explicit solution of the unsteady advection problem.
 * Using a standalone time stepping loop.
 */
void UnsteadyAdvection::v_DoSolve()
{
    // Initialise counters
    LibUtilities::Timer timer;
    double cpuTime = 0.0;

    // Set InitialConditions
    SetInitialConditionsField(m_in);

    // Time-stepping loop
    while (m_timeOp->GetStep() < m_steps ||
           m_timeOp->GetTime() < m_fintime - NekConstants::kNekZeroTol)
    {
        timer.Start();

        // Do time integration
        m_timeOp->Apply(m_in);

        // Get CPU time
        timer.Stop();
        cpuTime += timer.TimePerTest(1);

        // Verbose print
        if (m_infosteps && !(m_timeOp->GetStep() % m_infosteps))
        {
            std::cout
                // << std::scientific
                << "Steps: " << std::setw(8) << std::left << m_timeOp->GetStep()
                << " Time: " << std::setw(12) << std::left
                << m_timeOp->GetTime() << " CPU Time: " << std::setw(8)
                << std::left << cpuTime << "s" << std::endl;

            // Reset timer
            cpuTime = 0;
        }
    }

    m_time = m_timeOp->GetTime();

    // Write result into m_field for correct output to Fld file and
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
    // Solve advection problem
    m_advectionWeakDGOp->Apply(in, out);

    // Negate the RHS and multiply by time-step
    m_math.mul(-dt, out, out);
}

/**
 * @brief Compute the projection for the unsteady advection problem.
 *
 * @param in    Given fields.
 * @param out   DG-projected fields.
 * @param time  Time.
 */
void UnsteadyAdvection::DoProjection(Field<double, FieldState::Phys> &in,
                                     Field<double, FieldState::Phys> &out,
                                     const double time)
{
    // Update time-varying boundary conditions
    SetBoundaryConditions(time);

    // DG projection
    m_math.copy(in, out);
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
    m_fields[0]->GetTrace()->SetDataWarehouse();

    // Create operators
    m_advectionWeakDGOp =
        AdvectionWeakDGOp<double>::Create(m_fields[0], m_variables);
    m_volumeFluxOp = VolumeFluxOp<double>::Create(m_fields[0], m_variables);
    m_riemannSolverOp =
        RiemannSolverOp<double>::Create(m_fields[0], m_variables);

    // Set volume flux and Riemann solver for advection operator
    m_advectionWeakDGOp->SetVolumeFluxOp(m_volumeFluxOp);
    m_advectionWeakDGOp->SetRiemannSolver(m_riemannSolverOp);

    // Check if forcing is defined
    if (m_session->DefinesFunction("AdvectionVelocity"))
    {
        // Define Velocity fields
        std::vector<std::string> vel;
        vel.push_back("Vx");
        vel.push_back("Vy");
        vel.push_back("Vz");

        // Resize the advection velocities
        vel.resize(m_ndim);

        // Create operator
        m_expressionOp =
            ExpressionOp<double>::Create(m_fields[0], vel, execName, "Generic");

        m_getFwdBwdTracePhysOp =
            GetFwdBwdTracePhysOp<double>::Create(m_fields[0], vel);
        m_getFwdBwdTracePhysOp->SetFwdOnly(true);

        // Read initial conditions and configure operator
        std::vector<LibUtilities::EquationSharedPtr> velEquations;
        for (int i = 0; i < m_ndim; ++i)
        {
            velEquations.push_back(
                m_session->GetFunction("AdvectionVelocity", vel[i]));
        }
        m_expressionOp->SetExpressions(velEquations);
    }
}

/*
 *  Create and initialise Fields.
 *
 *  m_in is created from m_fields->GetPhys() with correct boundary and initial
 *  conditions applied.
 *  m_out is a FieldState::Phys workspace initialised to zero
 */
void UnsteadyAdvection::InitialiseFields()
{
    // Create blocks.
    auto blocks_in = GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
    auto blocks_trace =
        GetBlockAttributes<double, FieldState::Phys>(m_fields[0]->GetTrace());

    // Create fields.
    unsigned int numHomoModes = 1;
    m_in = Field<double, FieldState::Phys>("solution", blocks_in, m_nVariables,
                                           numHomoModes);
    m_advectVel      = Field<double, FieldState::Phys>("advectVel", blocks_in,
                                                  m_ndim, numHomoModes);
    m_traceAdvectVel = Field<double, FieldState::Phys>(
        "traceAdvectVel", blocks_trace, m_ndim, numHomoModes);

    // Initialise fields
    m_in.Initialize<NektarSpaces::HostSpace>(0.0);
    m_advectVel.Initialize<NektarSpaces::HostSpace>(0.0);
    m_traceAdvectVel.Initialize<NektarSpaces::HostSpace>(0.0);
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
        std::string execName = Operator<double>::GetOpExecSpace(m_session);

        // Initialise operators
        auto initialOp = ExpressionOp<double>::Create(
            m_fields[0], m_session->GetVariables(), execName, "Generic");

        // Read initial conditions and configure operator
        std::vector<LibUtilities::EquationSharedPtr> initialConditons;
        for (int i = 0; i < m_nVariables; ++i)
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
            for (int i = 0; i < m_nVariables; ++i)
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
        for (int i = 0; i < m_nVariables; i++)
        {
            field.Initialize<NektarSpaces::HostSpace>(0.0);

            if (m_session->GetComm()->GetRank() == 0)
            {
                std::cout << "  - Field " << m_variables[i] << ": 0 (default)"
                          << std::endl;
            }
        }
    }

    // Evaluate and add AdvectionVelocity function, if defined
    if (m_session->DefinesFunction("AdvectionVelocity"))
    {
        // Evaluate velocity expression
        m_expressionOp->Apply(m_advectVel, m_advectVel);

        // Extract trace advection velocity for upwind solver
        m_getFwdBwdTracePhysOp->Apply(m_advectVel, m_traceAdvectVel,
                                      m_traceAdvectVel);
    }

    // Set trace advection velocity for upwind solver
    m_riemannSolverOp->SetTraceAdvVel(m_traceAdvectVel);

    // Set advection velocity
    m_volumeFluxOp->SetAdvectVel(m_advectVel);
}

} // namespace Nektar
