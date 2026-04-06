/////////////////////////////////////////////////////////////////////////////
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

#include "Operators/TimeOps/TimeOp.hpp"

#include <CompressibleFlowSolverRedesign/EquationSystems/EulerCFE.h>
#include <LibUtilities/BasicUtils/Timer.h>

namespace Nektar
{
using namespace Operators;

std::string EulerCFE::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "EulerCFE", EulerCFE::create,
        "Euler equations in conservative variables.");

EulerCFE::EulerCFE(const LibUtilities::SessionReaderSharedPtr &pSession,
                   const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph), m_nVariables(1)
{
}

/**
 * @brief Initialisation object for the Euler equations in conservative
 * variables.
 */
void EulerCFE::v_InitObject(bool DeclareFields)
{
    // Call to the initialisation object of EquationSystem
    EquationSystem::v_InitObject(DeclareFields);

    // Load output/verbose parameters
    m_session->LoadParameter("IO_InfoSteps", m_infosteps, 0);

    ASSERTL0(m_session->DefinesSolverInfo("UPWINDTYPE"),
             "No UPWINDTYPE defined in session.");

    // Loading parameters from session file
    InitialiseParameters();

    // Initialise general boundary conditions
    SetBoundaryConditions(m_time);

    // Get variable strings and number of variables
    m_variables  = m_session->GetVariables();
    m_nVariables = m_variables.size();
    m_ndim       = m_fields[0]->GetExp(0)->GetCoordim();

    m_timeOp = TimeOp<double>::Create(m_fields[0], m_variables);
    m_timeOp->DefineExplicitRhs(&EulerCFE::DoAdvection, this);
    m_timeOp->DefineProjection(&EulerCFE::DoProjection, this);

    // Create and initialise all operators
    InitialiseOperators();
    InitialiseFields();
}

/**
 * @brief Explicit solution of Euler equations.
 * Using a standalone time stepping loop.
 */
void EulerCFE::v_DoSolve()
{
    // Initialise counters
    LibUtilities::Timer timer;
    double cpuTime = 0.0;

    // Set InitialConditions
    SetInitialConditionsField(m_in); // Set initial conditions in m_in

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

void EulerCFE::v_GenerateSummary(SummaryList &s)
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
void EulerCFE::DoAdvection(Field<double, FieldState::Phys> &in,
                           Field<double, FieldState::Phys> &out,
                           [[maybe_unused]] const double &time,
                           const double &dt)
{
    // Solve advection problem
    m_advectionWeakDGOp->Apply(in, out);

    // Negate the RHS and multiply by time-step
    m_math.mul(-dt, out, out);
}

/**
 * @brief Compute the projection for Euler equations.
 *
 * @param in    Given fields.
 * @param out   DG-projected fields.
 * @param time  Time.
 */
void EulerCFE::DoProjection(Field<double, FieldState::Phys> &in,
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
void EulerCFE::InitialiseOperators()
{
    // Initialise Math
    std::string execName = Operator<double>::GetOpExecSpace(m_session);
    m_math               = Math(execName);

    m_fields[0]->SetDataWarehouse();
    m_fields[0]->GetTrace()->SetDataWarehouse();

    // Create operators
    m_advectionWeakDGOp =
        AdvectionWeakDGOp<double>::Create(m_fields[0], m_variables);
    std::string riemannMethod = m_session->GetSolverInfo("UpwindType");
    m_riemannSolverOp         = CompressibleSolverOp<double>::Create(
        m_fields[0], m_variables, riemannMethod, execName);
    m_volumeFluxOp = VolumeFluxOp<double>::Create(m_fields[0], m_variables);

    // Set volume flux and Riemann solver for advection operator
    m_advectionWeakDGOp->SetVolumeFluxOp(m_volumeFluxOp);
    m_advectionWeakDGOp->SetRiemannSolver(m_riemannSolverOp);
}

/*
 *  Create and initialise Fields.
 *
 *  m_in is created from m_fields->GetPhys() with correct boundary and initial
 *  conditions applied.
 *  m_out is a FieldState::Phys workspace initialised to zero
 */
void EulerCFE::InitialiseFields()
{
    // Create blocks.
    auto blocks_in = GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
    auto blocks_trace =
        GetBlockAttributes<double, FieldState::Phys>(m_fields[0]->GetTrace());

    // Create fields.
    unsigned int numHomoModes = 1;
    m_in = Field<double, FieldState::Phys>("solution", blocks_in, m_nVariables,
                                           numHomoModes);
    // Initialise fields
    m_math.zero(m_in);
}

void EulerCFE::SetInitialConditionsField(Field<double, FieldState::Phys> &field)
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
            m_math.zero(field);

            if (m_session->GetComm()->GetRank() == 0)
            {
                std::cout << "  - Field " << m_variables[i] << ": 0 (default)"
                          << std::endl;
            }
        }
    }
}

/**
 * @brief Load CFS parameters from the session file.
 */
void EulerCFE::InitialiseParameters()
{
    // Get gamma parameter from session file.
    m_session->LoadParameter("Gamma", m_gamma, 1.4);

    // // Shock capture
    // m_session->LoadSolverInfo("ShockCaptureType", m_shockCaptureType, "Off");
    //
    // // Check if the shock capture type is supported
    // std::string err_msg = "Warning, ShockCaptureType = " + m_shockCaptureType
    // +
    //                       " is not supported by this solver";
    // Load parameters for exponential filtering
    // m_session->MatchSolverInfo("ExponentialFiltering", "True",
    // m_useFiltering,
    //                            false);
    // if (m_useFiltering)
    // {
    //     m_session->LoadParameter("FilterAlpha", m_filterAlpha, 36);
    //     m_session->LoadParameter("FilterExponent", m_filterExponent, 16);
    //     m_session->LoadParameter("FilterCutoff", m_filterCutoff, 0);
    // }
    // // Load CFL for local time-stepping (for steady state)
    // m_session->MatchSolverInfo("LocalTimeStep", "True", m_useLocalTimeStep,
    //                            false);
}

} // namespace Nektar
