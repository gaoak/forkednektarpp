///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyDiffusion.cpp
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
// Description: Unsteady diffusion solve routines
//
///////////////////////////////////////////////////////////////////////////////

#include "Operators/TimeOps/TimeOp.hpp"

#include <LibUtilities/BasicUtils/Timer.h>
#include <RedesignSolver/EquationSystems/UnsteadyDiffusion.h>

namespace Nektar
{
using namespace Operators;

std::string UnsteadyDiffusion::className =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "UnsteadyDiffusion", UnsteadyDiffusion::create);

UnsteadyDiffusion::UnsteadyDiffusion(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph), m_epsilon(1.0), m_lambda(1.0),
      m_diffCoeff(std::vector<double>(1.0)), m_nVariables(1)
{
}

/**
 * @brief Initialisation object for the unsteady diffusion problem.
 */
void UnsteadyDiffusion::v_InitObject(bool DeclareFields)
{
    EquationSystem::v_InitObject(DeclareFields);

    // Load output/verbose parameters
    m_session->LoadParameter("IO_InfoSteps", m_infosteps, 0);

    // Initialise boundary conditions
    SetBoundaryConditions(m_time);

    // Get variable strings and number of variables
    m_variables  = m_session->GetVariables();
    m_nVariables = m_variables.size();

    // Initialise Time-stepping operator
    m_timeOp = TimeOp<double>::Create(m_fields[0], m_session->GetVariables());
    m_timeOp->DefineImplicit(&UnsteadyDiffusion::DoDiffusion, this);
    m_timeOp->DefineExplicitRhs(&UnsteadyDiffusion::DoReaction, this);
    m_timeOp->DefineProjection(&UnsteadyDiffusion::DoProjection, this);

    // Load diffusion coefficient
    m_session->LoadParameter("epsilon", m_epsilon, 1.0);

    // Initialise default diffusion coefficients
    const auto coordDim      = m_fields[0]->GetCoordim(0);
    const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
    m_diffCoeff.resize(diffCoeffSize);

    // Set up (isotropic) diffusion coefficient.
    if (coordDim == 1)
    {
        m_diffCoeff[0] = 1.0; // D00
    }
    else if (coordDim == 2)
    {
        m_diffCoeff[0] = 1.0; // D00
        m_diffCoeff[2] = 1.0; // D11
    }
    else
    {
        m_diffCoeff[0] = 1.0; // D00
        m_diffCoeff[2] = 1.0; // D11
        m_diffCoeff[5] = 1.0; // D22
    }

    // Create and initialise all operators
    InitialiseOperators();
    InitialiseFields();
}

/**
 * @brief Implicit solution of the unsteady diffusion problem.
 * Using a standalone time stepping loop.
 */
void UnsteadyDiffusion::v_DoSolve()
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

    // Write result into m_fields.m_coeffs for correct output to Fld file and
    // check against exact solution
    // Note outcoeffs has data arranged as [comp0, comp1, comp2] in a single
    // array
    Array<OneD, double> outcoeffs = m_wsp_coeff.ToArray<double>();
    auto sizeCoeffs               = 0;
    for (unsigned int i = 0; i < m_nVariables; i++)
    {
        // Set PhysState to false in order to use coeffs for comparison against
        // exact solution
        m_fields[i]->SetPhysState(false);

        // Get physical size for this field
        auto nCoeff = m_fields[i]->GetNcoeffs();

        // Copy result into m_fields.m_phys
        Array<OneD, double> outcoeffs_var = outcoeffs + sizeCoeffs;
        Vmath::Vcopy(nCoeff, outcoeffs_var, 1, m_fields[i]->UpdateCoeffs(), 1);

        // Increment
        sizeCoeffs += nCoeff;
    }
}

void UnsteadyDiffusion::v_GenerateSummary(SummaryList &s)
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
 *  @brief Setup rhs forcing term and solve Helmholtz system.
 *
 *  Upon input
 *  param in: = \sum_q=0^{J-1} \alpha_q / \gamma u^{n-q}
 *      + \Delta t \sum_q=0^{J-1} \beta_q / \gamma g(u^{n-q})
 *  the first term is used for implicit schemes, e.g. BDFImplicit
 *  the second term is used for explicit schemes, e.g. Adams-Bashforth
 *  both terms are used for implicit-explicit (IMEX) schemes
 *  param out: = param in
 *  param time: = t^{n+1}
 *  param dt_inv_gamma: = \Delta t / \gamma
 *
 *  Upon output
 *  param in: = u^{n+1}
 *  param out: = param in
 */
void UnsteadyDiffusion::DoDiffusion(
    Field<double, FieldState::Phys> &in,
    [[maybe_unused]] Field<double, FieldState::Phys> &out,
    [[maybe_unused]] const double &time, const double &dt_inv_gamma)
{
    // Update \lambda = \gamma / \Delta t / \epsilon
    m_lambda = 1.0 / dt_inv_gamma / m_epsilon;

    // Update HelmSolve
    m_helmSolveOp->SetLambda(m_lambda);
    if (m_preconOp.find(dt_inv_gamma) == m_preconOp.end())
    {
        // Configure and cache preconditioner
        m_preconOp.insert(
            {dt_inv_gamma,
             PreconOp<double>::Create(m_fields[0], m_session->GetVariables())});
        m_helmSolveOp->SetPrecon(m_preconOp[dt_inv_gamma]);
        m_helmSolveOp->UpdatePrecon();
    }
    else
    {
        // Re-use preconditioner
        m_helmSolveOp->SetPrecon(m_preconOp[dt_inv_gamma]);
    }

    // Multiply by negative lambda
    m_math.mul(-m_lambda, in, out);

    // Solve diffusion problem
    m_helmSolveOp->Apply(out, m_wsp_coeff);

    // Transform to physical space
    m_bwdTransOp->Apply(m_wsp_coeff, out);
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
void UnsteadyDiffusion::DoReaction(Field<double, FieldState::Phys> &in,
                                   Field<double, FieldState::Phys> &out,
                                   const double &time, const double &dt)
{
    // Evaluate and add forcing function, if defined
    if (m_session->DefinesFunction("BodyForce"))
    {
        // Set forcing operator and evaluate
        m_forcingOp->SetTime(time);
        m_forcingOp->SetScale(dt);
        m_forcingOp->Apply(in, out);
    }
}

/**
 * @brief Compute the projection for the unsteady diffusion problem.
 *
 * @param in    Given fields.
 * @param out   CG-projected fields.
 * @param time  Time.
 */
void UnsteadyDiffusion::DoProjection(Field<double, FieldState::Phys> &in,
                                     Field<double, FieldState::Phys> &out,
                                     [[maybe_unused]] const double time)
{
    // Update time-varying boundary conditions
    // SetBoundaryConditions(time);

    // CG projection
    // Note we could use the cheaper operators: AvgAssemble or GlobalToLocal
    m_fwdTransOp->Apply(in, m_wsp_coeff);
    m_bwdTransOp->Apply(m_wsp_coeff, out);
}

/*
 *  Create and initialise all operators for this solver
 */
void UnsteadyDiffusion::InitialiseOperators()
{
    // Initialise Math
    std::string execName = Operator<double>::GetOpExecSpace(m_session);
    m_math               = Math(execName);

    // Create Helmsolve and BwdTrans operators
    m_bwdTransOp =
        BwdTransOp<double>::Create(m_fields[0], m_session->GetVariables());
    m_helmSolveOp =
        HelmSolveOp<double>::Create(m_fields[0], m_session->GetVariables());

    // Configure HelmSolve with default parameters
    m_linearSolverOp =
        LinearSolverOp<double>::Create(m_fields[0], m_session->GetVariables());
    m_helmSolveOp->SetLinearSolver(m_linearSolverOp);
    m_helmSolveOp->SetLinearSolver(m_linearSolverOp);
    m_helmSolveOp->SetDiffCoeff(m_diffCoeff);

    // Check if forcing is defined
    if (m_session->DefinesFunction("BodyForce"))
    {
        // Create operator
        m_forcingOp = ExpressionOp<double>::Create(
            m_fields[0], m_session->GetVariables(), "Serial", "Generic");

        // Read initial conditions and configure operator
        std::vector<LibUtilities::EquationSharedPtr> forcingEquations;
        for (int i = 0; i < m_nVariables; ++i)
        {
            forcingEquations.push_back(m_session->GetFunction("BodyForce", i));
        }
        m_forcingOp->SetExpressions(forcingEquations);
        m_forcingOp->SetTime(m_time);
    }

    // Initialise forward transform operator to evaluate initial condition
    // in coefficient space.
    m_fwdTransOp =
        FwdTransOp<double>::Create(m_fields[0], m_session->GetVariables());
    auto preconOp =
        PreconOp<double>::Create(m_fields[0], m_session->GetVariables());
    auto linsolverOp =
        LinearSolverOp<double>::Create(m_fields[0], m_session->GetVariables());
    m_fwdTransOp->SetLinearSolver(linsolverOp);
    m_fwdTransOp->SetPrecon(preconOp);
    m_fwdTransOp->UpdatePrecon();
}

/*
 *  Create and initialise Fields.
 */
void UnsteadyDiffusion::InitialiseFields()
{
    // Create blocks.
    auto bAtr_phys = GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
    auto bAtr_coeff =
        GetBlockAttributes<double, FieldState::Coeff>(m_fields[0]);

    // Create fields.
    unsigned int nhomo = m_npointsZ; // Note read in EquationSystem.cpp
    m_in = Field<double, FieldState::Phys>("solution", bAtr_phys, m_nVariables,
                                           nhomo);
    m_wsp_coeff = Field<double, FieldState::Coeff>("wsp_coeff", bAtr_coeff,
                                                   m_nVariables, nhomo);
}

void UnsteadyDiffusion::SetInitialConditionsField(
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
            m_fields[0], m_session->GetVariables(), "Serial", "Generic");

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
        m_math.zero(field, "Serial"); // Use Serial for now as in initialOp.
        m_math.zero(m_wsp_coeff,
                    "Serial"); // Use Serial for now as in initialOp
        initialOp->Apply(field, field);

        // Note we could use the cheaper operators: AvgAssemble or GlobalToLocal
        m_fwdTransOp->Apply(field, m_wsp_coeff);
        m_bwdTransOp->Apply(m_wsp_coeff, field);

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
