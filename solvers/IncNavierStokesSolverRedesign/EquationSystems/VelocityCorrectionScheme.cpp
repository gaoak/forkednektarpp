///////////////////////////////////////////////////////////////////////////////
//
// File: VelocityCorrectionScheme.cpp
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
// Description: Velocity correction scheme for solving the
//              incompressible Navier-Stokes equations.
//
///////////////////////////////////////////////////////////////////////////////

#include "Operators/TimeOps/TimeOp.hpp"
#include "SolverUtils/Forcing/Forcing.h"
#include <IncNavierStokesSolverRedesign/EquationSystems/VelocityCorrectionScheme.h>

namespace Nektar
{
using namespace Operators;

std::string VelocityCorrectionScheme::className =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "VelocityCorrectionScheme", VelocityCorrectionScheme::create);

VelocityCorrectionScheme::VelocityCorrectionScheme(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph), m_kinvis(1.0),
      m_diffCoeff(std::vector<double>(1.0)), m_lambda(1.0), m_nVariables(1),
      m_pressureIndex(0)
{
}

/**
 * @brief Initialisation object for the unsteady diffusion problem.
 */
void VelocityCorrectionScheme::v_InitObject(bool DeclareFields)
{
    EquationSystem::v_InitObject(DeclareFields);

    // Load output/verbose parameters
    m_session->LoadParameter("IO_InfoSteps", m_infosteps, 0);

    // Initialise boundary conditions
    SetBoundaryConditions(m_time);

    // Get variable strings and number of variables
    // For IncNavierStokes, we assume: u,v,w,theta,p in this order
    // first velocities, second (passive) scalar fields, last pressure
    m_variables = m_session->GetVariables();
    m_variablesVel =
        std::vector(m_variables.begin(), m_variables.begin() + m_spacedim);
    m_variablesFields = std::vector(m_variables.begin(), m_variables.end() - 1);
    m_variablesAddScalars =
        std::vector(m_variables.begin() + m_spacedim, m_variables.end() - 1);
    m_variablesPressure.push_back(m_variables.back());
    m_nVariables    = m_variables.size();
    m_pressureIndex = m_nVariables - 1;

    // Initialise Time-stepping operator
    m_timeOp = TimeOp<double>::Create(m_fields[0], m_variablesFields);
    m_timeOp->DefineImplicit(
        &VelocityCorrectionScheme::SolveUnsteadyStokesSystem, this);
    m_timeOp->DefineExplicitRhs(
        &VelocityCorrectionScheme::EvaluateAdvection_SetPressureBCs, this);
    m_timeOp->DefineProjection(&VelocityCorrectionScheme::DoProjection, this);

    // Load diffusion coefficient
    m_session->LoadParameter("kinvis", m_kinvis, 1.0);

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Create and initialise all operators
    InitialiseOperators();
    InitialiseFields();
}

/**
 * @brief Implicit solution of the unsteady diffusion problem.
 * Using a standalone time stepping loop.
 */
void VelocityCorrectionScheme::v_DoSolve()
{
    // Set InitialConditions
    SetInitialConditionsField(m_in);

    // Time-stepping loop
    while (m_timeOp->GetStep() < m_steps ||
           m_timeOp->GetTime() < m_fintime - NekConstants::kNekZeroTol)
    {

        // Do time integration
        m_timeOp->Apply(m_in);

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

    m_time = m_timeOp->GetTime();

    // TODO : Remove the below code, when updated with Redesign solverUtils
    //  ----------------------------------------------------------------------
    // Write result into m_fields.m_coeffs for correct output to Fld file and
    // check against exact solution. Velocity/scalar fields and pressure use
    // separate redesign workspaces, so copy them back separately.
    Array<OneD, double> outcoeffs = m_wsp_coeff.ToArray<double>();
    auto sizeCoeffs               = 0;
    for (unsigned int i = 0; i < m_variablesFields.size(); i++)
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

    m_fields[m_pressureIndex]->SetPhysState(false);
    Vmath::Vcopy(m_fields[m_pressureIndex]->GetNcoeffs(),
                 m_pressure_coeff.ToArray<double>(), 1,
                 m_fields[m_pressureIndex]->UpdateCoeffs(), 1);
    //--------------------------------------------------------------------------
}

void VelocityCorrectionScheme::v_GenerateSummary(SummaryList &s)
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

Array<OneD, bool> VelocityCorrectionScheme::v_GetSystemSingularChecks()
{
    int nVar = m_session->GetVariables().size();
    Array<OneD, bool> checks(nVar, false);

    const auto pressureVar = m_session->GetVariables().back();
    const SpatialDomains::BoundaryConditions bcs(m_session, m_graph);
    const auto &bndConditions = bcs.GetBoundaryConditions();

    bool pressureSingular = true;
    for (const auto &[regionId, bndCondMap] : bndConditions)
    {
        auto condIt = bndCondMap->find(pressureVar);
        ASSERTL1(condIt != bndCondMap->end(),
                 "Unable to locate pressure boundary condition for region " +
                     std::to_string(regionId));

        auto type = condIt->second->GetBoundaryConditionType();
        if (type != SpatialDomains::eNeumann &&
            type != SpatialDomains::ePeriodic)
        {
            pressureSingular = false;
            break;
        }
    }

    checks[nVar - 1] = pressureSingular;
    return checks;
}

/*
 *  @brief Evaluate explicit advection term, forcing terms
 *  and higher-order pressure boundary conditions.
 *
 *  Upon input
 *  param in: = U^{n} = [u^{n}, v^{n}, ...]
 *            = "m_in" member variable of this solver
 *  param out: "m_explicits" datastructure within IMEX operator
 *
 *  Upon output
 *  param out: = U^{n} \cdot \nabla U^{n}
 *             + f(U^{n})
 *             + \frac{\partial p}{\partial \mathbf{n}} |_{\Gamma_N}
 */
void VelocityCorrectionScheme::EvaluateAdvection_SetPressureBCs(
    Field<double, FieldState::Phys> &in, Field<double, FieldState::Phys> &out,
    const double &time, const double &dt)
{
    m_math.zero(out);

    m_math.copy(in, m_advVel);
    m_advectionOp->SetAdvVel(m_advVel);
    m_advectionOp->Apply(in, out);
    m_math.mul(-dt, out, out);

    // Evaluate and add forcing function, if defined
    if (m_session->DefinesFunction("BodyForce"))
    {
        // Set forcing operator and evaluate
        m_forcingOp->SetTime(time);
        m_forcingOp->SetScale(dt);

        m_forcingOp->Apply(in, out);
    }

    // TODO Calculate High-Order pressure boundary conditions
    // m_extrapolation->EvaluatePressureBCs(inarray, outarray, m_kinvis);
    // m_IncNavierStokesBCs->Update(inarray, outarray, params);
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
void VelocityCorrectionScheme::SolveUnsteadyStokesSystem(
    Field<double, FieldState::Phys> &in,
    [[maybe_unused]] Field<double, FieldState::Phys> &out,
    [[maybe_unused]] const double &time, const double &dt_inv_gamma)
{
    /// Set up forcing term for pressure Poisson equation
    // Compute divergence of RHS
    // TODO this should only apply to the velocity components!
    m_math.zero(m_wsp_phys_1c);
    m_divergenceOp->Apply(in, m_wsp_phys_1c);

    // Scale divergence by \gamma / \Delta t
    m_math.mul(1.0 / dt_inv_gamma, m_wsp_phys_1c, m_wsp_phys_1c);

    /// Solve Pressure System
    // Update or re-use preconditioner
    if (m_preconPressureOpMap.find(dt_inv_gamma) == m_preconPressureOpMap.end())
    {
        // Configure and cache preconditioner
        m_preconPressureOpMap.insert(
            {dt_inv_gamma, PreconOp<double>::Create(m_fields[m_pressureIndex],
                                                    m_variablesPressure)});
        m_poissonSolveOp->SetPrecon(m_preconPressureOpMap[dt_inv_gamma]);
        m_poissonSolveOp->UpdatePrecon();
    }
    else
    {
        // Re-use preconditioner
        m_poissonSolveOp->SetPrecon(m_preconPressureOpMap[dt_inv_gamma]);
    }

    m_poissonSolveOp->Apply(m_wsp_phys_1c, m_pressure_coeff);

    /// Set up forcing term for Helmholtz problems
    // Evaluate p^{n+1} in physical space
    // m_math.zero(m_wsp_phys_1c);
    m_bwdTransPressureOp->Apply(m_pressure_coeff, m_wsp_phys_1c);

    // Compute derivative \nabla p^{n+1}
    m_physDerivPressureOp->Apply(m_wsp_phys_1c, m_wsp_phys_deriv_pressure);

    // Subtract inarray/(aii_dt) and divide by kinvis
    m_math.daxpy(-1 / dt_inv_gamma, in, m_wsp_phys_deriv_pressure, m_wsp_phys);
    m_math.mul(1 / m_kinvis, m_wsp_phys, m_wsp_phys);

    /// Solve velocity systems
    // Update \lambda = \gamma / \Delta t / \nu
    // TODO check lambda sign
    m_lambda = 1.0 / dt_inv_gamma / m_kinvis;
    m_helmSolveOp->SetLambda(m_lambda);

    // Update or re-use preconditioner
    if (m_preconFieldsOpMap.find(dt_inv_gamma) == m_preconFieldsOpMap.end())
    {
        // Configure and cache preconditioner
        m_preconFieldsOpMap.insert(
            {dt_inv_gamma,
             PreconOp<double>::Create(m_fields[0], m_variablesFields)});
        m_helmSolveOp->SetPrecon(m_preconFieldsOpMap[dt_inv_gamma]);
        m_helmSolveOp->UpdatePrecon();
    }
    else
    {
        // Re-use preconditioner
        m_helmSolveOp->SetPrecon(m_preconFieldsOpMap[dt_inv_gamma]);
    }

    // Solve diffusion problem for each component
    m_math.zero(m_wsp_coeff);

    m_helmSolveOp->Apply(m_wsp_phys, m_wsp_coeff);

    // Transform to physical space
    m_bwdTransFieldsOp->Apply(m_wsp_coeff, out);
}

/**
 * @brief Compute the projection for the unsteady diffusion problem.
 *
 * @param in    Given fields.
 * @param out   CG-projected fields.
 * @param time  Time.
 */
void VelocityCorrectionScheme::DoProjection(
    Field<double, FieldState::Phys> &in, Field<double, FieldState::Phys> &out,
    [[maybe_unused]] const double time)
{
    // Update time-varying boundary conditions
    // SetBoundaryConditions(time);

    // CG projection
    // Note we could use the cheaper operators: AvgAssemble or GlobalToLocal
    m_fwdTransFieldsOp->Apply(in, m_wsp_coeff);
    m_bwdTransFieldsOp->Apply(m_wsp_coeff, out);
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
void VelocityCorrectionScheme::SetDiffusionCoeff()
{
    // Set-up anisotropic diffusion coefficient
    // Default value for D_ij = 1, if i = j
    // Default value for D_ij = 0, if i != j
    const auto coordDim      = m_fields[0]->GetCoordim(0);
    const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
    m_diffCoeff.resize(diffCoeffSize);

    if (coordDim == 1)
    {
        m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
    }
    else if (coordDim == 2)
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

/*
 *  Create and initialise all operators for this solver
 */
void VelocityCorrectionScheme::InitialiseOperators()
{
    // Initialise Math
    std::string execName = Operator<double>::GetOpExecSpace(m_session);
    m_math               = Math(execName);

    // Create velocity operators
    m_linearSolverFieldsOp = LinearSolverOp<double>::Create(
        m_fields[0], m_variablesFields, "ConjGrad");
    m_advectionOp = AdvectionOp<double>::Create(m_fields[0], m_variablesFields);
    m_bwdTransFieldsOp =
        BwdTransOp<double>::Create(m_fields[0], m_variablesFields);
    m_divergenceOp =
        DivergenceOp<double>::Create(m_fields[0], m_variablesFields);
    m_helmSolveOp = HelmSolveOp<double>::Create(m_fields[0], m_variablesFields);

    // Configure HelmSolve
    m_helmSolveOp->SetLinearSolver(m_linearSolverFieldsOp);
    m_helmSolveOp->SetDiffCoeff(m_diffCoeff);

    // Create pressure operators
    m_bwdTransPressureOp = BwdTransOp<double>::Create(m_fields[m_pressureIndex],
                                                      m_variablesPressure);
    m_physDerivPressureOp = PhysDerivOp<double>::Create(
        m_fields[m_pressureIndex], m_variablesPressure);
    m_poissonSolveOp = PoissonSolveOp<double>::Create(m_fields[m_pressureIndex],
                                                      m_variablesPressure);
    m_linearSolverPressureOp = LinearSolverOp<double>::Create(
        m_fields[m_pressureIndex], m_variablesPressure, "ConjGrad");

    // Configure PoissonSolve
    m_poissonSolveOp->SetLinearSolver(m_linearSolverPressureOp);
    m_poissonSolveOp->SetDiffCoeff(m_diffCoeff);

    // Initialise forcing operator, if defined in session file
    // Note we assume forcing acts only on velocity components
    if (m_session->DefinesFunction("BodyForce"))
    {
        // Create operator
        // TODO check which variables/components here?
        m_forcingOp = ExpressionOp<double>::Create(m_fields[0], m_variablesVel,
                                                   "Serial", "Generic");

        // Read forcing functions for all variables and configure operator
        std::vector<LibUtilities::EquationSharedPtr> forcingEquations;
        for (int i = 0; i < m_variablesVel.size(); ++i)
        {
            forcingEquations.push_back(m_session->GetFunction("BodyForce", i));
        }
        m_forcingOp->SetExpressions(forcingEquations);
        m_forcingOp->SetTime(m_time);
        // Set to append for VelocityCorrectionScheme
        m_forcingOp->SetAppend(true);
    }

    // Initialise forward transform operator to evaluate initial condition
    // in coefficient space.
    m_fwdTransFieldsOp =
        FwdTransOp<double>::Create(m_fields[0], m_variablesFields);
    auto preconOp = PreconOp<double>::Create(m_fields[0], m_variablesFields);
    auto linsolverOp =
        LinearSolverOp<double>::Create(m_fields[0], m_variablesFields);
    m_fwdTransFieldsOp->SetLinearSolver(linsolverOp);
    m_fwdTransFieldsOp->SetPrecon(preconOp);
    m_fwdTransFieldsOp->UpdatePrecon();
}

/*
 *  Create and initialise Fields.
 */
void VelocityCorrectionScheme::InitialiseFields()
{
    // Create blocks for velocity and passive scalars.
    auto bAtr_phys = GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
    auto bAtr_coeff =
        GetBlockAttributes<double, FieldState::Coeff>(m_fields[0]);

    // Create blocks for pressure.
    auto bAtr_phys_pressure =
        GetBlockAttributes<double, FieldState::Phys>(m_fields[m_pressureIndex]);
    auto bAtr_coeff_pressure = GetBlockAttributes<double, FieldState::Coeff>(
        m_fields[m_pressureIndex]);

    // Note this is parsed in EquationSystem.cpp
    unsigned int nhomo = m_npointsZ;

    // Create fields for velocity and passive scalars.
    m_in = Field<double, FieldState::Phys>("velocity and passive scalars",
                                           bAtr_phys, m_variablesFields.size(),
                                           nhomo);
    m_wsp_coeff = Field<double, FieldState::Coeff>(
        "wsp_coeff", bAtr_coeff, m_variablesFields.size(), nhomo);
    m_wsp_phys = Field<double, FieldState::Phys>(
        "wsp_phys", bAtr_phys, m_variablesFields.size(), nhomo);
    m_wsp_phys_deriv_pressure = Field<double, FieldState::Phys>(
        "wsp_phys_deriv_pressure", bAtr_phys_pressure, m_spacedim, nhomo);

    // Create fields for pressure.
    m_pressure = Field<double, FieldState::Phys>(
        "pressure", bAtr_phys_pressure, m_variablesPressure.size(), nhomo);
    m_pressure_coeff =
        Field<double, FieldState::Coeff>("pressure coeff", bAtr_coeff_pressure,
                                         m_variablesPressure.size(), nhomo);
    m_wsp_phys_1c = Field<double, FieldState::Phys>(
        "wsp_phys_1c", bAtr_phys_pressure, m_variablesPressure.size(), nhomo);

    m_advVel = Field<double, FieldState::Phys>(
        "explicit advection velocity", bAtr_phys, m_variablesVel.size(), nhomo);
}

void VelocityCorrectionScheme::SetInitialConditionsField(
    Field<double, FieldState::Phys> &fields)
{
    // Print to log/console
    if (m_session->GetComm()->GetRank() == 0)
    {
        std::cout << "Initial Conditions:" << std::endl;
    }

    // Set initial conditions from session file
    if (m_session->DefinesFunction("InitialConditions"))
    {
        // Initialise expression operator
        auto initialVelocityOp =
            ExpressionOp<double>::Create(m_fields[0], m_variablesFields);
        initialVelocityOp->SetTime(m_time);

        // Read initial conditions and configure operator for velocity and
        // passive scalars
        std::vector<LibUtilities::EquationSharedPtr> initialConditons;
        for (int i = 0; i < m_variablesFields.size(); ++i)
        {
            initialConditons.push_back(
                m_session->GetFunction("InitialConditions", i));
        }
        initialVelocityOp->SetExpressions(initialConditons);

        // Set initial conditions defined in session and update coefficients
        m_math.zero(fields);
        m_math.zero(m_wsp_coeff);
        initialVelocityOp->Apply(fields, fields);

        // Global C0-projection of initial conditions
        // Note we could use the cheaper operators: AvgAssemble or GlobalToLocal
        m_fwdTransFieldsOp->Apply(fields, m_wsp_coeff);
        m_bwdTransFieldsOp->Apply(m_wsp_coeff, fields);

        /// Set pressure initial conditions
        // Read initial conditions and re-configure operator for pressure
        initialConditons.clear();
        for (int i = m_variablesFields.size(); i < m_nVariables; ++i)
        {
            initialConditons.push_back(
                m_session->GetFunction("InitialConditions", i));
        }
        auto initialPressureOp = ExpressionOp<double>::Create(
            m_fields[m_pressureIndex], m_variablesPressure);
        initialPressureOp->SetTime(m_time);
        initialPressureOp->SetExpressions(initialConditons);

        // Set initial conditions defined in session and update coefficients
        m_math.zero(m_pressure);
        m_math.zero(m_pressure_coeff);
        initialPressureOp->Apply(m_pressure, m_pressure);

        // Project the pressure initial condition into coefficient space so the
        // first Stokes step has a usable pressure state.
        auto fwdTransPressureOp = FwdTransOp<double>::Create(
            m_fields[m_pressureIndex], m_variablesPressure);
        auto preconPressureOp = PreconOp<double>::Create(
            m_fields[m_pressureIndex], m_variablesPressure);
        auto linsolverPressureOp = LinearSolverOp<double>::Create(
            m_fields[m_pressureIndex], m_variablesPressure);
        fwdTransPressureOp->SetLinearSolver(linsolverPressureOp);
        fwdTransPressureOp->SetPrecon(preconPressureOp);
        fwdTransPressureOp->UpdatePrecon();
        fwdTransPressureOp->Apply(m_pressure, m_pressure_coeff);
        m_bwdTransPressureOp->Apply(m_pressure_coeff, m_pressure);

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
        // Zero all components in fields and pressure
        m_math.zero(fields);
        m_math.zero(m_pressure);
        m_math.zero(m_pressure_coeff);

        for (int i = 0; i < m_nVariables; i++)
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
