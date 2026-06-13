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

#include <IncNavierStokesSolverRedesign/EquationSystems/VelocityCorrectionScheme.h>

namespace Nektar
{
using namespace Operators;

std::string VelocityCorrectionScheme::className =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "VelocityCorrectionScheme", VelocityCorrectionScheme::create,
        "Velocity correciton scheme for solving incompressible Navier-Stokes "
        "in segregated form.");

VelocityCorrectionScheme::VelocityCorrectionScheme(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), m_kinvis(1.0),
      m_diffCoeff(std::vector<double>(1.0)), m_lambda(1.0), m_pressureIndex(0)
{
    ASSERTL0(m_projectionType == MultiRegions::eGalerkin,
             "The VelocityCorrectionScheme is only implemented for "
             "projectionType Galerkin");

    // Get variable strings and number of variables
    // For IncNavierStokes, we assume: u,v,w,theta,p in this order
    // 1. velocities: u, v, w (in 3D)
    // 2. passive scalar fields: theta (not required)
    // 3. pressure: p
    m_variablesVel =
        std::vector(m_variables.begin(), m_variables.begin() + m_coordim);
    m_variablesFields = std::vector(m_variables.begin(), m_variables.end() - 1);
    m_variablesAddScalars =
        std::vector(m_variables.begin() + m_coordim, m_variables.end() - 1);
    m_variablesPressure = {m_variables.back()};
    m_pressureIndex     = m_nVariables - 1;

    // Save list of all variables for output with pressure
    m_variablesTotal = m_variables;

    // Overwrite EquationSystem::m_variables
    // Note this is required to correctly use virtual routines from parent
    // classes: EquationSystem and UnsteadySystem
    m_variables  = m_variablesFields;
    m_nVariables = m_variables.size();
}

/**
 * @brief Initialisation object for the unsteady diffusion problem.
 */
void VelocityCorrectionScheme::v_InitObject(bool declareExpansionLists)
{
    // Call to the initialisation object of EquationSystem
    UnsteadySystem::v_InitObject(declareExpansionLists);

    // Declare ExpansionList for pressure
    auto velocityExpansionList =
        std::dynamic_pointer_cast<MultiRegions::ContField>(m_expansionLists[0]);
    if (m_graph->SameExpansionInfo(m_variables[0], m_variablesPressure[0]))
    {
        m_expansionLists.push_back(
            MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                *velocityExpansionList, m_graph, m_variablesPressure[0], true,
                m_checkIfSystemSingular[m_pressureIndex]));
    }
    else
    {
        m_expansionLists.push_back(
            MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                m_session, m_graph, m_variablesPressure[0], true,
                m_checkIfSystemSingular[m_pressureIndex]));
    }
    m_expansionLists[m_pressureIndex]->SetDataWarehouse();

    /// Create Field for solution m_fields and others
    // Note we use expansionList for velocity (and added scalar) fields
    InitialiseFields();

    // Load physical parameters
    InitialiseParameters();

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Create and initialise all operators
    InitialiseOperators();

    // Configure Time-integration
    // Note we exclude pressure from time integration
    InitialiseTimeOp();
    m_timeOp->DefineImplicit(
        &VelocityCorrectionScheme::SolveUnsteadyStokesSystem, this);
    m_timeOp->DefineExplicitRhs(
        &VelocityCorrectionScheme::EvaluateAdvection_SetPressureBCs, this);
}

void VelocityCorrectionScheme::v_GenerateSummary(SummaryList &s)
{
    UnsteadySystem::v_GenerateSummary(s);

    AddSummaryItem(s, "Equations", "Incompressible Navier-Stokes");
    AddSummaryItem(s, "Algorithm", "Velocity correction scheme");
    AddSummaryItem(s, "Formulation", "Semi-implicit");

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
void VelocityCorrectionScheme::SolveUnsteadyStokesSystem(
    Field<double, FieldState::Phys> &in,
    [[maybe_unused]] Field<double, FieldState::Phys> &out,
    [[maybe_unused]] const double &time, const double &dt_inv_gamma)
{
    /// Set up forcing term for pressure Poisson equation
    // Compute divergence of RHS
    // TODO this should only apply to the velocity components!
    m_divergenceOp->Apply(in, m_wsp_phys_1c);

    // Scale divergence by \gamma / \Delta t
    m_math.mul(1.0 / dt_inv_gamma, m_wsp_phys_1c, m_wsp_phys_1c);

    /// Solve Pressure System
    // Update or re-use preconditioner
    if (m_preconPressureOpMap.find(dt_inv_gamma) == m_preconPressureOpMap.end())
    {
        // Configure and cache preconditioner
        m_preconPressureOpMap.insert(
            {dt_inv_gamma,
             PreconOp<double>::Create(m_expansionLists[m_pressureIndex],
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
    m_bwdTransPressureOp->Apply(m_pressure_coeff, m_wsp_phys_1c);
    m_math.copy(m_wsp_phys_1c, m_pressure);

    // Compute derivative \nabla p^{n+1}
    m_physDerivPressureOp->Apply(m_wsp_phys_1c, m_wsp_phys_deriv_pressure);

    // Subtract inarray/(aii_dt) and divide by kinvis
    m_math.daxpy(-1 / dt_inv_gamma, in, m_wsp_phys_deriv_pressure, m_wsp_phys);
    m_math.mul(1 / m_kinvis, m_wsp_phys, m_wsp_phys);

    /// Solve velocity systems
    // Update \lambda = \gamma / \Delta t / \nu
    m_lambda = 1.0 / dt_inv_gamma / m_kinvis;
    m_helmSolveOp->SetLambda(m_lambda);

    // Update or re-use preconditioner
    if (m_preconFieldsOpMap.find(dt_inv_gamma) == m_preconFieldsOpMap.end())
    {
        // Configure and cache preconditioner
        m_preconFieldsOpMap.insert(
            {dt_inv_gamma,
             PreconOp<double>::Create(m_expansionLists[0], m_variablesFields)});
        m_helmSolveOp->SetPrecon(m_preconFieldsOpMap[dt_inv_gamma]);
        m_helmSolveOp->UpdatePrecon();
    }
    else
    {
        // Re-use preconditioner
        m_helmSolveOp->SetPrecon(m_preconFieldsOpMap[dt_inv_gamma]);
    }

    // Solve diffusion problem for each component
    m_math.zero(m_fields_coeff);

    m_helmSolveOp->Apply(m_wsp_phys, m_fields_coeff);

    // Transform to physical space
    m_bwdTransOp->Apply(m_fields_coeff, out);
}

/*
 *  @brief Evaluate explicit advection term, forcing terms
 *  and higher-order pressure boundary conditions.
 *
 *  Upon input
 *  param in: = U^{n} = [u^{n}, v^{n}, ...]
 *            = "m_fields" member variable of this solver
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

    m_advectionOp->SetScale(-dt);
    m_math.copy(in, m_advVel);
    m_advectionOp->SetAdvVel(m_advVel);
    m_advectionOp->Apply(in, out);

    // Evaluate and add forcing terms. Advection has already written to out, so
    // all configured forcings append their contribution to the explicit RHS.
    for (auto &forcing : m_forcing)
    {
        forcing->SetAppend(true);
        forcing->Apply(in, out, time, dt);
    }

    // TODO Calculate High-Order pressure boundary conditions
    // m_extrapolation->EvaluatePressureBCs(inarray, outarray, m_kinvis);
    // m_IncNavierStokesBCs->Update(inarray, outarray, params);
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
    const auto diffCoeffSize = m_coordim * (m_coordim + 1) / 2;
    m_diffCoeff.resize(diffCoeffSize);

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

/*
 *  Create and initialise all operators for this solver
 */
void VelocityCorrectionScheme::v_InitialiseOperators()
{
    EquationSystem::v_InitialiseOperators();

    // Create velocity operators
    m_linearSolverFieldsOp = LinearSolverOp<double>::Create(
        m_expansionLists[0], m_variablesFields, "ConjGrad");
    m_advectionOp =
        AdvectionOp<double>::Create(m_expansionLists[0], m_variablesFields);
    m_divergenceOp =
        DivergenceOp<double>::Create(m_expansionLists[0], m_variablesFields);
    m_helmSolveOp =
        HelmSolveOp<double>::Create(m_expansionLists[0], m_variablesFields);

    // Configure HelmSolve
    m_helmSolveOp->SetLinearSolver(m_linearSolverFieldsOp);
    m_helmSolveOp->SetDiffCoeff(m_diffCoeff);

    // Create pressure operators
    m_bwdTransPressureOp = BwdTransOp<double>::Create(
        m_expansionLists[m_pressureIndex], m_variablesPressure);
    m_physDerivPressureOp = PhysDerivOp<double>::Create(
        m_expansionLists[m_pressureIndex], m_variablesPressure);
    m_poissonSolveOp = PoissonSolveOp<double>::Create(
        m_expansionLists[m_pressureIndex], m_variablesPressure);
    m_linearSolverPressureOp = LinearSolverOp<double>::Create(
        m_expansionLists[m_pressureIndex], m_variablesPressure, "ConjGrad");

    // Configure PoissonSolve
    m_poissonSolveOp->SetLinearSolver(m_linearSolverPressureOp);
    m_poissonSolveOp->SetDiffCoeff(m_diffCoeff);

    // Load forcing terms, if defined in the session file. For IncNS the
    // pressure variable is excluded, so forcing acts only on velocity fields.
    m_forcing = Forcing::Load(m_session, m_expansionLists[0], m_variablesVel);
}

/*
 *  Create and initialise Fields.
 */
void VelocityCorrectionScheme::v_InitialiseFields()
{
    // EquationSystem initialises m_fields and m_fields_coeff
    EquationSystem::v_InitialiseFields();

    /// Create fields for velocity and passive scalars.
    // Get block attributes
    auto bAtr_phys =
        GetBlockAttributes<double, FieldState::Phys>(m_expansionLists[0]);

    // Create fields
    m_wsp_phys = Field<double, FieldState::Phys>(
        "wsp_phys", bAtr_phys, m_variablesFields.size(), m_npointsZ);
    m_advVel = Field<double, FieldState::Phys>("explicit advection velocity",
                                               bAtr_phys, m_variablesVel.size(),
                                               m_npointsZ);

    /// Create fields for pressure.
    // Get block attributes
    auto bAtr_phys_pressure = GetBlockAttributes<double, FieldState::Phys>(
        m_expansionLists[m_pressureIndex]);
    auto bAtr_coeff_pressure = GetBlockAttributes<double, FieldState::Coeff>(
        m_expansionLists[m_pressureIndex]);
    // Create fields
    m_pressure = Field<double, FieldState::Phys>(
        "pressure", bAtr_phys_pressure, m_variablesPressure.size(), m_npointsZ);
    m_pressure_coeff = Field<double, FieldState::Coeff>(
        "pressure coeff", bAtr_coeff_pressure, m_variablesPressure.size(),
        m_npointsZ);
    m_wsp_phys_deriv_pressure = Field<double, FieldState::Phys>(
        "wsp_phys_deriv_pressure", bAtr_phys_pressure, m_coordim, m_npointsZ);
    m_wsp_phys_1c =
        Field<double, FieldState::Phys>("wsp_phys_1c", bAtr_phys_pressure,
                                        m_variablesPressure.size(), m_npointsZ);
}

/*
 *  Set initial condition manually because of the
 *  auxiliary pressure field for incompressible Navier-Stokes.
 */
void VelocityCorrectionScheme::v_SetInitialConditions(double initialTime)
{
    /// Do initial conditions for velocity and additional scalar fields
    // Set initial conditions in m_fields using m_variables
    UnsteadySystem::v_SetInitialConditions(initialTime);

    /// Do initial conditions for auxiliary pressure field
    // Set time to initial time
    m_time = initialTime;

    // Initialise velocity, pressure and workspace fields
    m_math.zero(m_pressure);
    m_math.zero(m_pressure_coeff);

    // Set initial conditions
    auto vType = GetInitialConditionType(m_variablesPressure);
    SessionFunction initialConditions(
        m_session, m_expansionLists[m_pressureIndex], "InitialConditions");
    if (vType == LibUtilities::eFunctionTypeExpression)
    {
        initialConditions.EvaluateExpression(m_variablesPressure, m_pressure,
                                             m_time);

        // Project the pressure initial condition into coefficient space so the
        // first Stokes step has a usable pressure state.
        auto fwdTransPressureOp = FwdTransOp<double>::Create(
            m_expansionLists[m_pressureIndex], m_variablesPressure);
        auto preconPressureOp = PreconOp<double>::Create(
            m_expansionLists[m_pressureIndex], m_variablesPressure);
        auto linsolverPressureOp = LinearSolverOp<double>::Create(
            m_expansionLists[m_pressureIndex], m_variablesPressure);
        fwdTransPressureOp->SetLinearSolver(linsolverPressureOp);
        fwdTransPressureOp->SetPrecon(preconPressureOp);
        fwdTransPressureOp->UpdatePrecon();

        // Project initial conditions
        fwdTransPressureOp->Apply(m_pressure, m_pressure_coeff);
        m_bwdTransPressureOp->Apply(m_pressure_coeff, m_pressure);
    }
    // Set initial conditions from file
    else if (vType == LibUtilities::eFunctionTypeFile)
    {
        // Field files store modal coefficients. Load coefficient space first
        // and reconstruct physical values with the device support BwdTrans
        // operator.
        initialConditions.EvaluateFld(m_variablesPressure, m_pressure_coeff,
                                      m_time);
        m_bwdTransPressureOp->Apply(m_pressure_coeff, m_pressure);
    }

    // Print pressure initial condition
    if (m_session->GetComm()->GetRank() == 0)
    {
        for (const auto &varName : m_variablesPressure)
        {
            std::cout << "  - Field " << varName << ": "
                      << initialConditions.Describe(varName) << std::endl;
        }
    }
}

void VelocityCorrectionScheme::v_PrintNorms(std::ostream &out)
{
    /// Print norms for velocity and additional scalar fields
    EquationSystem::v_PrintNorms(out);

    /// Print norms for axuiliary pressure field
    // Create workspace Field
    auto wsp_phys = Field<double, FieldState::Phys>(
        "exact solution pressure",
        GetBlockAttributes<double, FieldState::Phys>(
            m_expansionLists[m_pressureIndex]),
        m_variablesPressure.size(), m_npointsZ);
    m_math.zero(wsp_phys);

    // Evaluate exact solution and compute diff to discrete solution
    if (m_session->DefinesFunction("ExactSolution"))
    {
        SessionFunction exactSolution(
            m_session, m_expansionLists[m_pressureIndex], "ExactSolution");
        exactSolution.EvaluateExpression(m_variablesPressure, wsp_phys, m_time);
    }
    m_math.sub(m_pressure, wsp_phys, wsp_phys);

    // // Pressure is determined only up to an arbitrary constant. Remove the
    // // mean pressure error before reporting pressure norms.
    // auto pressureDiff = wsp_phys.ToVector<double>();
    // auto ones = std::vector<double>(pressureDiff.size(), 1.0);
    // Array<OneD, const double> pressureDiffArray(pressureDiff.size(),
    //                                             pressureDiff.data());
    // Array<OneD, const double> onesArray(ones.size(), ones.data());
    // const double volume =
    //     m_expansionLists[m_pressureIndex]->Integral(onesArray);
    // const double meanPressureError =
    //     m_expansionLists[m_pressureIndex]->Integral(pressureDiffArray) /
    //     volume;
    // for (auto &value : pressureDiff)
    // {
    //     value -= meanPressureError;
    // }
    // wsp_phys.FromVector<NektarSpaces::HostSpace>(pressureDiff);

    // Compute L2 norm
    auto pressureL2NormOp = Operators::NormL2Op<double>::Create(
        m_expansionLists[m_pressureIndex], m_variablesPressure);
    pressureL2NormOp->Apply(wsp_phys);
    auto pressureL2Error = pressureL2NormOp->GetNorms()[0];

    // Compute Linf norm
    auto pressureLinfNormOp = Operators::NormLinfOp<double>::Create(
        m_expansionLists[m_pressureIndex], m_variablesPressure);
    pressureLinfNormOp->Apply(wsp_phys);
    auto pressureLinfError = pressureLinfNormOp->GetNorms()[0];

    // Print norms
    if (m_comm->GetRank() == 0)
    {
        out << "L 2 error (variable " << m_variablesPressure[0]
            << ") : " << pressureL2Error << std::endl;
        out << "L inf error (variable " << m_variablesPressure[0]
            << ") : " << pressureLinfError << std::endl;
    }
}

/**
 * @brief Load physical parameters from the session file.
 */
void VelocityCorrectionScheme::InitialiseParameters()
{
    // Get gamma parameter from session file.
    m_session->LoadParameter("Kinvis", m_kinvis, 1.0);
}

/*
 * Check if pressure field is singular that is has only
 * Neumann and/or Periodic boundary conditions.
 */
std::vector<bool> VelocityCorrectionScheme::v_GetSystemSingularChecks()
{
    unsigned int nVar = m_session->GetVariables().size();
    std::vector<bool> checks(nVar, false);

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

void VelocityCorrectionScheme::v_WriteFld(const std::string &outname)
{
    std::vector<std::vector<double>> fieldCoeffs;
    std::vector<std::string> variables(m_expansionLists.size());

    // Get vector with all component data and sort into vectors per component
    std::vector<double> allCoeffs = m_fields_coeff.ToVector<double>();
    auto ncomp                    = m_fields_coeff.GetNumComponents();
    auto ncoeffsTotal = allCoeffs.size(); // use allCoeffs to not have padding
    auto ncoeffsComp  = ncoeffsTotal / ncomp;
    fieldCoeffs.reserve(ncomp + 1); // Note +1 to add pressure
    for (unsigned int nc = 0; nc < ncomp; ++nc)
    {
        auto first = allCoeffs.begin() + nc * ncoeffsComp;
        auto last  = allCoeffs.begin() + (nc + 1) * ncoeffsComp;

        fieldCoeffs.emplace_back(std::make_move_iterator(first),
                                 std::make_move_iterator(last));
    }

    // TODO might have to interpolate pressure to velocity fields for correct
    // output? Add pressure to fieldCoeffs
    std::vector<double> pressureCoeffs = m_pressure_coeff.ToVector<double>();
    auto first                         = pressureCoeffs.begin();
    auto last                          = pressureCoeffs.end();
    fieldCoeffs.emplace_back(std::make_move_iterator(first),
                             std::make_move_iterator(last));

    // Note: FieldDef holds nummodes, basis, shapetype, variable strings, ..
    auto fieldDef = m_expansionLists[0]->GetFieldDefinitions();
    std::vector<std::vector<double>> fieldData(fieldDef.size());

    for (unsigned int nc = 0; nc < ncomp + 1; ++nc)
    {
        for (size_t i = 0; i < fieldDef.size(); ++i)
        {
            // Note Legacy transforms NodalToModal, if necessary, this code does
            // not
            fieldDef[i]->m_fields.push_back(m_variablesTotal[nc]);
            m_expansionLists[0]->AppendFieldData(fieldDef[i], fieldData[i],
                                                 fieldCoeffs[nc]);
        }
    }

    m_fieldMetaDataMap["Time"] = std::to_string(m_time);

    m_fieldIo->Write(outname, fieldDef, fieldData, m_fieldMetaDataMap,
                     m_session->GetBackups());
}
} // namespace Nektar
