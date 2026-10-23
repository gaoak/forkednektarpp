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

#include <LibUtilities/BasicUtils/StringUtils.hpp>
#include <MultiRegions/ElmtOps/Advection/AdvectionOp.hpp>
#include <MultiRegions/ElmtOps/AdvectionDealias/AdvectionDealiasOp.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/Divergence/DivergenceOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivOp.hpp>
#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/GlobalLinSysOps/LinearSolvers/LinearSolverOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/LinearSystemOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/PoissonSolve/PoissonSolveOp.hpp>
#include <SolverCore/MeanRemoval/MeanRemovalOp.hpp>
#include <SolverCore/NormOps/NormL2/NormL2Op.hpp>
#include <SolverCore/NormOps/NormLinf/NormLinfOp.hpp>
#include <SolverCore/PreconOps/PreconOp.hpp>
#include <SpatialDomains/Conditions.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <utility>

namespace Nektar
{
using namespace MultiRegions;

std::string VelocityCorrectionScheme::className =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "VelocityCorrectionScheme", VelocityCorrectionScheme::create,
        "Velocity correction scheme for solving incompressible Navier-Stokes "
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

    // Load formulation and physical parameters before field allocation since
    // the linear-implicit formulation requires additional workspaces.
    InitialiseParameters();

    /// Create Field for solution m_fields and others
    // Note we use expansionList for velocity (and added scalar) fields
    InitialiseFields();

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
    // The linear-implicit advection and the higher-order pressure boundary
    // data are combined with the same coefficients as the explicit velocity
    // contribution, for every IMEX step and startup stage.
    if (m_implicitAdvection || m_hasHighOrderPressureBCs)
    {
        m_timeOp->EnableExplicitContributionExtrapolation();
    }
}

void VelocityCorrectionScheme::v_GenerateSummary(SummaryList &s)
{
    UnsteadySystem::v_GenerateSummary(s);

    AddSummaryItem(s, "Equations", "Incompressible Navier-Stokes");
    AddSummaryItem(s, "Algorithm", "Velocity correction scheme");
    if (m_implicitAdvection)
    {
        AddSummaryItem(s, "Formulation", "Linear-implicit");
    }
    else
    {
        AddSummaryItem(s, "Formulation", "Semi-implicit");
    }
    AddSummaryItem(s, "Pressure form", m_weakPressure ? "Weak" : "Strong");

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
 *  param dt_inv_gamma: = \gamma \Delta t
 *
 *  Upon output
 *  param in: = u^{n+1}
 *  param out: = param in
 *
 *  The pressure solves
 *      \nabla^2 p^{n+1} = \nabla \cdot \mathbf{f}
 *  with \partial p / \partial n = \mathbf{n} \cdot (\mathbf{f}
 *      - \mathbf{w}^{n+1} / (\gamma \Delta t)
 *      - \nu \nabla \times \nabla \times \mathbf{u}^*)
 *  in the strong form, with \mathbf{f} = in / (\gamma \Delta t), and
 *      (\nabla \phi, \nabla p^{n+1}) = (\nabla \phi, \mathbf{f})
 *          - \int_\Gamma \phi \mathbf{n} \cdot (\mathbf{w}^{n+1} /
 *              (\gamma \Delta t) + \nu \nabla \times \nabla \times
 *              \mathbf{u}^*)
 *  in the weak form, on the higher-order pressure regions, where
 *  \mathbf{w} is the discrete Dirichlet velocity: the prescribed boundary
 *  velocity projected onto the boundary expansion, as the velocity solve
 *  imposes it. The weak form keeps the predictor, explicit advection and
 *  forcing in \mathbf{f} off the boundary.
 */
void VelocityCorrectionScheme::SolveUnsteadyStokesSystem(
    LibUtilities::Field<double, FieldState::Phys> &in,
    [[maybe_unused]] LibUtilities::Field<double, FieldState::Phys> &out,
    const double &time, const double &dt_inv_gamma)
{
    // Choose RHS for either semiimplicit or linearimplicit formulation
    LibUtilities::Field<double, FieldState::Phys> &fieldRhs =
        m_implicitAdvection ? m_wsp_fields_rhs : in;

    if (m_implicitAdvection)
    {
        auto &advectionRhsHistory = GetAdvectionRhsHistory(
            m_timeOp->GetExplicitContributionHistoryId(),
            m_timeOp->GetExplicitContributionHistorySize());
        m_timeOp->ExtrapolateExplicitContribution(advectionRhsHistory,
                                                  m_wsp_explicit_adv_rhs);
        m_math.daxpy(-1.0, m_wsp_explicit_adv_rhs, in, fieldRhs);
    }

    /// Set up forcing term for pressure Poisson equation
    if (m_weakPressure)
    {
        SetUpWeakPressureForcing(in, fieldRhs, dt_inv_gamma);
    }
    else
    {
        // Compute divergence of RHS. Keep the explicit advection
        // contribution in this RHS because the pressure projection requires
        // it.
        // TODO this should only apply to the velocity components!
        m_divergenceOp->Apply(in, m_wsp_phys_1c);

        // Scale divergence by 1 / (\gamma \Delta t)
        m_math.mul(1.0 / dt_inv_gamma, m_wsp_phys_1c, m_wsp_phys_1c);
    }

    // The histories hold the velocity and the explicit forcing. They are
    // combined with the coefficients of the explicit contributions, scaled to
    // sum to the implicit step size gamma * dt, by which the boundary
    // operators divide, so that a steady state gives the exact flux. For a
    // multistep scheme the coefficients extrapolate and sum to gamma; for a
    // later stage of an IMEXdirk scheme, such as the startup of a multistep
    // scheme, they are quadrature weights with another sum.
    //
    // The linear-implicit weak form takes the velocity of the curl-curl and
    // outflow terms at the latest level instead of extrapolating it, as it
    // takes its advection in SetUpWeakPressureForcing(). Extrapolating these
    // explicit terms of the pressure step limits the time step that the
    // implicit advection allows; the latest level keeps large time steps
    // stable at the cost of an O(dt) error in the pressure step, which
    // leaves the pressure near higher-order pressure regions close to first
    // order in time when the advection changes in time. The semi-implicit
    // weak form extrapolates the velocity as the strong form does.
    if (m_hasHighOrderPressureBCs)
    {
        const double pressureBCScale =
            dt_inv_gamma / m_timeOp->GetExplicitContributionCoefficientSum();
        auto &velocityHistory =
            GetVelocityHistory(m_timeOp->GetExplicitContributionHistoryId(),
                               m_timeOp->GetExplicitContributionHistorySize());
        if (m_weakPressure && m_implicitAdvection)
        {
            m_timeOp->LatestExplicitContribution(
                velocityHistory, m_pressureBCVelocity,
                SolverCore::TimeOpExtrapolationMode::Assign, pressureBCScale);
        }
        else
        {
            m_timeOp->ExtrapolateExplicitContribution(
                velocityHistory, m_pressureBCVelocity,
                SolverCore::TimeOpExtrapolationMode::Assign, pressureBCScale);
        }
        // The explicit forcing is extrapolated only for a region with a
        // velocity component that is not Dirichlet, whose flux is formed
        // from it.
        if (m_hasPressureBCExplicitRhs)
        {
            auto &pressureBCRhsHistory = GetPressureBCRhsHistory(
                m_timeOp->GetExplicitContributionHistoryId(),
                m_timeOp->GetExplicitContributionHistorySize());
            m_timeOp->ExtrapolateExplicitContribution(
                pressureBCRhsHistory, m_pressureBCExplicitRhs,
                SolverCore::TimeOpExtrapolationMode::Assign, pressureBCScale);
        }
        if (m_hoPressureBCOp)
        {
            m_hoPressureBCOp->SetKinvis(m_kinvis);
            m_hoPressureBCOp->SetTime(time);
            m_hoPressureBCOp->SetScale(1.0 / dt_inv_gamma);
            // The weak form integrates the predictor, advection and forcing
            // by parts onto the volume, so the boundary flux holds the
            // viscous term and the boundary acceleration only.
            if (m_weakPressure)
            {
                m_hoPressureBCOp->UpdateWeak(m_pressureBCExplicitRhs,
                                             m_pressureBCVelocity, in);
            }
            else
            {
                m_hoPressureBCOp->UpdateWithAcceleration(
                    m_pressureBCExplicitRhs, m_pressureBCVelocity, in);
            }
        }
        // The outflow pressure is a Dirichlet value, on which the weak
        // form's test functions vanish, so it is the same in both forms.
        if (m_hOutflowBCOp)
        {
            m_hOutflowBCOp->SetKinvis(m_kinvis);
            m_hOutflowBCOp->SetTime(time);
            m_hOutflowBCOp->SetScale(1.0 / dt_inv_gamma);
            m_hOutflowBCOp->UpdatePressure(m_pressureBCVelocity);
        }
    }

    // A singular (periodic) pressure system needs a zero-mean RHS; dealiased
    // advection's interpolate/project round trip can leave a small residual.
    // The forcing of the weak form is orthogonal to the constants, which
    // have no gradient, so it needs no mean removal.
    if (m_checkIfSystemSingular[m_pressureIndex] && !m_weakPressure)
    {
        if (!m_meanRemovalOp)
        {
            m_meanRemovalOp = SolverCore::MeanRemovalOp<double>::Create(
                m_expansionLists[m_pressureIndex], m_variablesPressure);
        }
        m_meanRemovalOp->Apply(m_wsp_phys_1c);
    }

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

    m_poissonSolveOp->UpdateBndCoeffs(time);
    if (m_weakPressure)
    {
        m_poissonSolveOp->ApplyRhs(m_wsp_coeff_1c, m_pressure_coeff);
    }
    else
    {
        m_poissonSolveOp->Apply(m_wsp_phys_1c, m_pressure_coeff);
    }

    /// Set up forcing term for Helmholtz problems
    // Evaluate p^{n+1} in physical space
    m_bwdTransPressureOp->Apply(m_pressure_coeff, m_pressure);

    // Compute derivative \nabla p^{n+1}
    m_physDerivPressureOp->Apply(m_pressure, m_wsp_phys_deriv_pressure);

    // Subtract the field-solve RHS/(aii_dt) and divide by kinvis. For the
    // linear-implicit formulation this RHS excludes explicit advection; the
    // pressure RHS above still includes it.
    m_math.daxpy(-1 / dt_inv_gamma, fieldRhs, m_wsp_phys_deriv_pressure,
                 m_wsp_phys);
    m_math.mul(1 / m_kinvis, m_wsp_phys, m_wsp_phys);

    /// Solve velocity systems
    // Update \lambda = 1 / (\gamma \Delta t \nu)
    m_lambda = 1.0 / dt_inv_gamma / m_kinvis;

    // Update LinearSystem
    if (m_implicitAdvection)
    {
        std::dynamic_pointer_cast<LinearADRSolveOp<double>>(m_fieldsSolveOp)
            ->SetLambda(m_lambda);

        // Update the advection velocity. The RHS above lost the explicit
        // advection dt * sum_i c_i (u_i . grad) u_i, with the coefficients
        // c_i of the explicit contributions and the velocities u_i of the
        // history. The solve puts it back linearised about the extrapolated
        // velocity u* = sum_i c_i u_i / sum_i c_i, as
        // dt * (sum_i c_i) (u* . grad) u. It is formulated after dividing
        // the momentum equation by the kinematic viscosity and weights the
        // advection by the implicit step size gamma * dt, so its advection
        // velocity is
        //     dt * (sum_i c_i) u* / (gamma * dt * nu)
        //         = dt * sum_i c_i u_i / (gamma * dt * nu),
        // in which the sum of the coefficients cancels. For a multistep
        // scheme the coefficients sum to gamma and u* extrapolates the
        // velocity to the new time level at the order of the scheme. For a
        // stage of an IMEXdirk scheme, such as the startup of a multistep
        // scheme, they are quadrature weights with another sum, and u* is a
        // weighted mean of the earlier stage values, accurate to first order
        // only, which limits a startup step to a second-order local error.
        auto &velocityHistory =
            GetVelocityHistory(m_timeOp->GetExplicitContributionHistoryId(),
                               m_timeOp->GetExplicitContributionHistorySize());
        m_timeOp->ExtrapolateExplicitContribution(
            velocityHistory, m_advVel,
            SolverCore::TimeOpExtrapolationMode::Assign,
            m_timeOp->GetTimeStep() / dt_inv_gamma / m_kinvis);
        std::dynamic_pointer_cast<LinearADRSolveOp<double>>(m_fieldsSolveOp)
            ->SetAdvVel(m_advVel);
    }
    else
    {
        std::dynamic_pointer_cast<HelmSolveOp<double>>(m_fieldsSolveOp)
            ->SetLambda(m_lambda);
    }

    // Update or re-use preconditioner
    if (m_implicitAdvection)
    {
        // The advection velocity changes the operator every step, so the
        // one preconditioner of the linear-implicit formulation is
        // configured again, in place, every step.
        m_fieldsSolveOp->SetPrecon(m_preconFieldsOp);
        m_fieldsSolveOp->UpdatePrecon();
    }
    else if (m_preconFieldsOpMap.find(dt_inv_gamma) ==
             m_preconFieldsOpMap.end())
    {
        // Create and cache preconditioner
        auto preconOp =
            PreconOp<double>::Create(m_expansionLists[0], m_variablesFields);
        m_preconFieldsOpMap.insert({dt_inv_gamma, preconOp});

        // Configure new preconditioner
        m_fieldsSolveOp->SetPrecon(preconOp);
        m_fieldsSolveOp->UpdatePrecon();
    }
    else
    {
        // Re-use preconditioner
        m_fieldsSolveOp->SetPrecon(m_preconFieldsOpMap[dt_inv_gamma]);
    }

    // Solve diffusion problem for each component. m_fields_coeff holds the
    // continuous solution of the previous solve, or the initial condition,
    // and is the initial guess; the solve replaces its Dirichlet
    // coefficients by the boundary values. The high-order outflow velocity
    // condition computes its flux from the solved pressure as the solve
    // applies it.
    m_fieldsSolveOp->UpdateBndCoeffs(time);
    if (m_hOutflowBCOp)
    {
        m_hOutflowBCOp->SetKinvis(m_kinvis);
        m_hOutflowBCOp->SetPressure(m_pressure);
    }
    m_fieldsSolveOp->Apply(m_wsp_phys, m_fields_coeff);

    // Transform to physical space
    m_bwdTransOp->Apply(m_fields_coeff, out);
}

/*
 *  @brief Set up the forcing of the weak pressure form in m_wsp_coeff_1c.
 *
 *  The forcing is (\nabla \phi, \mathbf{f}) with
 *      \mathbf{f} = \hat{\mathbf{u}} / (\gamma \Delta t)
 *          - [\mathbf{u} \cdot \nabla \mathbf{u}]^* + \mathbf{f}_b,
 *  the predictor, explicit advection and body forcing of the velocity
 *  equation. The semi-implicit formulation takes \mathbf{f} from in, which
 *  holds the extrapolated advection and forcing. The linear-implicit
 *  formulation takes its advection at the latest level only, added to
 *  fieldRhs, the RHS from which the extrapolated advection was removed,
 *  with the same total weight: extrapolated, this explicit advection would
 *  limit the time step that the implicit advection of the velocity solve
 *  allows. The inner product only reads the velocity components.
 *
 *  param in: = \hat{\mathbf{u}} + \gamma \Delta t (explicit contribution)
 *  param fieldRhs: = RHS of the velocity solve
 *  param dt_inv_gamma: = \gamma \Delta t
 */
void VelocityCorrectionScheme::SetUpWeakPressureForcing(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &fieldRhs,
    const double dt_inv_gamma)
{
    LibUtilities::Field<double, FieldState::Phys> *pressureRhs = &in;
    if (m_implicitAdvection)
    {
        // fieldRhs plus the latest advection, weighted as
        // TimeOp::LatestExplicitContribution() weights it, in one pass.
        auto &advectionRhsHistory = GetAdvectionRhsHistory(
            m_timeOp->GetExplicitContributionHistoryId(),
            m_timeOp->GetExplicitContributionHistorySize());
        m_math.daxpy(m_timeOp->GetExplicitContributionCoefficientSum(),
                     advectionRhsHistory.front(), fieldRhs, m_wsp_phys);
        pressureRhs = &m_wsp_phys;
    }

    m_iprodDerivPressureOp->SetScale(1.0 / dt_inv_gamma);
    m_iprodDerivPressureOp->Apply(*pressureRhs, m_wsp_coeff_1c);
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
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out, const double &time,
    const double &dt)
{
    // Store u for the advection velocity of the linear-implicit formulation
    // and for the higher-order pressure boundary conditions, so that u and
    // CurlCurl(u) are combined with the IMEX coefficients of the explicit
    // velocity contribution.
    if (m_implicitAdvection || m_hasHighOrderPressureBCs)
    {
        UpdateVelocityHistory(in);
    }

    EvaluateAdvectionContribution(in, out, time, dt);

    if (m_implicitAdvection)
    {
        // The linear-implicit solve moves advection to its left-hand side and
        // subtracts this extrapolated contribution from the IMEX RHS. Store
        // it before the body forcing is added, which stays on the RHS.
        UpdateAdvectionRhsHistory(out);
    }

    // Evaluate and add forcing terms. Advection has already written to out, so
    // all configured forcings append their contribution to the explicit RHS.
    for (auto &forcing : m_forcing)
    {
        forcing->SetAppend(true);
        forcing->Apply(in, out, time, dt);
    }

    // Store forcing - advection, out without its factor dt, for a
    // higher-order pressure region with a velocity component that is not
    // Dirichlet.
    if (m_hasPressureBCExplicitRhs)
    {
        UpdatePressureBCRhsHistory(out, dt);
    }
}

void VelocityCorrectionScheme::EvaluateAdvectionContribution(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out,
    [[maybe_unused]] const double &time, const double &dt)
{
    // The advection operators overwrite out, so it is not zeroed first.
    if (m_specHPDealiasing)
    {
        m_advectionDealiasOp->SetScale(-dt);
        m_advectionDealiasOp->SetAdvVel(in);
        m_advectionDealiasOp->Apply(in, out);
    }
    else
    {
        m_advectionOp->SetScale(-dt);
        m_advectionOp->SetAdvVel(in);
        m_advectionOp->Apply(in, out);
    }
}

/**
 * @brief The history of the explicit advection contribution of the scheme
 * with history id @p historyId, which combines @p historySize explicit
 * contributions in an implicit solve, created on first use.
 */
std::deque<LibUtilities::Field<double, FieldState::Phys>> &VelocityCorrectionScheme::
    GetAdvectionRhsHistory(unsigned int historyId, unsigned int historySize)
{
    ASSERTL0(m_implicitAdvection,
             "Advection RHS history is only used by the linear-implicit "
             "formulation.");

    FreeStartupHistories(historyId);
    auto [it, inserted] = m_advectionRhsHistories.try_emplace(historyId);
    auto &history       = it->second;
    if (inserted)
    {
        ASSERTL0(historySize > 0,
                 "The linear-implicit formulation requires a time integration "
                 "scheme whose implicit solves combine explicit "
                 "contributions, such as IMEX or IMEXdirk.");
        auto bAtrPhys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[0]);
        for (unsigned int i = 0; i < historySize; ++i)
        {
            history.emplace_back("advection rhs history " + std::to_string(i),
                                 bAtrPhys, m_variablesFields.size(),
                                 m_npointsZ);
            m_math.zero(history.back());
        }
    }

    return history;
}

/**
 * @brief The history of the velocity of the scheme with history id
 * @p historyId, which combines @p historySize explicit contributions in an
 * implicit solve, created on first use.
 *
 * The linear-implicit advection velocity and the higher-order boundary
 * conditions read only the velocity, which is the whole solution: they do
 * not support passive scalars.
 */
std::deque<LibUtilities::Field<double, FieldState::Phys>> &VelocityCorrectionScheme::
    GetVelocityHistory(unsigned int historyId, unsigned int historySize)
{
    FreeStartupHistories(historyId);
    auto [it, inserted] = m_velocityHistories.try_emplace(historyId);
    auto &history       = it->second;
    if (inserted)
    {
        ASSERTL0(historySize > 0,
                 "The linear-implicit formulation and the higher-order "
                 "pressure and outflow boundary conditions require a time "
                 "integration scheme whose implicit solves combine explicit "
                 "contributions, such as IMEX or IMEXdirk.");
        auto bAtrPhys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[0]);
        for (unsigned int i = 0; i < historySize; ++i)
        {
            history.emplace_back("velocity history " + std::to_string(i),
                                 bAtrPhys, m_variablesVel.size(), m_npointsZ);
            m_math.zero(history.back());
        }
    }

    return history;
}

/**
 * @brief The history of the explicit forcing of the scheme with history id
 * @p historyId, which combines @p historySize explicit contributions in an
 * implicit solve, created on first use.
 */
std::deque<LibUtilities::Field<double, FieldState::Phys>> &VelocityCorrectionScheme::
    GetPressureBCRhsHistory(unsigned int historyId, unsigned int historySize)
{
    FreeStartupHistories(historyId);
    auto [it, inserted] = m_pressureBCRhsHistories.try_emplace(historyId);
    auto &history       = it->second;
    if (inserted)
    {
        ASSERTL0(historySize > 0,
                 "The higher-order pressure boundary condition requires a "
                 "time integration scheme whose implicit solves combine "
                 "explicit contributions, such as IMEX or IMEXdirk.");
        auto bAtrPhys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[0]);
        for (unsigned int i = 0; i < historySize; ++i)
        {
            history.emplace_back("pressure BC rhs history " + std::to_string(i),
                                 bAtrPhys, m_variablesVel.size(), m_npointsZ);
            m_math.zero(history.back());
        }
    }

    return history;
}

/**
 * @brief Free the histories of the startup schemes once the main scheme
 * evaluates again.
 *
 * @details
 * Every startup step runs a scheme of its own, with a history id of its
 * own, which is dead once that step returns. The main scheme evaluates its
 * explicit RHS at the start of every step, so when @p historyId is the main
 * scheme's, every other history belongs to a finished startup step.
 */
void VelocityCorrectionScheme::FreeStartupHistories(unsigned int historyId)
{
    if (historyId != m_timeOp->GetOwnExplicitContributionHistoryId())
    {
        return;
    }
    for (auto *histories : {&m_advectionRhsHistories, &m_velocityHistories,
                            &m_pressureBCRhsHistories})
    {
        for (auto it = histories->begin(); it != histories->end();)
        {
            it = it->first == historyId ? std::next(it) : histories->erase(it);
        }
    }
}

void VelocityCorrectionScheme::UpdateAdvectionRhsHistory(
    LibUtilities::Field<double, FieldState::Phys> &advRhs)
{
    auto &history =
        GetAdvectionRhsHistory(m_timeOp->GetExplicitContributionHistoryId(),
                               m_timeOp->GetExplicitContributionHistorySize());

    auto tmp = std::move(history.back());
    history.pop_back();
    history.push_front(std::move(tmp));
    m_math.copy(advRhs, history.front());
}

void VelocityCorrectionScheme::UpdatePressureBCRhsHistory(
    LibUtilities::Field<double, FieldState::Phys> &rhs, const double dt)
{
    auto &history =
        GetPressureBCRhsHistory(m_timeOp->GetExplicitContributionHistoryId(),
                                m_timeOp->GetExplicitContributionHistorySize());

    auto tmp = std::move(history.back());
    history.pop_back();
    history.push_front(std::move(tmp));
    m_math.mul(1.0 / dt, rhs, history.front());
}

void VelocityCorrectionScheme::UpdateVelocityHistory(
    LibUtilities::Field<double, FieldState::Phys> &velocity)
{
    auto &history =
        GetVelocityHistory(m_timeOp->GetExplicitContributionHistoryId(),
                           m_timeOp->GetExplicitContributionHistorySize());

    auto tmp = std::move(history.back());
    history.pop_back();
    history.push_front(std::move(tmp));
    m_math.copy(velocity, history.front());
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

    // Create velocity operators. Only one of these two is actually used
    // (see EvaluateAdvectionContribution), selected by m_specHPDealiasing;
    // build only the one needed so the unused path doesn't pay for its own
    // operator/scratch-field setup.
    if (m_specHPDealiasing)
    {
        m_advectionDealiasOp = AdvectionDealiasOp<double>::Create(
            m_expansionLists[0], m_variablesFields);
    }
    else
    {
        m_advectionOp =
            AdvectionOp<double>::Create(m_expansionLists[0], m_variablesFields);
    }
    if (!m_weakPressure)
    {
        m_divergenceOp = DivergenceOp<double>::Create(m_expansionLists[0],
                                                      m_variablesFields);
    }

    if (m_implicitAdvection)
    {
        m_linearSolverFieldsOp = LinearSolverOp<double>::Create(
            m_expansionLists[0], m_variablesFields, "GMRES");
        auto linearADRSolveOp = LinearADRSolveOp<double>::Create(
            m_expansionLists[0], m_variablesFields);
        linearADRSolveOp->SetLinearSolver(m_linearSolverFieldsOp);
        linearADRSolveOp->SetDiffCoeff(m_diffCoeff);
        m_fieldsSolveOp = linearADRSolveOp;
        m_preconFieldsOp =
            PreconOp<double>::Create(m_expansionLists[0], m_variablesFields);
    }
    else
    {
        m_linearSolverFieldsOp = LinearSolverOp<double>::Create(
            m_expansionLists[0], m_variablesFields, "ConjGrad");
        auto helmSolveOp =
            HelmSolveOp<double>::Create(m_expansionLists[0], m_variablesFields);
        helmSolveOp->SetLinearSolver(m_linearSolverFieldsOp);
        helmSolveOp->SetDiffCoeff(m_diffCoeff);
        m_fieldsSolveOp = helmSolveOp;
    }

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

    if (m_weakPressure)
    {
        // The pressure inner product reads the velocity-equation RHS at the
        // pressure quadrature points.
        ASSERTL0(m_expansionLists[0]->SameQuadratureLayout(
                     m_expansionLists[m_pressureIndex]),
                 "The weak pressure form requires the velocity and pressure "
                 "expansions to share their quadrature points.");
        m_iprodDerivPressureOp =
            IProductWRTDerivBaseOp<FieldState::Coeff, double>::Create(
                m_expansionLists[m_pressureIndex], m_variablesPressure);
    }

    // The higher-order pressure and outflow condition operators are built
    // only for a session with a region of their user-defined type. A rank
    // holding none of the regions builds them with the others, since building
    // them is collective.
    //
    // The higher-order pressure condition applies the pressure Neumann
    // regions of its user-defined type in place of the default Neumann
    // operator of the pressure system.
    if (HasUserDefinedType(HOPressureBndCondOp<double>::userDefinedType,
                           m_variablesPressure))
    {
        m_hoPressureBCOp = HOPressureBndCondOp<double>::Create(
            m_expansionLists[m_pressureIndex], m_variablesPressure);
        m_hoPressureBCOp->SetVelocityContext(m_expansionLists[0],
                                             m_variablesVel);
        m_poissonSolveOp->AddNeumannBCOp(
            m_hoPressureBCOp, HOPressureBndCondOp<double>::userDefinedType);
    }

    // The forcing of the weak form is orthogonal to the constants, so a
    // singular pressure system is compatible only if its Neumann flux has a
    // zero surface integral, which the flux of the higher-order pressure
    // condition does not have in general. The weak form has no correction
    // for it.
    ASSERTL0(!(m_weakPressure && m_checkIfSystemSingular[m_pressureIndex] &&
               m_hoPressureBCOp),
             "The weak pressure form does not support a singular pressure "
             "system with the higher-order pressure boundary condition: give "
             "the pressure a Dirichlet boundary, or use "
             "PressureFormulation=Strong.");

    // Dong's high-order outflow condition couples a pressure Dirichlet
    // condition with a velocity Neumann condition, which share their
    // boundary data. The velocity condition applies the velocity Neumann
    // regions of its user-defined type in place of the default Neumann
    // operator of the velocity system. A velocity condition of the type
    // without its pressure condition also builds the operator, which refuses
    // it.
    std::vector<std::string> velocityAndPressure = m_variablesVel;
    velocityAndPressure.push_back(m_variablesPressure[0]);
    if (HasUserDefinedType(HOutflowBndCondOp<double>::userDefinedType,
                           velocityAndPressure))
    {
        m_hOutflowBCOp = HOutflowBndCondOp<double>::Create(
            m_expansionLists[m_pressureIndex], m_variablesPressure);
        m_hOutflowBCOp->SetVelocityContext(m_expansionLists[0], m_variablesVel);
        m_poissonSolveOp->AddDirichletBCOp(
            m_hOutflowBCOp->GetPressureDirichletOp());
        m_fieldsSolveOp->AddNeumannBCOp(
            m_hOutflowBCOp->GetVelocityNeumannOp(),
            HOutflowBndCondOp<double>::userDefinedType);
    }

    // The velocity history is filled from the solution, which holds only
    // the velocity while passive scalars are not supported with the
    // linear-implicit formulation or these conditions.
    ASSERTL0(!(m_implicitAdvection || m_hoPressureBCOp || m_hOutflowBCOp) ||
                 m_variablesFields.size() == m_variablesVel.size(),
             "The linear-implicit formulation and the higher-order pressure "
             "and outflow boundary conditions do not support passive "
             "scalars.");

    // The velocity history and its extrapolation are kept only on a rank
    // where either condition has a region.
    m_hasHighOrderPressureBCs =
        (m_hoPressureBCOp && m_hoPressureBCOp->GetNumBndDofs() > 0) ||
        (m_hOutflowBCOp &&
         m_hOutflowBCOp->GetPressureDirichletOp()->GetNumBndDofs() > 0);
    if (m_hasHighOrderPressureBCs)
    {
        m_pressureBCVelocity = LibUtilities::Field<double, FieldState::Phys>(
            "pressure BC velocity",
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[0],
                Operator<double>::GetDefaultInterleaveWidth(m_session)),
            m_variablesVel.size(), m_npointsZ);
    }

    // The explicit forcing history and its extrapolation are kept only on a
    // rank with a higher-order pressure region whose velocity is not
    // Dirichlet in every component.
    m_hasPressureBCExplicitRhs =
        m_hoPressureBCOp && m_hoPressureBCOp->HasNonDirichletVelocity();
    if (m_hasPressureBCExplicitRhs)
    {
        m_pressureBCExplicitRhs = LibUtilities::Field<double, FieldState::Phys>(
            "pressure BC explicit RHS",
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[0],
                Operator<double>::GetDefaultInterleaveWidth(m_session)),
            m_variablesVel.size(), m_npointsZ);
    }

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

    // Get interleave width
    auto interleaveWidth =
        Operator<double>::GetDefaultInterleaveWidth(m_session);

    /// Create fields for velocity and passive scalars.
    // Get block attributes
    auto bAtr_phys = MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
        m_expansionLists[0], interleaveWidth);

    // Create fields
    m_wsp_phys = LibUtilities::Field<double, FieldState::Phys>(
        "wsp_phys", bAtr_phys, m_variablesFields.size(), m_npointsZ);

    if (m_implicitAdvection)
    {
        m_advVel = LibUtilities::Field<double, FieldState::Phys>(
            "advection velocity", bAtr_phys, m_variablesVel.size(), m_npointsZ);
        m_wsp_fields_rhs = LibUtilities::Field<double, FieldState::Phys>(
            "wsp_fields_rhs", bAtr_phys, m_variablesFields.size(), m_npointsZ);
        m_wsp_explicit_adv_rhs = LibUtilities::Field<double, FieldState::Phys>(
            "wsp_explicit_adv_rhs", bAtr_phys, m_variablesFields.size(),
            m_npointsZ);
    }

    /// Create fields for pressure.
    // Get block attributes
    auto bAtr_phys_pressure =
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            m_expansionLists[m_pressureIndex], interleaveWidth);
    auto bAtr_coeff_pressure =
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            m_expansionLists[m_pressureIndex], interleaveWidth);
    // Create fields
    m_pressure = LibUtilities::Field<double, FieldState::Phys>(
        "pressure", bAtr_phys_pressure, m_variablesPressure.size(), m_npointsZ);
    m_pressure_coeff = LibUtilities::Field<double, FieldState::Coeff>(
        "pressure coeff", bAtr_coeff_pressure, m_variablesPressure.size(),
        m_npointsZ);
    m_wsp_phys_deriv_pressure = LibUtilities::Field<double, FieldState::Phys>(
        "wsp_phys_deriv_pressure", bAtr_phys_pressure, m_coordim, m_npointsZ);
    m_wsp_phys_1c = LibUtilities::Field<double, FieldState::Phys>(
        "wsp_phys_1c", bAtr_phys_pressure, m_variablesPressure.size(),
        m_npointsZ);
    if (m_weakPressure)
    {
        m_wsp_coeff_1c = LibUtilities::Field<double, FieldState::Coeff>(
            "wsp_coeff_1c", bAtr_coeff_pressure, m_variablesPressure.size(),
            m_npointsZ);
    }
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

    // Initialise pressure fields
    m_math.zero(m_pressure);
    m_math.zero(m_pressure_coeff);

    // The pressure is projected as the other fields are, its Dirichlet
    // values included; its mass matrix projection is built only for
    // Galerkin.
    SetContinuousInitialConditions(
        m_expansionLists[m_pressureIndex], m_variablesPressure, m_pressure,
        m_pressure_coeff, m_bwdTransPressureOp, nullptr);
    PrintInitialConditions(m_expansionLists[m_pressureIndex],
                           m_variablesPressure);
}

void VelocityCorrectionScheme::v_PrintNorms(std::ostream &out)
{
    /// Print norms for velocity and additional scalar fields
    EquationSystem::v_PrintNorms(out);

    /// Print norms for axuiliary pressure field
    // Create workspace Field, matching the interleave width of m_pressure so
    // that the subtraction below combines consistently laid-out data. This
    // is recomputed from the session rather than read off m_pressure's
    // blocks since GetBlocks() can be empty on a rank with no local elements.
    const auto interleaveWidth =
        Operator<double>::GetDefaultInterleaveWidth(m_session);
    auto wsp_phys = LibUtilities::Field<double, FieldState::Phys>(
        "exact solution pressure",
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            m_expansionLists[m_pressureIndex], interleaveWidth),
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

    // For a singular pressure system (Neumann/periodic boundaries only),
    // pressure is only determined up to an arbitrary constant, so remove
    // the mean pressure error before reporting pressure norms. For a
    // well-posed (non-singular) system the boundary conditions already fix
    // the gauge, so this correction is skipped rather than masking genuine
    // discretisation error.
    if (m_checkIfSystemSingular[m_pressureIndex])
    {
        // Execution-space dispatch (Serial/AVX/Device) is resolved entirely
        // inside MeanRemovalOp::Create() via the OperatorFactory, so no
        // opExecSpace check is needed here.
        if (!m_meanRemovalOp)
        {
            m_meanRemovalOp = SolverCore::MeanRemovalOp<double>::Create(
                m_expansionLists[m_pressureIndex], m_variablesPressure);
        }
        m_meanRemovalOp->Apply(wsp_phys);
    }

    // Compute L2 norm
    auto pressureL2NormOp = SolverCore::NormL2Op<double>::Create(
        m_expansionLists[m_pressureIndex], m_variablesPressure);
    pressureL2NormOp->Apply(wsp_phys);
    auto pressureL2Error = pressureL2NormOp->GetNorms()[0];

    // Compute Linf norm
    auto pressureLinfNormOp = SolverCore::NormLinfOp<double>::Create(
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

    // Get semi-implicit or linear-implicit time-stepping
    ASSERTL0(m_session->DefinesSolverInfo("FORMULATION"),
             "SolverInfo FORMULATION must be set to 'semiimplicit' or "
             "'linearimplicit'.");
    auto formulation =
        StringUtils::ToLowerCopy(m_session->GetSolverInfo("FORMULATION"));
    if (formulation == "linearimplicit")
    {
        m_implicitAdvection = true;
    }
    else if (formulation == "semiimplicit")
    {
        m_implicitAdvection = false;
    }
    else
    {
        std::stringstream ss;
        ss << "Unknown SolverInfo FORMULATION: " << formulation
           << ". Valid entries are 'semiimplicit' and 'linearimplicit'.";
        NEKERROR(ErrorUtil::efatal, ss.str());
    }

    // Strong or weak form of the pressure Poisson equation. The default is
    // the weak form with linear-implicit time-stepping, whose advection is
    // implicit, and the strong form with semi-implicit time-stepping.
    std::string pressureFormulation;
    m_session->LoadSolverInfo("PressureFormulation", pressureFormulation,
                              m_implicitAdvection ? "Weak" : "Strong");
    StringUtils::ToLower(pressureFormulation);
    if (pressureFormulation == "weak")
    {
        m_weakPressure = true;
    }
    else if (pressureFormulation == "strong")
    {
        m_weakPressure = false;
    }
    else
    {
        std::stringstream ss;
        ss << "Unknown SolverInfo PressureFormulation: " << pressureFormulation
           << ". Valid entries are 'strong' and 'weak'.";
        NEKERROR(ErrorUtil::efatal, ss.str());
    }

    // The linear-implicit formulation advects with a velocity that the
    // IMEXdirk startup stages combine from stage values, which limits it to
    // second order in time. Its weak form also takes the explicit terms of
    // the pressure equation at the latest level, which keeps large time
    // steps stable but lowers the order of the pressure near higher-order
    // pressure regions (SolveUnsteadyStokesSystem()).
    if (m_implicitAdvection && m_session->GetTimeIntScheme().order > 2)
    {
        const std::string message =
            "LinearImplicit is at most second order in time, not order " +
            std::to_string(m_session->GetTimeIntScheme().order) +
            ". PressureFormulation=Weak is more stable, with a lower-order "
            "pressure near higher-order pressure boundaries; "
            "PressureFormulation=Strong is more accurate but much less "
            "stable.";
        ROOTONLY_NEKERROR(ErrorUtil::ewarning, message.c_str());
    }

    // 3/2-rule spectral/hp dealiasing (local over-integration) for the
    // explicit nonlinear advection term. Mirrors legacy
    // EquationSystem.cpp's SPECTRALHPDEALIASING flag, but defaults to
    // enabled here (legacy defaults to disabled) now that the fused
    // AdvectionDealiasOp is validated across Serial/AVX/Device.
    m_session->MatchSolverInfo("SPECTRALHPDEALIASING", "True",
                               m_specHPDealiasing, true);
}

/**
 * @brief Whether a boundary region of any rank has a condition of
 * USERDEFINEDTYPE @p userDefinedType on one of @p variables.
 *
 * The boundary conditions hold only the regions on this rank's partition, so
 * the answer is reduced over the row communicator: every rank calls this, and
 * every rank receives the same answer.
 */
bool VelocityCorrectionScheme::HasUserDefinedType(
    const std::string &userDefinedType,
    const std::vector<std::string> &variables) const
{
    const SpatialDomains::BoundaryConditions bcs(m_session, m_graph);
    int hasType = 0;
    for (const auto &[regionId, bndCondMap] : bcs.GetBoundaryConditions())
    {
        for (const auto &variable : variables)
        {
            const auto condIt = bndCondMap->find(variable);
            if (condIt != bndCondMap->end() &&
                StringUtils::IEquals(condIt->second->GetUserDefined(),
                                     userDefinedType))
            {
                hasType = 1;
            }
        }
    }

    m_session->GetComm()->GetRowComm()->AllReduce(hasType,
                                                  LibUtilities::ReduceMax);
    return hasType != 0;
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

    // The boundary conditions hold only the regions on this rank's
    // partition. The pressure system is singular only if no rank has a
    // pressure boundary that is neither Neumann nor periodic.
    int nonSingular                     = pressureSingular ? 0 : 1;
    LibUtilities::CommSharedPtr rowComm = m_session->GetComm()->GetRowComm();
    rowComm->AllReduce(nonSingular, LibUtilities::ReduceMax);
    pressureSingular = (nonSingular == 0);

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

/**
 * @brief Offer the advection velocity to the Courant estimate.
 *
 * The incompressible equations carry no acoustic wave - pressure is
 * enforced rather than propagated - so the transport speed is the velocity
 * alone and there is no sound speed to weight.
 *
 * The velocity is read from the solution, whose leading components are the
 * velocity. m_advVel exists only for the linear-implicit formulation.
 */
LibUtilities::Field<double, FieldState::Phys> &VelocityCorrectionScheme::
    v_GetCFLVelocityField()
{
    return m_fields;
}

} // namespace Nektar
