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
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivOp.hpp>
#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/GlobalLinSysOps/LinearSolvers/LinearSolverOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/LinearSystemOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/PoissonSolve/PoissonSolveOp.hpp>
#include <SolverCore/MeanRemoval/MeanRemovalOp.hpp>
#include <SolverCore/PreconOps/PreconOp.hpp>
#include <SolverCore/TimeOps/IMEX/IMEXOp.hpp>
#include <SpatialDomains/Conditions.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
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
      m_diffCoeff(std::vector<double>(1.0)), m_lambda(1.0)
{
    ASSERTL0(m_projectionType == MultiRegions::eGalerkin,
             "The VelocityCorrectionScheme is only implemented for "
             "projectionType Galerkin");
}

/**
 * @brief Initialisation object for the unsteady diffusion problem.
 */
void VelocityCorrectionScheme::v_InitObject(bool declareExpansionLists)
{
    // Call to the initialisation object of EquationSystem
    UnsteadySystem::v_InitObject(declareExpansionLists);

    // Load formulation and physical parameters before field allocation since
    // the linear-implicit formulation requires additional workspaces.
    LoadSessionParameters();

    /// Create Fields for solution m_phys, m_coeff and others
    // One per expansion list: velocity (and added scalar) fields, and pressure
    v_InitialiseFields();

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Create and initialise all operators
    v_InitialiseOperators();

    // Configure Time-integration
    // Note we exclude pressure from time integration
    m_timeOp = TimeOp<double>::Create(m_expansionLists[0], m_variables[0]);
    m_timeOp->DefineProjection(&VelocityCorrectionScheme::DoProjection, this);
    // The linear-implicit advection and the higher-order pressure boundary
    // data keep histories that are combined with the coefficients of the
    // explicit contributions of every implicit solve.
    if (m_implicitAdvection || m_hasHighOrderPressureBCs)
    {
        auto timeOp = std::dynamic_pointer_cast<IMEXBaseOp<double>>(m_timeOp);
        ASSERTL0(timeOp,
                 "The linear-implicit formulation and the higher-order "
                 "pressure and outflow boundary conditions require a time "
                 "integration scheme whose implicit solves combine explicit "
                 "contributions, such as IMEX or IMEXdirk.");
        timeOp->DefineImplicitWithWeights(
            &VelocityCorrectionScheme::SolveUnsteadyStokesSystem, this);
        timeOp->DefineExplicitRhsWithWeights(
            &VelocityCorrectionScheme::EvaluateAdvection_SetPressureBCs, this);
    }
    else
    {
        m_timeOp->DefineImplicit(
            &VelocityCorrectionScheme::SolveUnsteadyStokesSystemWithoutHistory,
            this);
        m_timeOp->DefineExplicitRhs(
            &VelocityCorrectionScheme::
                EvaluateAdvection_SetPressureBCsWithoutHistory,
            this);
    }
    // The implicit coefficient of a multistep IMEX scheme is the sum of its
    // explicit contribution coefficients.
    if (m_hasMultistepStartup)
    {
        const auto coeffs =
            std::dynamic_pointer_cast<IMEXBaseOp<double>>(m_timeOp)
                ->GetExplicitCoefficients();
        m_startupImplicitCoefficient =
            std::accumulate(coeffs.begin(), coeffs.end(), 0.0);
    }
}

void VelocityCorrectionScheme::v_SetExpansionVariables()
{
    // Group the variables into velocity (and added scalars) and pressure.
    // For IncNavierStokes, we assume: u,v,w,theta,p in this order
    const auto variables             = m_session->GetVariables();
    const auto checkIfSystemSingular = v_GetSystemSingularChecks();

    // 1. velocities + passive scalar: u, v, w, theta
    m_variables.push_back(std::vector(variables.begin(), variables.end() - 1));
    m_checkIfSystemSingular.push_back(std::vector(
        checkIfSystemSingular.begin(), checkIfSystemSingular.end() - 1));
    // 1.a velocities : u, v, w
    m_variablesVel =
        std::vector(m_variables[m_velocityIndex].begin(),
                    m_variables[m_velocityIndex].begin() + m_coordim);
    // 1.b passive scalar fields: theta (not required)
    m_variablesAddScalars =
        std::vector(m_variables[m_velocityIndex].begin() + m_coordim,
                    m_variables[m_velocityIndex].end());

    // 2. pressure: p
    m_variables.push_back({variables.back()});
    m_checkIfSystemSingular.push_back({checkIfSystemSingular.back()});
}

FieldRegistry VelocityCorrectionScheme::v_GetFieldRegistry()
{
    // The velocity role leaves out the passive scalars, which the physical
    // groups of the base class retain.
    PhysicalFieldView velocity{
        m_phys[m_velocityIndex], m_expansionLists[m_velocityIndex], m_coordim,
        MakeFieldComponents(m_variables[m_velocityIndex], 0, m_coordim)};
    PhysicalFieldView pressure{
        m_phys[m_pressureIndex], m_expansionLists[m_pressureIndex], m_coordim,
        MakeFieldComponents(m_variables[m_pressureIndex])};

    FieldRegistry registry = UnsteadySystem::v_GetFieldRegistry();
    registry.SetRole(std::string(FieldRoles::Velocity), velocity);
    registry.SetRole(std::string(FieldRoles::Pressure), pressure);
    return registry;
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
    const double &time, const double &dt_inv_gamma,
    const ExplicitWeights &weights)
{
    if (m_implicitAdvection)
    {
        auto &advectionRhsHistory = GetAdvectionRhsHistory(weights);
        CombineExplicitContributions(weights, advectionRhsHistory,
                                     m_wsp_phys[m_velocityIndex]);
        m_math.daxpy(-1.0, m_wsp_phys[m_velocityIndex], in, m_wsp_fields_rhs);
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
            dt_inv_gamma / GetCoefficientSum(weights);
        auto &velocityHistory = GetVelocityHistory(weights);
        if (m_weakPressure && m_implicitAdvection)
        {
            // The latest level with the total weight of the extrapolation.
            m_math.mul(dt_inv_gamma, velocityHistory.front(),
                       m_pressureBCVelocity);
        }
        else
        {
            CombineExplicitContributions(weights, velocityHistory,
                                         m_pressureBCVelocity, pressureBCScale);
        }
        // The explicit forcing is extrapolated only for a region with a
        // velocity component that is not Dirichlet, whose flux is formed
        // from it.
        if (m_hasPressureBCExplicitRhs)
        {
            auto &pressureBCRhsHistory = GetPressureBCRhsHistory(weights);
            CombineExplicitContributions(weights, pressureBCRhsHistory,
                                         m_pressureBCExplicitRhs,
                                         pressureBCScale);
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

    // The forcing of a singular (all-Neumann or periodic) pressure system is
    // balanced against the surface integral of its Neumann values, which
    // are those of this step: the higher-order pressure flux just updated,
    // and the default Neumann values updated here.
    m_poissonSolveOp->UpdateBndCoeffs(time);

    // Choose RHS for either semiimplicit or linearimplicit formulation
    LibUtilities::Field<double, FieldState::Phys> &fieldRhs =
        m_implicitAdvection ? m_wsp_fields_rhs : in;

    /// Set up forcing term for pressure Poisson equation
    const double startupDivergenceWeight =
        GetStartupDivergenceWeight(dt_inv_gamma);
    if (m_weakPressure)
    {
        SetUpWeakPressureForcing(in, fieldRhs, dt_inv_gamma,
                                 startupDivergenceWeight, weights);
    }
    else
    {
        // Compute divergence of RHS. Keep the explicit advection
        // contribution in this RHS because the pressure projection requires
        // it. In a start-up stage the stage's initial velocity is removed
        // with weight beta first; see GetStartupDivergenceWeight().
        // TODO this should only apply to the velocity components!
        if (startupDivergenceWeight != 0.0)
        {
            m_math.daxpy(-startupDivergenceWeight, m_startupInitialSolution, in,
                         m_wsp_phys[m_velocityIndex]);
            m_divergenceOp->Apply(m_wsp_phys[m_velocityIndex],
                                  m_wsp_phys[m_pressureIndex]);
        }
        else
        {
            m_divergenceOp->Apply(in, m_wsp_phys[m_pressureIndex]);
        }

        // Scale divergence by 1 / (\gamma \Delta t)
        m_math.mul(1.0 / dt_inv_gamma, m_wsp_phys[m_pressureIndex],
                   m_wsp_phys[m_pressureIndex]);

        // A singular pressure system is solvable only when the domain
        // integral of its forcing matches the surface integral of its
        // Neumann values, which is not zero in general for a non-zero
        // Neumann condition, such as the higher-order pressure condition.
        // Driving the integral to it also removes the residual that
        // dealiased advection's interpolate/project round trip can leave.
        if (m_checkIfSystemSingular[m_pressureIndex][0])
        {
            m_meanRemovalOp[m_pressureIndex]->SetNeumannBalance(
                m_poissonSolveOp.get());
            m_meanRemovalOp[m_pressureIndex]->Apply(
                m_wsp_phys[m_pressureIndex]);
        }
    }

    /// Solve Pressure System
    // Update or re-use preconditioner
    if (m_preconPressureOpMap.find(dt_inv_gamma) == m_preconPressureOpMap.end())
    {
        // Configure and cache preconditioner
        m_preconPressureOpMap.insert(
            {dt_inv_gamma,
             PreconOp<double>::Create(m_expansionLists[m_pressureIndex],
                                      m_variables[m_pressureIndex])});
        m_poissonSolveOp->SetPrecon(m_preconPressureOpMap[dt_inv_gamma]);
        m_poissonSolveOp->UpdatePrecon();
    }
    else
    {
        // Re-use preconditioner
        m_poissonSolveOp->SetPrecon(m_preconPressureOpMap[dt_inv_gamma]);
    }

    if (m_weakPressure)
    {
        m_poissonSolveOp->ApplyRhs(m_wsp_coeff[m_pressureIndex],
                                   m_coeff[m_pressureIndex]);
    }
    else
    {
        m_poissonSolveOp->Apply(m_wsp_phys[m_pressureIndex],
                                m_coeff[m_pressureIndex]);
    }

    /// Set up forcing term for Helmholtz problems
    // Evaluate p^{n+1} in physical space
    m_bwdTransPressureOp->Apply(m_coeff[m_pressureIndex],
                                m_phys[m_pressureIndex]);

    // Compute derivative \nabla p^{n+1} into the velocity workspace
    m_physDerivPressureOp->Apply(m_phys[m_pressureIndex],
                                 m_wsp_phys[m_velocityIndex]);

    // Subtract the field-solve RHS/(aii_dt) and divide by kinvis. For the
    // linear-implicit formulation this RHS excludes explicit advection; the
    // pressure RHS above still includes it.
    m_math.daxpy(-1 / dt_inv_gamma, fieldRhs, m_wsp_phys[m_velocityIndex],
                 m_wsp_phys[m_velocityIndex]);
    m_math.mul(1 / m_kinvis, m_wsp_phys[m_velocityIndex],
               m_wsp_phys[m_velocityIndex]);

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
        auto &velocityHistory = GetVelocityHistory(weights);
        CombineExplicitContributions(weights, velocityHistory, m_advVel,
                                     m_timeOp->GetTimeStep() / dt_inv_gamma /
                                         m_kinvis);
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
        auto preconOp = PreconOp<double>::Create(
            m_expansionLists[m_velocityIndex], m_variables[m_velocityIndex]);
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

    // Solve diffusion problem for each component. m_coeff[m_velocityIndex]
    // holds the continuous solution of the previous solve, or the initial
    // condition, and is the initial guess; the solve replaces its Dirichlet
    // coefficients by the boundary values. The high-order outflow velocity
    // condition computes its flux from the solved pressure as the solve
    // applies it.
    m_fieldsSolveOp->UpdateBndCoeffs(time);
    if (m_hOutflowBCOp)
    {
        m_hOutflowBCOp->SetKinvis(m_kinvis);
        m_hOutflowBCOp->SetPressure(m_phys[m_pressureIndex]);
    }
    m_fieldsSolveOp->Apply(m_wsp_phys[m_velocityIndex],
                           m_coeff[m_velocityIndex]);

    // Transform to physical space
    m_bwdTransOp->Apply(m_coeff[m_velocityIndex], out);
}

/**
 * @brief Project the velocity onto the continuous space with its Dirichlet
 * values at time. The coefficients are left in m_coeff[m_velocityIndex],
 * of which out is then the backward transform, as after a velocity solve.
 */
void VelocityCorrectionScheme::DoProjection(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out, const double &time)
{
    m_fwdTransOp->UpdateBndCoeffs(time);
    m_fwdTransOp->Apply(in, m_coeff[m_velocityIndex]);
    m_bwdTransOp->Apply(m_coeff[m_velocityIndex], out);
}

/**
 * @brief SolveUnsteadyStokesSystem() for a solver that keeps no histories.
 */
void VelocityCorrectionScheme::SolveUnsteadyStokesSystemWithoutHistory(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out, const double &time,
    const double &lambda)
{
    SolveUnsteadyStokesSystem(in, out, time, lambda, {});
}

/*
 *  @brief Weight \beta of the stage's initial velocity in the pressure
 *  forcing of a start-up stage, or zero.
 *
 *  The velocity of the Helmholtz solve is not discretely divergence-free.
 *  An implicit stage of size \lambda_s \Delta t is forced by
 *  \nabla \cdot \hat{\mathbf{u}} / (\lambda_s \Delta t), so its pressure
 *  holds the divergence-cleaning part
 *      p_{div} = \nabla^{-2} (\nabla \cdot \mathbf{u}_s) /
 *          (\lambda_s \Delta t)
 *  of the solution \mathbf{u}_s the stage starts from: \mathbf{u}^n in the
 *  first stage of the step, the solution of the previous stage in a later
 *  one. A multistep scheme with implicit coefficient \lambda weights it by
 *  1 / (\lambda \Delta t). The one-step scheme that starts it up has a
 *  different \lambda_s, so p_{div}, and with it the wall pressure, jumps at
 *  every start and restart and rings while it relaxes.
 *
 *  A start-up stage removes
 *      (1 / (\lambda_s \Delta t) - 1 / (\lambda \Delta t))
 *          \nabla \cdot \mathbf{u}_s
 *          = \beta \nabla \cdot \mathbf{u}_s / (\lambda_s \Delta t),
 *      \beta = 1 - \lambda_s / \lambda,
 *  from the volume forcing, so p_{div} keeps the multistep weight. The
 *  stage's projection then removes only the fraction \lambda_s / \lambda of
 *  the divergence of \mathbf{u}_s, and the later stages and steps remove
 *  the rest. The term vanishes for a divergence-free \mathbf{u}_s. The
 *  boundary conditions keep the complete predictor: \mathbf{u}_s satisfies
 *  the Dirichlet velocity, so p_{div} has no boundary flux of its own.
 *
 *  Where the Helmholtz solve regenerates the divergence in every solve, as
 *  in a developed flow, \mathbf{u}^n and every stage solution have the same
 *  divergence. Where an earlier stage has removed part of the divergence of
 *  \mathbf{u}^n, as at a start from a field that is not divergence-free,
 *  the stage cleans what is left in \mathbf{u}_s: re-weighting the
 *  divergence of \mathbf{u}^n instead would remove divergence the earlier
 *  stage has already removed, and drive the stage pressure the other way.
 *
 *  \beta is zero outside the start-up stages of a multistep scheme, so the
 *  weight has no effect under a one-step scheme or on a steady state.
 *
 *  param dt_inv_gamma: = \lambda_s \Delta t of the active implicit solve
 */
double VelocityCorrectionScheme::GetStartupDivergenceWeight(
    const double dt_inv_gamma) const
{
    if (!IsStartupStep())
    {
        return 0.0;
    }
    return 1.0 - dt_inv_gamma /
                     (m_startupImplicitCoefficient * m_timeOp->GetTimeStep());
}

/**
 * @brief Whether the multistep scheme is taking one of its start-up steps,
 * whose implicit solves are start-up stages.
 */
bool VelocityCorrectionScheme::IsStartupStep() const
{
    return m_timeOp->GetStep() < m_startupSteps;
}

/*
 *  @brief Set up the forcing of the weak pressure form in
 *  m_wsp_coeff[m_pressureIndex].
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
 *  A physical-space forcing F of the strong form, \nabla^2 p = F, enters
 *  as -(\phi, F), ahead of the volume term. In a start-up stage F is
 *  -\beta \nabla \cdot \mathbf{u}_s / (\lambda_s \Delta t), the term the
 *  strong form removes from its divergence forcing; see
 *  GetStartupDivergenceWeight(). It is not
 *  -\beta (\nabla \phi, \mathbf{u}_s) / (\lambda_s \Delta t): that
 *  would also remove the boundary flux of \mathbf{u}_s, which the boundary
 *  acceleration term balances, and so drive a flux through the Neumann
 *  pressure boundaries.
 *
 *  For a singular pressure system the volume term is orthogonal to the
 *  constants, so the solvability condition requires the forcing to cancel
 *  the surface integral of the Neumann pressure values. Outside a start-up
 *  stage, the constant c whose domain integral is that surface integral
 *  enters as the forcing -(\phi, c) = -c (\phi, 1), from the integrals
 *  (\phi, 1) of the pressure basis computed at set-up. In a start-up stage
 *  F is shifted by the constant that makes its domain integral that
 *  surface integral.
 *
 *  param in: = \hat{\mathbf{u}} + \gamma \Delta t (explicit contribution)
 *  param fieldRhs: = RHS of the velocity solve
 *  param dt_inv_gamma: = \gamma \Delta t
 *  param startupDivergenceWeight: = \beta, zero outside a start-up stage
 */
void VelocityCorrectionScheme::SetUpWeakPressureForcing(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &fieldRhs,
    const double dt_inv_gamma, const double startupDivergenceWeight,
    const ExplicitWeights &weights)
{
    LibUtilities::Field<double, FieldState::Phys> *pressureRhs = &in;
    if (m_implicitAdvection)
    {
        // fieldRhs plus the latest advection, with the total weight of the
        // extrapolation, in one pass.
        auto &advectionRhsHistory = GetAdvectionRhsHistory(weights);
        m_math.daxpy(GetCoefficientSum(weights), advectionRhsHistory.front(),
                     fieldRhs, m_wsp_phys[m_velocityIndex]);
        pressureRhs = &m_wsp_phys[m_velocityIndex];
    }

    const bool singular = m_checkIfSystemSingular[m_pressureIndex][0];
    const bool startup  = startupDivergenceWeight != 0.0;
    if (startup)
    {
        m_divergenceOp->Apply(m_startupInitialSolution,
                              m_wsp_phys[m_pressureIndex]);
        m_math.mul(-startupDivergenceWeight / dt_inv_gamma,
                   m_wsp_phys[m_pressureIndex], m_wsp_phys[m_pressureIndex]);
        if (singular)
        {
            // Shift F by the constant that makes its domain integral the
            // surface integral of the Neumann values.
            m_meanRemovalOp[m_pressureIndex]->SetNeumannBalance(
                m_poissonSolveOp.get());
            m_meanRemovalOp[m_pressureIndex]->Apply(
                m_wsp_phys[m_pressureIndex]);
        }
        m_iprodPressureOp->SetScale(-1.0);
        m_iprodPressureOp->Apply(m_wsp_phys[m_pressureIndex],
                                 m_wsp_coeff[m_pressureIndex]);
    }

    m_iprodDerivPressureOp->SetAppend(startup);
    m_iprodDerivPressureOp->SetScale(1.0 / dt_inv_gamma);
    m_iprodDerivPressureOp->Apply(*pressureRhs, m_wsp_coeff[m_pressureIndex]);

    if (singular && !startup)
    {
        // The constant whose domain integral is the surface integral of the
        // Neumann values.
        m_meanRemovalOp[m_pressureIndex]->SetNeumannBalance(
            m_poissonSolveOp.get());
        const double balancedMean =
            m_meanRemovalOp[m_pressureIndex]->GetBalancedMean()[0];

        m_math.daxpy(-balancedMean, m_pressureBasisIntegrals,
                     m_wsp_coeff[m_pressureIndex],
                     m_wsp_coeff[m_pressureIndex]);
    }
}

/*
 *  @brief Evaluate explicit advection term, forcing terms
 *  and higher-order pressure boundary conditions.
 *
 *  Upon input
 *  param in: = U^{n} = [u^{n}, v^{n}, ...]
 *            = "m_phys[m_velocityIndex]" member variable of this solver
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
    const double &dt, const ExplicitWeights &weights)
{
    // A start-up stage evaluates its explicit RHS at the solution it starts
    // from, just before its implicit solve.
    if (IsStartupStep())
    {
        m_math.copy(in, m_startupInitialSolution);
    }

    // Store u for the advection velocity of the linear-implicit formulation
    // and for the higher-order pressure boundary conditions, so that u and
    // CurlCurl(u) are combined with the IMEX coefficients of the explicit
    // velocity contribution.
    if (m_implicitAdvection || m_hasHighOrderPressureBCs)
    {
        UpdateVelocityHistory(in, weights);
    }

    EvaluateAdvectionContribution(in, out, time, dt);

    if (m_implicitAdvection)
    {
        // The linear-implicit solve moves advection to its left-hand side and
        // subtracts this extrapolated contribution from the IMEX RHS. Store
        // it before the body forcing is added, which stays on the RHS.
        UpdateAdvectionRhsHistory(out, weights);
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
        UpdatePressureBCRhsHistory(out, dt, weights);
    }
}

/**
 * @brief EvaluateAdvection_SetPressureBCs() for a solver that keeps no
 * histories.
 */
void VelocityCorrectionScheme::EvaluateAdvection_SetPressureBCsWithoutHistory(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out, const double &time,
    const double &dt)
{
    EvaluateAdvection_SetPressureBCs(in, out, time, dt, {});
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
 * that @p weights belong to, created on first use.
 */
std::deque<LibUtilities::Field<double, FieldState::Phys>> &VelocityCorrectionScheme::
    GetAdvectionRhsHistory(const ExplicitWeights &weights)
{
    ASSERTL0(m_implicitAdvection,
             "Advection RHS history is only used by the linear-implicit "
             "formulation.");

    FreeStartupHistories(weights.historyId);
    auto [it, inserted] =
        m_advectionRhsHistories.try_emplace(weights.historyId);
    auto &history = it->second;
    if (inserted)
    {
        ASSERTL0(weights.historySize > 0,
                 "The linear-implicit formulation requires a time integration "
                 "scheme whose implicit solves combine explicit "
                 "contributions, such as IMEX or IMEXdirk.");
        auto bAtrPhys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[m_velocityIndex]);
        for (unsigned int i = 0; i < weights.historySize; ++i)
        {
            history.emplace_back("advection rhs history " + std::to_string(i),
                                 bAtrPhys, m_variables[m_velocityIndex].size(),
                                 m_npointsZ);
            m_math.zero(history.back());
        }
    }

    return history;
}

/**
 * @brief The history of the velocity of the scheme that @p weights
 * belongs to, created on first use.
 *
 * The linear-implicit advection velocity and the higher-order boundary
 * conditions read only the velocity, which is the whole solution: they do
 * not support passive scalars.
 */
std::deque<LibUtilities::Field<double, FieldState::Phys>> &VelocityCorrectionScheme::
    GetVelocityHistory(const ExplicitWeights &weights)
{
    FreeStartupHistories(weights.historyId);
    auto [it, inserted] = m_velocityHistories.try_emplace(weights.historyId);
    auto &history       = it->second;
    if (inserted)
    {
        ASSERTL0(weights.historySize > 0,
                 "The linear-implicit formulation and the higher-order "
                 "pressure and outflow boundary conditions require a time "
                 "integration scheme whose implicit solves combine explicit "
                 "contributions, such as IMEX or IMEXdirk.");
        auto bAtrPhys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[m_velocityIndex]);
        for (unsigned int i = 0; i < weights.historySize; ++i)
        {
            history.emplace_back("velocity history " + std::to_string(i),
                                 bAtrPhys, m_variablesVel.size(), m_npointsZ);
            m_math.zero(history.back());
        }
    }

    return history;
}

/**
 * @brief The history of the explicit forcing of the scheme that
 * @p weights belong to, created on first use.
 */
std::deque<LibUtilities::Field<double, FieldState::Phys>> &VelocityCorrectionScheme::
    GetPressureBCRhsHistory(const ExplicitWeights &weights)
{
    FreeStartupHistories(weights.historyId);
    auto [it, inserted] =
        m_pressureBCRhsHistories.try_emplace(weights.historyId);
    auto &history = it->second;
    if (inserted)
    {
        ASSERTL0(weights.historySize > 0,
                 "The higher-order pressure boundary condition requires a "
                 "time integration scheme whose implicit solves combine "
                 "explicit contributions, such as IMEX or IMEXdirk.");
        auto bAtrPhys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[m_velocityIndex]);
        for (unsigned int i = 0; i < weights.historySize; ++i)
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
 * own, which is dead once that step returns. Outside the startup steps
 * only the main scheme evaluates, so @p historyId is the main scheme's and
 * every other history belongs to a finished startup step.
 */
void VelocityCorrectionScheme::FreeStartupHistories(unsigned int historyId)
{
    if (IsStartupStep())
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
    LibUtilities::Field<double, FieldState::Phys> &advRhs,
    const ExplicitWeights &weights)
{
    auto &history = GetAdvectionRhsHistory(weights);

    auto tmp = std::move(history.back());
    history.pop_back();
    history.push_front(std::move(tmp));
    m_math.copy(advRhs, history.front());
}

void VelocityCorrectionScheme::UpdatePressureBCRhsHistory(
    LibUtilities::Field<double, FieldState::Phys> &rhs, const double dt,
    const ExplicitWeights &weights)
{
    auto &history = GetPressureBCRhsHistory(weights);

    auto tmp = std::move(history.back());
    history.pop_back();
    history.push_front(std::move(tmp));
    m_math.mul(1.0 / dt, rhs, history.front());
}

void VelocityCorrectionScheme::UpdateVelocityHistory(
    LibUtilities::Field<double, FieldState::Phys> &velocity,
    const ExplicitWeights &weights)
{
    auto &history = GetVelocityHistory(weights);

    auto tmp = std::move(history.back());
    history.pop_back();
    history.push_front(std::move(tmp));
    m_math.copy(velocity, history.front());
}

/**
 * @brief out = scale * sum_i c_i history[i], with the coefficients c_i of the
 * explicit contributions of the implicit solve, in one pass over out; levels
 * of zero coefficient are skipped.
 */
void VelocityCorrectionScheme::CombineExplicitContributions(
    const ExplicitWeights &weights,
    std::deque<LibUtilities::Field<double, FieldState::Phys>> &history,
    LibUtilities::Field<double, FieldState::Phys> &out, const double scale)
{
    const auto &coeffs = weights.coefficients;
    ASSERTL0(!coeffs.empty(),
             "No explicit contribution coefficients are available for this "
             "implicit solve.");
    ASSERTL0(history.size() >= coeffs.size(),
             "Explicit contribution history is shorter than the current "
             "coefficient list.");

    std::vector<double> alpha;
    std::vector<LibUtilities::Field<double, FieldState::Phys> *> x;
    for (unsigned int i = 0; i < coeffs.size(); ++i)
    {
        if (coeffs[i] != 0.0)
        {
            alpha.push_back(scale * coeffs[i]);
            x.push_back(&history[i]);
        }
    }
    m_math.linearCombination(alpha, x, out, false);
}

/**
 * @brief Sum of the coefficients with which the implicit solve combines the
 * explicit contributions. It is the implicit coefficient gamma of a multistep
 * IMEX scheme but, in general, not that of an IMEXdirk stage, whose
 * coefficients are quadrature weights.
 */
double VelocityCorrectionScheme::GetCoefficientSum(
    const ExplicitWeights &weights)
{
    const auto &coeffs = weights.coefficients;
    ASSERTL0(!coeffs.empty(),
             "No explicit contribution coefficients are available for this "
             "implicit solve.");
    return std::accumulate(coeffs.begin(), coeffs.end(), 0.0);
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
    const auto nDiffCoeff = m_coordim * (m_coordim + 1) / 2;
    m_diffCoeff.resize(nDiffCoeff);

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
    // Velocity transforms
    m_bwdTransOp = BwdTransOp<double>::Create(m_expansionLists[m_velocityIndex],
                                              m_variables[m_velocityIndex]);
    m_fwdTransOp = CreateFwdTransOp(m_expansionLists[m_velocityIndex],
                                    m_variables[m_velocityIndex]);

    // Create velocity operators. Only one of these two is actually used
    // (see EvaluateAdvectionContribution), selected by m_specHPDealiasing;
    // build only the one needed so the unused path doesn't pay for its own
    // operator/scratch-field setup.
    if (m_specHPDealiasing)
    {
        m_advectionDealiasOp = AdvectionDealiasOp<double>::Create(
            m_expansionLists[m_velocityIndex], m_variables[m_velocityIndex]);
    }
    else
    {
        m_advectionOp = AdvectionOp<double>::Create(
            m_expansionLists[m_velocityIndex], m_variables[m_velocityIndex]);
    }
    // The weak form takes a divergence only in a start-up stage.
    if (!m_weakPressure || m_hasMultistepStartup)
    {
        m_divergenceOp = DivergenceOp<double>::Create(
            m_expansionLists[m_velocityIndex], m_variables[m_velocityIndex]);
    }

    if (m_implicitAdvection)
    {
        m_linearSolverFieldsOp = LinearSolverOp<double>::Create(
            m_expansionLists[m_velocityIndex], m_variables[m_velocityIndex],
            "GMRES");
        auto linearADRSolveOp = LinearADRSolveOp<double>::Create(
            m_expansionLists[m_velocityIndex], m_variables[m_velocityIndex]);
        linearADRSolveOp->SetLinearSolver(m_linearSolverFieldsOp);
        linearADRSolveOp->SetDiffCoeff(m_diffCoeff);
        m_fieldsSolveOp  = linearADRSolveOp;
        m_preconFieldsOp = PreconOp<double>::Create(
            m_expansionLists[m_velocityIndex], m_variables[m_velocityIndex]);
    }
    else
    {
        m_linearSolverFieldsOp = LinearSolverOp<double>::Create(
            m_expansionLists[m_velocityIndex], m_variables[m_velocityIndex],
            "ConjGrad");
        auto helmSolveOp = HelmSolveOp<double>::Create(
            m_expansionLists[m_velocityIndex], m_variables[m_velocityIndex]);
        helmSolveOp->SetLinearSolver(m_linearSolverFieldsOp);
        helmSolveOp->SetDiffCoeff(m_diffCoeff);
        m_fieldsSolveOp = helmSolveOp;
    }

    // The velocity and pressure are combined point by point: the velocity
    // divergence forces the pressure, the pressure gradient corrects the
    // velocity, and the weak pressure form takes the inner product of the
    // velocity-equation RHS with the pressure basis. Their expansions may
    // differ in order, but must share their quadrature and phys layout.
    ASSERTL0(m_expansionLists[m_velocityIndex]->SameQuadratureLayout(
                 m_expansionLists[m_pressureIndex]),
             "The velocity and pressure expansions must share their quadrature "
             "points.");

    // Create pressure operators
    m_bwdTransPressureOp = BwdTransOp<double>::Create(
        m_expansionLists[m_pressureIndex], m_variables[m_pressureIndex]);
    m_physDerivPressureOp = PhysDerivOp<double>::Create(
        m_expansionLists[m_pressureIndex], m_variables[m_pressureIndex]);
    m_poissonSolveOp = PoissonSolveOp<double>::Create(
        m_expansionLists[m_pressureIndex], m_variables[m_pressureIndex]);
    m_linearSolverPressureOp = LinearSolverOp<double>::Create(
        m_expansionLists[m_pressureIndex], m_variables[m_pressureIndex],
        "ConjGrad");

    // Configure PoissonSolve
    m_poissonSolveOp->SetLinearSolver(m_linearSolverPressureOp);
    m_poissonSolveOp->SetDiffCoeff(m_diffCoeff);

    // A singular pressure system balances its forcing against its Neumann
    // values, and its reported error has zero mean.
    if (m_checkIfSystemSingular[m_pressureIndex][0])
    {
        m_meanRemovalOp.resize(m_expansionLists.size());
        m_meanRemovalOp[m_pressureIndex] = MeanRemovalOp<double>::Create(
            m_expansionLists[m_pressureIndex], m_variables[m_pressureIndex]);
    }

    if (m_weakPressure)
    {
        m_iprodDerivPressureOp =
            IProductWRTDerivBaseOp<FieldState::Coeff, double>::Create(
                m_expansionLists[m_pressureIndex],
                m_variables[m_pressureIndex]);
        if (m_checkIfSystemSingular[m_pressureIndex][0] ||
            m_hasMultistepStartup)
        {
            m_iprodPressureOp = IProductWRTBaseOp<double>::Create(
                m_expansionLists[m_pressureIndex],
                m_variables[m_pressureIndex]);
        }
        // The integrals (phi, 1) of the pressure basis, from which the
        // constant forcing that balances a singular system is formed.
        if (m_checkIfSystemSingular[m_pressureIndex][0])
        {
            m_math.fill(1.0, m_wsp_phys[m_pressureIndex]);
            m_iprodPressureOp->SetScale(1.0);
            m_iprodPressureOp->Apply(m_wsp_phys[m_pressureIndex],
                                     m_pressureBasisIntegrals);
        }
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
                           m_variables[m_pressureIndex]))
    {
        m_hoPressureBCOp = HOPressureBndCondOp<double>::Create(
            m_expansionLists[m_pressureIndex], m_variables[m_pressureIndex]);
        m_hoPressureBCOp->SetVelocityContext(m_expansionLists[m_velocityIndex],
                                             m_variablesVel);
        m_poissonSolveOp->AddNeumannBCOp(
            m_hoPressureBCOp, HOPressureBndCondOp<double>::userDefinedType);
    }

    // Dong's high-order outflow condition couples a pressure Dirichlet
    // condition with a velocity Neumann condition, which share their
    // boundary data. The velocity condition applies the velocity Neumann
    // regions of its user-defined type in place of the default Neumann
    // operator of the velocity system. A velocity condition of the type
    // without its pressure condition also builds the operator, which refuses
    // it.
    std::vector<std::string> velocityAndPressure = m_variablesVel;
    velocityAndPressure.push_back(m_variables[m_pressureIndex][0]);
    if (HasUserDefinedType(HOutflowBndCondOp<double>::userDefinedType,
                           velocityAndPressure))
    {
        m_hOutflowBCOp = HOutflowBndCondOp<double>::Create(
            m_expansionLists[m_pressureIndex], m_variables[m_pressureIndex]);
        m_hOutflowBCOp->SetVelocityContext(m_expansionLists[m_velocityIndex],
                                           m_variablesVel);
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
                 m_variables[m_velocityIndex].size() == m_variablesVel.size(),
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
                m_expansionLists[m_velocityIndex]),
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
                m_expansionLists[m_velocityIndex]),
            m_variablesVel.size(), m_npointsZ);
    }

    // Load forcing terms, if defined in the session file. For IncNS the
    // pressure variable is excluded, so forcing acts on the velocity and
    // passive scalar fields.
    m_forcing = Forcing::Load(m_session, m_expansionLists[m_velocityIndex],
                              m_variables[m_velocityIndex]);
}

/*
 *  Create and initialise Fields.
 */
void VelocityCorrectionScheme::v_InitialiseFields()
{
    // EquationSystem initialises m_phys, m_coeff and their workspaces for
    // velocity and pressure
    EquationSystem::v_InitialiseFields();

    // Each Field takes the default interleave widths of its own expansion
    // list, since the default implementation, and so the width, follows the
    // shape and order of its blocks.

    /// Create fields for velocity and passive scalars.
    // Get block attributes
    auto bAtr_phys = MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
        m_expansionLists[m_velocityIndex]);

    if (m_implicitAdvection)
    {
        m_advVel = LibUtilities::Field<double, FieldState::Phys>(
            "advection velocity", bAtr_phys, m_variablesVel.size(), m_npointsZ);
        m_wsp_fields_rhs = LibUtilities::Field<double, FieldState::Phys>(
            "wsp_fields_rhs", bAtr_phys, m_variables[m_velocityIndex].size(),
            m_npointsZ);
    }
    if (m_hasMultistepStartup)
    {
        m_startupInitialSolution =
            LibUtilities::Field<double, FieldState::Phys>(
                "startup initial solution", bAtr_phys,
                m_variables[m_velocityIndex].size(), m_npointsZ);
    }

    /// Create fields for pressure.
    // Get block attributes
    auto bAtr_coeff =
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            m_expansionLists[m_pressureIndex]);
    if (m_weakPressure && m_checkIfSystemSingular[m_pressureIndex][0])
    {
        m_pressureBasisIntegrals =
            LibUtilities::Field<double, FieldState::Coeff>(
                "pressure basis integrals", bAtr_coeff,
                m_variables[m_pressureIndex].size(), m_npointsZ);
    }
}

/*
 *  Set initial condition manually because of the
 *  auxiliary pressure field for incompressible Navier-Stokes.
 */
void VelocityCorrectionScheme::v_SetInitialConditions(double initialTime)
{
    /// Do initial conditions for velocity and additional scalar fields
    UnsteadySystem::v_SetInitialConditions(initialTime);

    /// Do initial conditions for auxiliary pressure field
    // Set time to initial time
    m_time = initialTime;

    // Initialise pressure fields
    m_math.zero(m_phys[m_pressureIndex]);
    m_math.zero(m_coeff[m_pressureIndex]);

    // The pressure is evaluated on the velocity expansion, whose quadrature
    // points it shares, so a file written at the velocity order is read at
    // that order, and projected element by element onto its own expansion.
    SessionFunction pressureConditions(
        m_session, m_expansionLists[m_velocityIndex],
        m_variables[m_pressureIndex], "InitialConditions");
    pressureConditions.Evaluate(m_phys[m_pressureIndex], m_time);
    auto iprod = LibUtilities::Field<double, FieldState::Coeff>(
        "pressure inner product",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            m_expansionLists[m_pressureIndex]),
        m_variables[m_pressureIndex].size(), m_npointsZ);
    IProductWRTBaseOp<double>::Create(m_expansionLists[m_pressureIndex],
                                      m_variables[m_pressureIndex])
        ->Apply(m_phys[m_pressureIndex], iprod);
    MultiplyByElmtInvMassOp<double>::Create(m_expansionLists[m_pressureIndex],
                                            m_variables[m_pressureIndex])
        ->Apply(iprod, m_coeff[m_pressureIndex]);

    // The pressure is then made continuous as the other fields are, its
    // Dirichlet values included; its mass matrix projection is built only
    // for Galerkin.
    SetContinuousInitialConditions(
        m_expansionLists[m_pressureIndex], m_variables[m_pressureIndex],
        m_phys[m_pressureIndex], m_coeff[m_pressureIndex], m_bwdTransPressureOp,
        nullptr);
    PrintInitialConditions(pressureConditions);
}

/**
 * @brief Load physical parameters from the session file.
 */
void VelocityCorrectionScheme::LoadSessionParameters()
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

    // A multistep IMEX scheme of order two or more starts with IMEXdirk
    // steps, whose stages keep the multistep weight of the divergence of
    // their initial velocity in the pressure forcing; see
    // GetStartupDivergenceWeight().
    const auto &timeIntScheme = m_session->GetTimeIntScheme();
    m_hasMultistepStartup     = timeIntScheme.method == IMEXOp<double>::name &&
                            timeIntScheme.order >= 2;
    if (m_hasMultistepStartup)
    {
        m_startupSteps = timeIntScheme.order - 1;
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
    std::vector<bool> checks(m_session->GetVariables().size(), false);

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

    checks.back() = pressureSingular;
    return checks;
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
    return m_phys[m_velocityIndex];
}

} // namespace Nektar
