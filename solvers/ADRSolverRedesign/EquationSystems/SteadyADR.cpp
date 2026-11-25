///////////////////////////////////////////////////////////////////////////////
//
// File: SteadyADR.cpp
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
// Description: SteadyADR problem solve routines
//
///////////////////////////////////////////////////////////////////////////////

#include <ADRSolverRedesign/EquationSystems/SteadyADR.h>

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/GlobalLinSysOps/LinearSolvers/LinearSolverOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/LinearSystemOp.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/PoissonSolve/PoissonSolveOp.hpp>
#include <SolverCore/PreconOps/PreconOp.hpp>

namespace Nektar
{
using namespace MultiRegions;

std::string SteadyADR::className1 =
    GetEquationSystemFactory().RegisterCreatorFunction("Poisson",
                                                       SteadyADR::create);
std::string SteadyADR::className2 =
    GetEquationSystemFactory().RegisterCreatorFunction("Helmholtz",
                                                       SteadyADR::create);
std::string SteadyADR::className3 =
    GetEquationSystemFactory().RegisterCreatorFunction("SteadyADR",
                                                       SteadyADR::create);

SteadyADR::SteadyADR(const LibUtilities::SessionReaderSharedPtr &pSession,
                     const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph), m_epsilon(1.0), m_lambda(0.0)
{
    ASSERTL0(m_projectionType == MultiRegions::eGalerkin,
             "The SteadyADR is only implemented for "
             "projectionType Galerkin");
}

/**
 * @brief Initialisation object for the SteadyADR problem.
 */
void SteadyADR::v_InitObject(bool declareExpansionLists)
{
    // Call to the initialisation object of EquationSystem
    EquationSystem::v_InitObject(declareExpansionLists);

    /// Load parameters from session
    LoadSessionParameters();

    /// Create Fields for solution m_phys, m_coeff and others
    v_InitialiseFields();

    // Set advection velocity
    SetAdvectionVel();

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Create and initialise all operators
    v_InitialiseOperators();
}

/**
 * @brief Solves SteadyADR problem.
 */
void SteadyADR::v_DoSolve()
{
    // Evaluate forcing terms. The first forcing assigns into the workspace;
    // subsequent forcings append.
    for (unsigned int i = 0; i < m_forcing.size(); ++i)
    {
        m_forcing[i]->SetAppend(i > 0);
        m_forcing[i]->Apply(m_fce, m_fce, m_time);
    }

    // Zero the forcing if there is none.
    if (m_forcing.empty())
    {
        m_math.zero(m_fce);
    }

    // Apply method for solving the SteadyADR problem
    m_linearSystemOp->Apply(m_fce, m_coeff[0]);
    m_bwdTransOp->Apply(m_coeff[0], m_phys[0]);
}

void SteadyADR::v_GenerateSummary(SummaryList &s)
{
    EquationSystem::v_GenerateSummary(s);
    AddSummaryItem(s, "Equations", "Advection-Diffusion-Reaction");

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

void SteadyADR::LoadSessionParameters()
{
    // Load lambda parameter for SteadyADR and Helmholtz problems.
    m_session->LoadParameter("lambda", m_lambda, 0.0);

    // Get diffusion parameters.
    const auto nDiffCoeff = m_coordim * (m_coordim + 1) / 2;
    m_diffCoeff.resize(nDiffCoeff);

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
    else if (m_coordim == 3)
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
 *  Create and initialise Fields.
 *
 * m_fce is FieldState::Phys workspace for the forcing term
 */
void SteadyADR::v_InitialiseFields()
{
    // Initialise solution "m_phys" via EquationSystem routine
    EquationSystem::v_InitialiseFields();

    auto bAtr_phys = MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
        m_expansionLists[0]);
    m_fce = LibUtilities::Field<double, FieldState::Phys>(
        "m_fce", bAtr_phys, m_variables[0], m_npointsZ);
}

/*
 *  @brief Sets the Advection Veclocity
 */
void SteadyADR::SetAdvectionVel()
{
    if (m_session->GetSolverInfo("EQTYPE") == "SteadyADR")
    {
        // Read advection velocity from session.
        if (m_session->DefinesFunction("AdvectionVelocity"))
        {
            auto bAtr_phys =
                MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                    m_expansionLists[0]);
            m_advectionVel = LibUtilities::Field<double, FieldState::Phys>(
                "advVel", bAtr_phys, m_coordim, m_npointsZ);

            // Reads the Session File Velocity defined as function.
            std::vector<std::string> vel = {"Vx", "Vy", "Vz"};
            vel.resize(m_coordim);
            SessionFunction advectionVelocity(m_session, m_expansionLists[0],
                                              vel, "AdvectionVelocity");
            advectionVelocity.Evaluate(m_advectionVel, m_time);
        }
        else
        {
            NEKERROR(ErrorUtil::efatal, "Function 'AdvectionVelocity' was "
                                        "not defined in session file.")
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
void SteadyADR::SetDiffusionCoeff()
{
    // Optional physical tensor, in the same D00,D01,D11,D02,D12,D22 order
    // as the constant parameters. epsilon scales every tensor component.
    if (m_session->DefinesFunction("DiffusionCoefficient"))
    {
        std::vector<std::string> names = {"D00", "D01", "D11",
                                          "D02", "D12", "D22"};
        names.resize(m_coordim * (m_coordim + 1) / 2);
        auto bAtr_phys =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansionLists[0]);
        m_varDiffCoeff = LibUtilities::Field<double, FieldState::Phys>(
            "DiffusionCoefficient", bAtr_phys, names.size(), m_npointsZ);
        m_varDiffCoeffFunction = std::make_shared<SessionFunction>(
            m_session, m_expansionLists[0], names, "DiffusionCoefficient");
        m_varDiffCoeffFunction->SetScale(m_epsilon);
        m_varDiffCoeffFunction->Evaluate(m_varDiffCoeff, m_time);
    }
    else
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
        else if (m_coordim == 3)
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
void SteadyADR::v_InitialiseOperators()
{
    m_bwdTransOp =
        BwdTransOp<double>::Create(m_expansionLists[0], m_variables[0]);

    // Create LinearADRSolve, preconditioner, and linear system operators
    m_preconOp       = PreconOp<double>::Create(m_expansionLists[0],
                                                m_session->GetVariables());
    m_linearSolverOp = LinearSolverOp<double>::Create(
        m_expansionLists[0], m_session->GetVariables());

    if (m_session->GetSolverInfo("EQTYPE") == "SteadyADR")
    {
        auto linearADRSolveOp = LinearADRSolveOp<double>::Create(
            m_expansionLists[0], m_session->GetVariables());
        linearADRSolveOp->SetLinearSolver(m_linearSolverOp);
        linearADRSolveOp->SetLambda(m_lambda);
        if (m_varDiffCoeffFunction)
        {
            linearADRSolveOp->SetVarDiffCoeff(m_varDiffCoeff);
        }
        else
        {
            linearADRSolveOp->SetDiffCoeff(m_diffCoeff);
        }
        linearADRSolveOp->SetAdvVel(m_advectionVel);
        linearADRSolveOp->SetPrecon(m_preconOp);
        linearADRSolveOp->UpdatePrecon();
        m_linearSystemOp = linearADRSolveOp;
    }
    else if (m_session->GetSolverInfo("EQTYPE") == "Helmholtz")
    {
        auto helmSolveOp = HelmSolveOp<double>::Create(
            m_expansionLists[0], m_session->GetVariables());
        helmSolveOp->SetLambda(m_lambda);
        helmSolveOp->SetLinearSolver(m_linearSolverOp);
        if (m_varDiffCoeffFunction)
        {
            helmSolveOp->SetVarDiffCoeff(m_varDiffCoeff);
        }
        else
        {
            helmSolveOp->SetDiffCoeff(m_diffCoeff);
        }
        helmSolveOp->SetPrecon(m_preconOp);
        helmSolveOp->UpdatePrecon();
        m_linearSystemOp = helmSolveOp;
    }
    else if (m_session->GetSolverInfo("EQTYPE") == "Poisson")
    {
        auto poissonSolveOp = PoissonSolveOp<double>::Create(
            m_expansionLists[0], m_session->GetVariables());
        poissonSolveOp->SetLinearSolver(m_linearSolverOp);
        if (m_varDiffCoeffFunction)
        {
            poissonSolveOp->SetVarDiffCoeff(m_varDiffCoeff);
        }
        else
        {
            poissonSolveOp->SetDiffCoeff(m_diffCoeff);
        }
        poissonSolveOp->SetPrecon(m_preconOp);
        poissonSolveOp->UpdatePrecon();
        m_linearSystemOp = poissonSolveOp;
    }

    // Load forcing terms, if defined in the session file.
    m_forcing = Forcing::Load(m_session, m_expansionLists[0], m_variables[0]);
}

} // namespace Nektar
