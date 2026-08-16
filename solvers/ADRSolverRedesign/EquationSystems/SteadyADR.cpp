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

#include <Operators/ElmtOps/Expression/ExpressionOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/PoissonSolve/PoissonSolveOp.hpp>

#include <ADRSolverRedesign/EquationSystems/SteadyADR.h>

namespace Nektar
{
using namespace Operators;

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
    InitialiseParameters();

    /// Create Field for solution m_fields and others
    InitialiseFields();

    // Set advection velocity
    SetAdvectionVel();

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Create and initialise all operators
    InitialiseOperators();
}

/**
 * @brief Solves SteadyADR problem.
 */
void SteadyADR::v_DoSolve()
{
    // Evaluate forcing terms. The first forcing assigns into the workspace;
    // subsequent forcings append.
    m_math.zero(m_wsp_fce);
    for (unsigned int i = 0; i < m_forcing.size(); ++i)
    {
        m_forcing[i]->SetAppend(i > 0);
        m_forcing[i]->Apply(m_wsp_fce, m_wsp_fce, m_time);
    }

    // Apply method for solving the SteadyADR problem
    m_linearSystemOp->Apply(m_wsp_fce, m_fields_coeff);
    m_bwdTransOp->Apply(m_fields_coeff, m_fields);
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
    else
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

/*
 *  Create and initialise all operators for this solver
 */
void SteadyADR::v_InitialiseOperators()
{
    EquationSystem::v_InitialiseOperators();

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
        linearADRSolveOp->SetDiffCoeff(m_diffCoeff);
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
        helmSolveOp->SetDiffCoeff(m_diffCoeff);
        helmSolveOp->SetPrecon(m_preconOp);
        helmSolveOp->UpdatePrecon();
        m_linearSystemOp = helmSolveOp;
    }
    else if (m_session->GetSolverInfo("EQTYPE") == "Poisson")
    {
        auto poissonSolveOp = PoissonSolveOp<double>::Create(
            m_expansionLists[0], m_session->GetVariables());
        poissonSolveOp->SetLinearSolver(m_linearSolverOp);
        poissonSolveOp->SetDiffCoeff(m_diffCoeff);
        poissonSolveOp->SetPrecon(m_preconOp);
        poissonSolveOp->UpdatePrecon();
        m_linearSystemOp = poissonSolveOp;
    }

    // Load forcing terms, if defined in the session file.
    m_forcing = Forcing::Load(m_session, m_expansionLists[0], m_variables);
}

/*
 *  Create and initialise Fields.
 *
 * m_wsp_fce is FieldState::Phys workspace initialised to zero for the forcing
 * term m_fields_coeff is a FieldState::Coeff workspace initialised to zero for
 * the solution
 */
void SteadyADR::v_InitialiseFields()
{
    // Initialise solution "m_fields" via EquationSystem routine
    EquationSystem::v_InitialiseFields();

    // Get block Attributes.
    auto bAtr_phys = MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
        m_expansionLists[0]);
    auto bAtr_coeff =
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            m_expansionLists[0]);

    // Create fields.
    unsigned int nhomo = m_npointsZ; // Note read in EquationSystem.cpp

    m_wsp_fce = LibUtilities::Field<double, FieldState::Phys>(
        "m_wsp_fce", bAtr_phys, m_nVariables, nhomo);

    if (m_session->GetSolverInfo("EQTYPE") == "SteadyADR")
    {
        unsigned int coordDim = m_expansionLists[0]->GetCoordim(0);
        m_advectionVel        = LibUtilities::Field<double, FieldState::Phys>(
            "advVel", bAtr_phys, coordDim, nhomo);
    }

    // Initialise fields
    m_math.zero(m_wsp_fce);
}

/*
 *  @brief Sets the Advection Veclocity
 */
void SteadyADR::SetAdvectionVel()
{
    if (m_session->GetSolverInfo("EQTYPE") == "SteadyADR")
    {
        // Read advection velocity from session
        if (m_session->DefinesFunction("AdvectionVelocity"))
        {
            // Reads the Session File Velocity defined as function
            std::vector<std::string> vel;
            vel.push_back("Vx");
            vel.push_back("Vy");
            vel.push_back("Vz");
            vel.resize(m_coordim);

            // Initialise operators
            auto expressionOp =
                ExpressionOp<double>::Create(m_expansionLists[0], vel);

            // Read advection velocity expressions and configure operator
            std::vector<LibUtilities::EquationSharedPtr> advectionVelocities;
            for (unsigned int i = 0; i < m_coordim; ++i)
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

void SteadyADR::InitialiseParameters()
{
    // Load lambda parameter for SteadyADR and Helmholtz problems
    m_session->LoadParameter("lambda", m_lambda, 0.0);

    // Get diffusion parameters
    const auto diffCoeffSize = m_coordim * (m_coordim + 1) / 2;
    m_diffCoeff.resize(diffCoeffSize);

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

} // namespace Nektar
