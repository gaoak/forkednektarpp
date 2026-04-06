///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyADR.cpp
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
// Description: UnsteadyADR problem solve routines
//
///////////////////////////////////////////////////////////////////////////////

#include "Operators/TimeOps/TimeOp.hpp"

#include <ADRSolverRedesign/EquationSystems/UnsteadyADR.h>
#include <LibUtilities/BasicUtils/Timer.h>

namespace Nektar
{
using namespace Operators;

std::string UnsteadyADR::className =
    GetEquationSystemFactory().RegisterCreatorFunction("UnsteadyADRRedesign",
                                                       UnsteadyADR::create);

UnsteadyADR::UnsteadyADR(const LibUtilities::SessionReaderSharedPtr &pSession,
                         const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph)
{
}

/**
 * @brief Initialisation object for the UnsteadyADR problem.
 */
void UnsteadyADR::v_InitObject(bool DeclareFields)
{
    EquationSystem::v_InitObject(DeclareFields);

    // Initialise boundary conditions
    SetBoundaryConditions(m_time);

    // Get variable strings and number of variables
    m_variables  = m_session->GetVariables();
    m_nVariables = m_variables.size();

    // Initialise Time-stepping operator
    m_timeOp = TimeOp<double>::Create(m_fields[0], m_session->GetVariables());
    m_timeOp->DefineImplicit(&UnsteadyADR::DoLinearADR, this);
    m_timeOp->DefineExplicitRhs(&UnsteadyADR::DoReaction, this);
    m_timeOp->DefineProjection(&UnsteadyADR::DoProjection, this);

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Set Advection Vecloity
    SetAdvectionVel();

    // Create and initialise all operators
    InitialiseOperators();

    // Create and initialise all fields
    InitialiseFields();
}

/**
 * @brief Solves UnsteadyADR problem.
 */
void UnsteadyADR::v_DoSolve()
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

    // TODO : Remove the below code, when updated with Redesign solverUtils
    //  ----------------------------------------------------------------------
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
    //--------------------------------------------------------------------------
}

void UnsteadyADR::v_GenerateSummary(SummaryList &s)
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

void UnsteadyADR::DoLinearADR(
    Field<double, FieldState::Phys> &in,
    [[maybe_unused]] Field<double, FieldState::Phys> &out,
    [[maybe_unused]] const double &time, const double &dt_inv_gamma)
{
    // Update \lambda = \gamma / \Delta t
    m_lambda = 1.0 / dt_inv_gamma;

    // Update LinearADRSolve
    m_linearADRSolveOp->SetLambda(m_lambda);
    if (m_preconOp.find(dt_inv_gamma) == m_preconOp.end())
    {
        // Configure and cache preconditioner
        m_preconOp.insert(
            {dt_inv_gamma,
             PreconOp<double>::Create(m_fields[0], m_session->GetVariables())});
        m_linearADRSolveOp->SetPrecon(m_preconOp[dt_inv_gamma]);
        m_linearADRSolveOp->UpdatePrecon();
    }
    else
    {
        // Re-use preconditioner
        m_linearADRSolveOp->SetPrecon(m_preconOp[dt_inv_gamma]);
    }

    // Multiply by negative lambda
    m_math.mul(-m_lambda, in, out);

    // Solve LinearADR problem
    m_linearADRSolveOp->Apply(out, m_wsp_coeff);

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
void UnsteadyADR::DoReaction(Field<double, FieldState::Phys> &in,
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
void UnsteadyADR::DoProjection(Field<double, FieldState::Phys> &in,
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
void UnsteadyADR::SetDiffusionCoeff()
{
    // Get the scalar part
    m_session->LoadParameter("epsilon", m_epsilon, 1.0);

    // Set-up anisotropic diffusion coefficient
    // Default value for D_ij = 1, if i = j
    // Default value for D_ij = 0, if i != j
    const auto coordDim      = m_fields[0]->GetCoordim(0);
    const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
    m_diffCoeff.resize(diffCoeffSize);

    if (coordDim == 1)
    {
        m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
        // Multiply by Scalar Part
        m_diffCoeff[0] *= m_epsilon;
    }
    else if (coordDim == 2)
    {
        m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
        m_session->LoadParameter("D01", m_diffCoeff[1], 0.0);
        m_session->LoadParameter("D11", m_diffCoeff[2], 1.0);
        // Multiply by Scalar Part
        m_diffCoeff[0] *= m_epsilon;
        m_diffCoeff[1] *= m_epsilon;
        m_diffCoeff[2] *= m_epsilon;
    }
    else
    {
        m_session->LoadParameter("D00", m_diffCoeff[0], 1.0);
        m_session->LoadParameter("D01", m_diffCoeff[1], 0.0);
        m_session->LoadParameter("D11", m_diffCoeff[2], 1.0);
        m_session->LoadParameter("D02", m_diffCoeff[3], 0.0);
        m_session->LoadParameter("D12", m_diffCoeff[4], 0.0);
        m_session->LoadParameter("D22", m_diffCoeff[5], 1.0);
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
void UnsteadyADR::InitialiseOperators()
{
    // Initialise Math
    std::string execName = Operator<double>::GetOpExecSpace(m_session);
    m_math               = Math(execName);

    // Create LinearADRSolve, preconditioner, and linear system operators
    m_linearSolverOp =
        LinearSolverOp<double>::Create(m_fields[0], m_session->GetVariables());
    m_bwdTransOp =
        BwdTransOp<double>::Create(m_fields[0], m_session->GetVariables());
    m_linearADRSolveOp = LinearADRSolveOp<double>::Create(
        m_fields[0], m_session->GetVariables());

    // Configure UnsteadyADRSolve
    unsigned int coordDim = m_fields[0]->GetCoordim(0);
    auto advelblockAttr =
        GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
    auto vel =
        Field<double, FieldState::Phys>("vel", advelblockAttr, coordDim, 1);
    vel.template CopyArray<NektarSpaces::HostSpace>(m_AdVel);
    m_linearADRSolveOp->SetLinearSolver(m_linearSolverOp);
    m_linearADRSolveOp->SetDiffCoeff(m_diffCoeff);
    m_linearADRSolveOp->SetAdvVel(vel);

    // Check if forcing is defined
    if (m_session->DefinesFunction("BodyForce"))
    {
        // Create operator
        m_forcingOp = ExpressionOp<double>::Create(m_fields[0],
                                                   m_session->GetVariables());

        // Read initial conditions and configure operator
        std::vector<LibUtilities::EquationSharedPtr> forcingEquations;
        for (unsigned int i = 0; i < m_nVariables; ++i)
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
void UnsteadyADR::InitialiseFields()
{
    // Get block Attributes.
    auto block_attr_phys =
        GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
    auto block_attr_coeff =
        GetBlockAttributes<double, FieldState::Coeff>(m_fields[0]);

    // Create fields.
    unsigned int nhomo = m_npointsZ; // Note read in EquationSystem.cpp
    m_in        = Field<double, FieldState::Phys>("solution", block_attr_phys,
                                           m_nVariables, nhomo);
    m_wsp_coeff = Field<double, FieldState::Coeff>(
        "m_wsp_coeff", block_attr_coeff, m_nVariables, nhomo);
}

/*
 *  @brief Sets the Advection Veclocity
 */
void UnsteadyADR::SetAdvectionVel()
{
    unsigned int nhomo    = m_npointsZ; // Note read in EquationSystem.cpp
    unsigned int coordDim = m_fields[0]->GetCoordim(0);
    double Vx;

    // Set advection velocity
    size_t nphys = m_fields[0]->GetTotPoints() / nhomo;
    std::cout << std::endl;
    std::cout << "nphys = " << nphys << " , coordDim = " << coordDim
              << std::endl;

    // Get the advection velocity
    m_AdVel = Array<OneD, double>(nphys * coordDim);

    if (m_session->DefinesFunction("BaseFlow"))
    {
        // Reads the Session File Vecoity defined as function
        std::vector<std::string> vel;
        vel.push_back("Vx");
        vel.push_back("Vy");
        vel.push_back("Vz");

        // Resize the advection velocities vector to dimension of the problem
        vel.resize(coordDim);

        // Get Advection Velocity from Session file
        // A Temp variable, tmp of type Array<OneD, Array<OneD, NekDouble>>
        // tmp reads from Session, with Legacy routnie
        Array<OneD, Array<OneD, NekDouble>> tmp =
            Array<OneD, Array<OneD, NekDouble>>(coordDim);
        GetFunction("BaseFlow")->Evaluate(vel, tmp);

        // Rewrite into m_AdVel of type Array<OneD, double>
        // Since, the Redeisgn expects adevection velocity of that type
        size_t count = 0;
        for (unsigned int i = 0; i < coordDim; i++)
        {
            for (size_t j = 0; j < nphys; j++)
            {
                m_AdVel[count] = tmp[i][j];
                count += 1;
            }
        }
    }
    else
    {
        // Reads the Session File velocity defined as paramter for a constant
        // value

        // For the first dimension
        m_session->LoadParameter("Vx", Vx, 0.0);
        for (size_t i = 0; i < nphys; i++)
        {
            m_AdVel[i] = Vx;
        }
        if (coordDim >= 2)
        {
            // For the second dimension
            m_session->LoadParameter("Vy", Vx, 0.0);
            for (size_t i = nphys; i < nphys * 2; i++)
            {
                m_AdVel[i] = Vx;
            }
        }
        if (coordDim == 3)
        {
            // For the third dimension
            m_session->LoadParameter("Vz", Vx, 0.0);
            for (size_t i = 2 * nphys; i < nphys * 3; i++)
            {
                m_AdVel[i] = Vx;
            }
        }
    }
}

void UnsteadyADR::SetInitialConditionsField(
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
        m_math.zero(m_wsp_coeff);
        initialOp->Apply(field, field);

        // Note we could use the cheaper operators: AvgAssemble or GlobalToLocal
        m_fwdTransOp->Apply(field, m_wsp_coeff);
        m_bwdTransOp->Apply(m_wsp_coeff, field);

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
