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

#include <LibUtilities/BasicUtils/Timer.h>

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
    : EquationSystem(pSession, pGraph)
{
}

/**
 * @brief Initialisation object for the SteadyADR problem.
 */
void SteadyADR::v_InitObject(bool DeclareFields)
{
    EquationSystem::v_InitObject(DeclareFields);

    // Load output/verbose parameters
    m_session->LoadParameter("IO_InfoSteps", m_infosteps, 0);

    // Get variable strings and number of variables
    m_variables  = m_session->GetVariables();
    m_nVariables = m_variables.size();

    // Set up diffusion Coeff.
    SetDiffusionCoeff();

    // Set Advection Vecloity
    if (m_session->GetSolverInfo("EQTYPE") == "SteadyADR")
    {
        SetAdvectionVel();
    }

    // Create and initialise all operators
    InitialiseOperators();

    // Create and initialise all fields
    InitialiseFields();
}

/**
 * @brief Solves SteadyADR problem.
 */
void SteadyADR::v_DoSolve()
{
    // Evaluate the forcing
    if (m_session->DefinesFunction("BodyForce"))
    {
        // Evaluate forcing operator
        m_forcingOp->Apply(m_wsp_fce, m_wsp_fce);
    }

    // Apply method for solving the SteadyADR problem
    m_linearSystemOp->Apply(m_wsp_fce, m_wsp_coeff);

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

void SteadyADR::v_GenerateSummary(SummaryList &s)
{
    SessionSummary(s);

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
void SteadyADR::InitialiseOperators()
{
    // Initialise Math
    std::string execName = Operator<double>::GetOpExecSpace(m_session);
    m_math               = Math(execName);

    // Create LinearADRSolve, preconditioner, and linear system operators
    m_preconOp =
        PreconOp<double>::Create(m_fields[0], m_session->GetVariables());
    m_linearSolverOp =
        LinearSolverOp<double>::Create(m_fields[0], m_session->GetVariables());

    if (m_session->GetSolverInfo("EQTYPE") == "SteadyADR")
    {
        unsigned int coordDim = m_fields[0]->GetCoordim(0);
        auto advelblockAttr =
            GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
        auto vel =
            Field<double, FieldState::Phys>("vel", advelblockAttr, coordDim, 1);
        vel.template CopyArray<NektarSpaces::HostSpace>(m_AdVel);
        double lambda;
        m_session->LoadParameter("lambda", lambda, 0.0);
        auto linearADRSolveOp = LinearADRSolveOp<double>::Create(
            m_fields[0], m_session->GetVariables());
        linearADRSolveOp->SetLinearSolver(m_linearSolverOp);
        linearADRSolveOp->SetLambda(lambda);
        linearADRSolveOp->SetDiffCoeff(m_diffCoeff);
        linearADRSolveOp->SetAdvVel(vel);
        linearADRSolveOp->SetPrecon(m_preconOp);
        linearADRSolveOp->UpdatePrecon();
        m_linearSystemOp = linearADRSolveOp;
    }
    else if (m_session->GetSolverInfo("EQTYPE") == "Helmholtz")
    {
        double lambda;
        m_session->LoadParameter("lambda", lambda, 0.0);
        auto helmSolveOp =
            HelmSolveOp<double>::Create(m_fields[0], m_session->GetVariables());
        helmSolveOp->SetLambda(lambda);
        helmSolveOp->SetLinearSolver(m_linearSolverOp);
        helmSolveOp->SetDiffCoeff(m_diffCoeff);
        helmSolveOp->SetPrecon(m_preconOp);
        helmSolveOp->UpdatePrecon();
        m_linearSystemOp = helmSolveOp;
    }
    else if (m_session->GetSolverInfo("EQTYPE") == "Poisson")
    {
        auto poissonSolveOp = PoissonSolveOp<double>::Create(
            m_fields[0], m_session->GetVariables());
        poissonSolveOp->SetLinearSolver(m_linearSolverOp);
        poissonSolveOp->SetDiffCoeff(m_diffCoeff);
        poissonSolveOp->SetPrecon(m_preconOp);
        poissonSolveOp->UpdatePrecon();
        m_linearSystemOp = poissonSolveOp;
    }

    // Check if forcing is defined
    if (m_session->DefinesFunction("BodyForce"))
    {
        // Create operator
        m_forcingOp = ExpressionOp<double>::Create(
            m_fields[0], m_session->GetVariables(), "Serial", "Generic");

        // Read initial conditions and configure operator
        std::vector<LibUtilities::EquationSharedPtr> forcingEquations;
        for (unsigned int i = 0; i < m_nVariables; ++i)
        {
            forcingEquations.push_back(m_session->GetFunction("BodyForce", i));
        }
        m_forcingOp->SetExpressions(forcingEquations);
    }
}

/*
 *  Create and initialise Fields.
 *
 * m_wsp_fce is FieldState::Phys workspace initialised to zero for the forcing
 * term m_wsp_coeff is a FieldState::Coeff workspace initialised to zero for the
 * solution
 */
void SteadyADR::InitialiseFields()
{
    // Get block Attributes.
    auto block_attr_phys =
        GetBlockAttributes<double, FieldState::Phys>(m_fields[0]);
    auto block_attr_coeff =
        GetBlockAttributes<double, FieldState::Coeff>(m_fields[0]);

    // Create fields.
    unsigned int nhomo = m_npointsZ; // Note read in EquationSystem.cpp

    m_wsp_fce = Field<double, FieldState::Phys>("m_wsp_fce", block_attr_phys,
                                                m_nVariables, nhomo);

    m_wsp_coeff = Field<double, FieldState::Coeff>(
        "m_wsp_coeff", block_attr_coeff, m_nVariables, nhomo);

    // Initialise fields
    m_math.zero(m_wsp_fce);
    m_math.zero(m_wsp_coeff);
}

/*
 *  @brief Sets the Advection Veclocity
 */
void SteadyADR::SetAdvectionVel()
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

} // namespace Nektar
