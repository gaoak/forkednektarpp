///////////////////////////////////////////////////////////////////////////////
//
// File: AdjointTemperature.cpp
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
// Description: Steady temperature adjoint equation
//
///////////////////////////////////////////////////////////////////////////////

#include <ADRSolver/EquationSystems/AdjointTemperature.h>

#include <LibUtilities/BasicUtils/Vmath.hpp>

#include <ADRSolver/Adjoint/AdjointBoundaryCheck.h>

namespace Nektar
{

std::string AdjointTemperature::className =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "AdjointTemperature", AdjointTemperature::create,
        "Steady temperature adjoint equation");

AdjointTemperature::AdjointTemperature(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : SteadyAdvectionDiffusion(pSession, pGraph), m_rho(1.0), m_cp(1.0),
      m_conductivity(1.0)
{
}

void AdjointTemperature::v_InitObject(bool DeclareFields)
{
    // Xxt (sparse Cholesky) silently returns wrong answers for this operator
    // when Neumann/Robin conditions meet a base flow; see the helper.
    AdjointGuardXxtGlobalSysSoln(m_session, m_graph);

    // Deliberately skip SteadyAdvectionDiffusion::v_InitObject, which would
    // require a 'BaseFlow' function evaluated through the SessionFunction
    // machinery.  The adjoint base flow is handled by AdjointBaseFlow.
    EquationSystem::v_InitObject(DeclareFields);

    m_session->LoadParameter("Rho", m_rho, 1.0);
    m_session->LoadParameter("Cp", m_cp, 1.0);
    m_session->LoadParameter("Conductivity", m_conductivity, 1.0);

    if (m_session->DefinesParameter("epsilon"))
    {
        m_session->LoadParameter("epsilon", m_epsilon, 1.0);
    }
    else
    {
        ASSERTL0(m_rho > 0.0 && m_cp > 0.0 && m_conductivity > 0.0,
                 "Rho, Cp and Conductivity must be positive.");
        m_epsilon = m_conductivity / (m_rho * m_cp);
    }
    ASSERTL0(m_epsilon > 0.0, "Adjoint diffusion coefficient must be > 0.");

    m_lambda = 0.0;
    m_session->LoadParameter("lambda", m_lambda, 0.0);

    m_baseFlow.Init(m_session, m_fields);

    NekDouble baseFlowTime = m_baseFlow.GetTimeStart();
    m_session->LoadParameter("BaseFlowTime", baseFlowTime, baseFlowTime);
    m_baseFlow.Update(baseFlowTime);

    // Shallow copy: the arrays are owned by m_baseFlow and remain valid.
    m_velocity = m_baseFlow.GetVelocityMinus();

    if (m_session->GetComm()->GetRank() == 0)
    {
        std::cout << "Steady temperature adjoint: nu = " << m_epsilon
                  << ", base flow: " << m_baseFlow.GetSummary() << std::endl;
    }
}

void AdjointTemperature::v_DoInitialise(bool dumpInitialConditions)
{
    // Volumetric source s = w_T/(rho cp).  'AdjointSource' is the documented
    // name; fall back to 'Forcing' for compatibility with the base class.
    m_sourceFunction = "AdjointSource";
    if (!m_session->DefinesFunction("AdjointSource") &&
        m_session->DefinesFunction("Forcing"))
    {
        m_sourceFunction = "Forcing";
    }

    ASSERTL0(m_session->DefinesFunction(m_sourceFunction),
             "Temperature adjoint requires a session function "
             "'AdjointSource' (or 'Forcing') giving s = w_T/(rho cp) for "
             "the variable psi.");

    GetFunction(m_sourceFunction)
        ->Evaluate(m_session->GetVariables(), m_fields);

    // Cache the source.  v_DoSolve overwrites the field coefficients, so
    // taking the source from m_fields[i]->GetPhys() there would turn the
    // previous solution into the source if the solver were invoked twice.
    m_source = Array<OneD, Array<OneD, NekDouble>>(m_fields.size());
    for (int i = 0; i < m_fields.size(); ++i)
    {
        const int nPts = m_fields[i]->GetNpoints();
        m_source[i]    = Array<OneD, NekDouble>(nPts);
        Vmath::Vcopy(nPts, m_fields[i]->GetPhys(), 1, m_source[i], 1);
    }

    EquationSystem::v_DoInitialise(dumpInitialConditions);
}

void AdjointTemperature::v_DoSolve()
{
    StdRegions::ConstFactorMap factors;
    StdRegions::VarCoeffMap varcoeffs;

    // Nektar's LinearAdvectionDiffusionReactionSolve assembles the operator
    //   c.grad(u) - laplacian(u) - lambda u = -f,
    // i.e. it negates the physical-space forcing internally (see the comment
    // in ContField::v_LinearAdvectionDiffusionReactionSolve).  We want
    //   -u.grad(psi) - epsilon laplacian(psi) + lambda psi = s,
    // so pass c = -u/epsilon, lambda_factor = -lambda/epsilon and f = -s/eps.
    factors[StdRegions::eFactorLambda] = -m_lambda / m_epsilon;

    // VarCoeff velocities c/epsilon with c = -u.  Do not modify m_velocity
    // in place: it aliases the base-flow storage and this method may be
    // called more than once.
    StdRegions::VarCoeffType varcoefftypes[] = {StdRegions::eVarCoeffVelX,
                                                StdRegions::eVarCoeffVelY,
                                                StdRegions::eVarCoeffVelZ};
    Array<OneD, Array<OneD, NekDouble>> scaledVel(m_spacedim);
    for (int i = 0; i < m_spacedim; ++i)
    {
        scaledVel[i] = Array<OneD, NekDouble>(m_fields[0]->GetNpoints());
        Vmath::Smul(m_velocity[i].size(), 1.0 / m_epsilon, m_velocity[i], 1,
                    scaledVel[i], 1);
        varcoeffs[varcoefftypes[i]] = scaledVel[i];
    }

    for (int i = 0; i < m_fields.size(); ++i)
    {
        const int nPts    = m_fields[i]->GetNpoints();
        const int nCoeffs = m_fields[i]->GetNcoeffs();

        // Negated source: see the sign discussion above.
        ASSERTL1(static_cast<int>(m_source.size()) == m_fields.size(),
                 "Source workspace was not initialised.");
        ASSERTL1(m_source[i].size() == nPts,
                 "Source workspace has the wrong size.");
        Array<OneD, NekDouble> rhs(nPts);
        Vmath::Smul(nPts, -1.0 / m_epsilon, m_source[i], 1, rhs, 1);

        Vmath::Zero(nCoeffs, m_fields[i]->UpdateCoeffs(), 1);
        m_fields[i]->LinearAdvectionDiffusionReactionSolve(
            rhs, m_fields[i]->UpdateCoeffs(), factors, varcoeffs);
        m_fields[i]->SetPhysState(false);
    }
}

void AdjointTemperature::v_GenerateSummary(SolverUtils::SummaryList &s)
{
    EquationSystem::v_GenerateSummary(s);
    SolverUtils::AddSummaryItem(s, "Adjoint equation", "steady temperature");
    SolverUtils::AddSummaryItem(s, "Adjoint nu", m_epsilon);
    SolverUtils::AddSummaryItem(s, "Adjoint base flow",
                                m_baseFlow.GetSummary());
}

} // namespace Nektar
