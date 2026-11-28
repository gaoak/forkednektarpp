///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyAdjointTemperature.cpp
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
// Description: Unsteady temperature adjoint advanced in pseudo-time
//
///////////////////////////////////////////////////////////////////////////////

#include <ADRSolver/EquationSystems/UnsteadyAdjointTemperature.h>

#include <LibUtilities/BasicUtils/Vmath.hpp>

#include <ADRSolver/Adjoint/AdjointBoundaryCheck.h>

namespace Nektar
{

std::string UnsteadyAdjointTemperature::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "UnsteadyAdjointTemperature", UnsteadyAdjointTemperature::create,
        "Unsteady temperature adjoint equation (pseudo-time)");

UnsteadyAdjointTemperature::UnsteadyAdjointTemperature(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), // virtual base
      UnsteadyAdvectionDiffusion(pSession, pGraph), m_rho(1.0), m_cp(1.0),
      m_conductivity(1.0), m_nu(1.0), m_t0(0.0), m_t1(0.0)
{
}

void UnsteadyAdjointTemperature::v_InitObject(bool DeclareFields)
{
    // Xxt (sparse Cholesky) silently returns wrong answers for this operator
    // when Neumann/Robin conditions meet a base flow; see the helper.
    AdjointGuardXxtGlobalSysSoln(m_session, m_graph);

    // The base class unconditionally evaluates AdvectionVelocity to size the
    // velocity arrays.  Require a zero placeholder and explain why.
    ASSERTL0(m_session->DefinesFunction("AdvectionVelocity"),
             "UnsteadyAdjointTemperature requires a placeholder session "
             "function 'AdvectionVelocity' with zero components "
             "(Vx=0, Vy=0[, Vz=0]); it is overwritten by the base flow.");

    UnsteadyAdvectionDiffusion::v_InitObject(DeclareFields);

    ASSERTL0(!m_subSteppingScheme,
             "Sub-stepping time integration is not supported by "
             "UnsteadyAdjointTemperature.");
    ASSERTL0(m_steadyStateTol == 0.0,
             "Steady-state stopping is not supported by "
             "UnsteadyAdjointTemperature.");

    m_session->LoadParameter("Rho", m_rho, 1.0);
    m_session->LoadParameter("Cp", m_cp, 1.0);
    m_session->LoadParameter("Conductivity", m_conductivity, 1.0);

    if (m_session->DefinesParameter("epsilon"))
    {
        m_session->LoadParameter("epsilon", m_nu, 1.0);
    }
    else
    {
        ASSERTL0(m_rho > 0.0 && m_cp > 0.0 && m_conductivity > 0.0,
                 "Rho, Cp and Conductivity must be positive.");
        m_nu = m_conductivity / (m_rho * m_cp);
    }
    ASSERTL0(m_nu > 0.0, "Adjoint diffusion coefficient must be > 0.");
    SetEpsilon(m_nu);

    // Physical observation window.
    m_session->LoadParameter("AdjointStartTime", m_t0, 0.0);
    if (m_session->DefinesParameter("AdjointEndTime"))
    {
        m_session->LoadParameter("AdjointEndTime", m_t1, 0.0);
    }
    else
    {
        ASSERTL0(m_fintime > 0.0,
                 "UnsteadyAdjointTemperature requires 'AdjointEndTime' "
                 "(or Nektar's 'FinTime') to define the end of the "
                 "observation window.");
        m_t1 = m_fintime;
    }
    ASSERTL0(m_t1 > m_t0,
             "AdjointEndTime must be greater than AdjointStartTime.");

    // March in pseudo-time from 0 to t1-t0.
    m_time   = 0.0;
    m_fintime = m_t1 - m_t0;
    m_steps  = 0;

    m_fieldMetaDataMap["PhysicalTime"] = std::to_string(m_t1);
    m_fieldMetaDataMap["AdjointTimeShift"] = std::to_string(m_t1);

    m_baseFlow.Init(m_session, m_fields);
    m_baseFlow.Update(m_t1);
    CopyVelocityMinusToFields();

    // Rebind the time-integration operators to the pseudo-time aware ones.
    // The base class definitions would evaluate boundary conditions and the
    // source at tau instead of the physical time t1-tau.
    m_ode.DefineOdeRhs(&UnsteadyAdjointTemperature::DoOdeRhs, this);
    m_ode.DefineProjection(&UnsteadyAdjointTemperature::DoOdeProjection, this);
    m_ode.DefineImplicitSolve(&UnsteadyAdjointTemperature::DoImplicitSolve,
                              this);

    ASSERTL0(m_session->DefinesFunction("AdjointSource"),
             "UnsteadyAdjointTemperature requires a session function "
             "'AdjointSource' giving w_T/(rho cp) for psi.  It is evaluated "
             "by the solver at the physical time t = t1 - tau; do not add a "
             "FORCING block with the same function (it would be counted "
             "twice).");

    m_source = Array<OneD, NekDouble>(m_fields[0]->GetNpoints(), 0.0);

    if (m_session->GetComm()->GetRank() == 0)
    {
        std::cout << "Pseudo-time temperature adjoint: nu = " << m_nu
                  << ", physical window [" << m_t0 << ", " << m_t1
                  << "], pseudo-time [0, " << m_fintime << "]" << std::endl;
        std::cout << "Base flow: " << m_baseFlow.GetSummary() << std::endl;
    }
}

void UnsteadyAdjointTemperature::v_DoInitialise(bool dumpInitialConditions)
{
    // Handles restart (InitialConditions file metadata 'Time' = tau), the
    // terminal condition at tau = t1-t0 and the initial checkpoint.
    UnsteadyAdvectionDiffusion::v_DoInitialise(dumpInitialConditions);

    // The base initialisation evaluated BCs at tau; re-impose them at the
    // physical time.
    SetBoundaryConditions(m_t1 - m_time);

    m_fieldMetaDataMap["PhysicalTime"] = std::to_string(m_t1 - m_time);

    m_baseFlow.Update(m_t1 - m_time);
    CopyVelocityMinusToFields();
}

void UnsteadyAdjointTemperature::CopyVelocityMinusToFields()
{
    const Array<OneD, Array<OneD, NekDouble>> &vel =
        m_baseFlow.GetVelocityMinus();
    const int nq = m_fields[0]->GetNpoints();

    ASSERTL1(static_cast<int>(m_velocity.size()) == m_spacedim,
             "Velocity storage has the wrong number of components.");

    for (int d = 0; d < m_spacedim; ++d)
    {
        ASSERTL1(m_velocity[d].size() == nq,
                 "Velocity storage has the wrong number of points.");
        Vmath::Vcopy(nq, vel[d], 1, m_velocity[d], 1);
    }
}

void UnsteadyAdjointTemperature::DoOdeRhs(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time)
{
    const NekDouble t = m_t1 - time; // physical time

    m_baseFlow.Update(t);
    CopyVelocityMinusToFields();

    // Advection + diffusion.  Note: the session's FORCING blocks (if any)
    // are applied here at the physical time by the base class.
    UnsteadyAdvectionDiffusion::DoOdeRhs(inarray, outarray, t);

    // Volumetric source w_T/(rho cp), evaluated at the physical time.  We do
    // this through a fresh (uncached) SessionFunction because the cached
    // evaluation in ForcingBody assumes monotonically increasing time; the
    // physical time decreases in the pseudo-time march.
    SolverUtils::SessionFunctionSharedPtr source = GetFunction("AdjointSource");
    const int nq = m_fields[0]->GetNpoints();
    for (int i = 0; i < m_fields.size(); ++i)
    {
        const std::string var = m_session->GetVariable(i);
        if (!m_session->DefinesFunction("AdjointSource", var))
        {
            continue;
        }
        source->Evaluate(var, m_source, t);
        ASSERTL1(static_cast<int>(m_source.size()) == nq,
                 "Source workspace has the wrong size.");
        Vmath::Vadd(nq, outarray[i], 1, m_source, 1, outarray[i], 1);
    }
}

void UnsteadyAdjointTemperature::DoOdeProjection(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time)
{
    UnsteadyAdvectionDiffusion::DoOdeProjection(inarray, outarray,
                                                m_t1 - time);
}

void UnsteadyAdjointTemperature::DoImplicitSolve(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, NekDouble time,
    NekDouble lambda)
{
    UnsteadyAdvectionDiffusion::DoImplicitSolve(inarray, outarray,
                                                m_t1 - time, lambda);
}

bool UnsteadyAdjointTemperature::v_PostIntegrate(int step)
{
    // Checkpoints are written after this call and use m_fieldMetaDataMap.
    // Keep 'Time' as pseudo-time (needed by CheckForRestartTime) and record
    // the physical time separately.
    m_fieldMetaDataMap["PhysicalTime"] = std::to_string(m_t1 - m_time);

    return UnsteadyAdvectionDiffusion::v_PostIntegrate(step);
}

void UnsteadyAdjointTemperature::v_GenerateSummary(SolverUtils::SummaryList &s)
{
    UnsteadyAdvectionDiffusion::v_GenerateSummary(s);
    SolverUtils::AddSummaryItem(s, "Adjoint equation",
                                "unsteady temperature (pseudo-time)");
    SolverUtils::AddSummaryItem(s, "Adjoint nu", m_nu);
    SolverUtils::AddSummaryItem(
        s, "Physical window",
        "[" + std::to_string(m_t0) + ", " + std::to_string(m_t1) + "]");
    SolverUtils::AddSummaryItem(s, "Adjoint base flow",
                                m_baseFlow.GetSummary());
}

} // namespace Nektar
