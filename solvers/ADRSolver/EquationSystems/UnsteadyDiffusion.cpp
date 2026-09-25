///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyDiffusion.cpp
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
// Description: Unsteady diffusion solve routines
//
///////////////////////////////////////////////////////////////////////////////

#include <ADRSolver/EquationSystems/UnsteadyDiffusion.h>
#include <SolverUtils/Diffusion/DiffusionIP.h>

namespace Nektar
{

std::string UnsteadyDiffusion::className =
    GetEquationSystemFactory().RegisterCreatorFunction(
        "UnsteadyDiffusion", UnsteadyDiffusion::create);

UnsteadyDiffusion::UnsteadyDiffusion(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph)
{
}

/**
 * @brief Initialisation object for the unsteady diffusion problem.
 */
void UnsteadyDiffusion::v_InitObject(bool DeclareFields)
{
    UnsteadySystem::v_InitObject(DeclareFields);

    // Load diffusion parameter
    m_session->LoadParameter("epsilon", m_epsilon, 1.0);

    m_session->MatchSolverInfo("SpectralVanishingViscosity", "True",
                               m_useSpecVanVisc, false);

    if (m_useSpecVanVisc)
    {
        m_session->LoadParameter("SVVCutoffRatio", m_sVVCutoffRatio, 0.75);
        m_session->LoadParameter("SVVDiffCoeff", m_sVVDiffCoeff, 0.1);
    }

    int npoints = m_fields[0]->GetNpoints();

    if (m_session->DefinesParameter("d00"))
    {
        m_d00 = m_session->GetParameter("d00");
        m_varcoeff[StdRegions::eVarCoeffD00] =
            Array<OneD, NekDouble>(npoints, m_session->GetParameter("d00"));
    }
    if (m_session->DefinesParameter("d11"))
    {
        m_d11 = m_session->GetParameter("d11");
        m_varcoeff[StdRegions::eVarCoeffD11] =
            Array<OneD, NekDouble>(npoints, m_session->GetParameter("d11"));
    }
    if (m_session->DefinesParameter("d22"))
    {
        m_d22 = m_session->GetParameter("d22");
        m_varcoeff[StdRegions::eVarCoeffD22] =
            Array<OneD, NekDouble>(npoints, m_session->GetParameter("d22"));
    }

    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            std::string diffName;

            // Do not forwards transform initial condition
            m_homoInitialFwd = false;

            m_session->LoadSolverInfo("DiffusionType", diffName, "LDG");
            m_diffusion = SolverUtils::GetDiffusionFactory().CreateInstance(
                diffName, diffName);
            m_diffusion->SetFluxVector(&UnsteadyDiffusion::GetFluxVector, this);

            if (diffName == "InteriorPenalty")
            {
                // DiffusionIP does not read the flux vector set above; it
                // drives its own functors, and each one it reaches while
                // unbound is a bad_function_call mid-run. See the note on the
                // declarations in the header.
                m_diffusion->SetDiffusionFluxCons(
                    &UnsteadyDiffusion::GetFluxVectorCons, this);
                m_diffusion->SetDiffusionFluxConsTrace(
                    &UnsteadyDiffusion::GetFluxVectorConsTrace, this);
                m_diffusion->SetDiffusionSymmFluxCons(
                    &UnsteadyDiffusion::GetSymmFluxVectorCons, this);
                m_diffusion->SetSpecialBndTreat(
                    &UnsteadyDiffusion::SpecialBndTreat, this);

                // The fields here are independent scalars, and DiffusionIP's
                // default average treats them as compressible conservative
                // variables - which divides by the first one. See
                // DiffusionIP::SetScalarAveraging().
                std::static_pointer_cast<SolverUtils::DiffusionIP>(m_diffusion)
                    ->SetScalarAveraging(true);
            }

            m_diffusion->InitObject(m_session, m_fields);
            break;
        }
        case MultiRegions::eGalerkin:
        {
            // In case of Galerkin explicit diffusion gives an error
            if (m_explicitDiffusion)
            {
                ASSERTL0(false, "Explicit Galerkin diffusion not set up.");
            }
            // In case of Galerkin implicit diffusion: do nothing
            break;
        }
        default:
        {
            ASSERTL0(false, "Unknown projection scheme");
            break;
        }
    }

    m_ode.DefineOdeRhs(&UnsteadyDiffusion::DoOdeRhs, this);
    m_ode.DefineProjection(&UnsteadyDiffusion::DoOdeProjection, this);
    m_ode.DefineImplicitSolve(&UnsteadyDiffusion::DoImplicitSolve, this);
}

/* @brief Compute the right-hand side for the unsteady diffusion problem.
 *
 * @param inarray    Given fields.
 * @param outarray   Calculated solution.
 * @param time       Time.
 */
void UnsteadyDiffusion::DoOdeRhs(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray,
    [[maybe_unused]] const NekDouble time)
{
    // Number of fields (variables of the problem)
    int nVariables = inarray.size();

    // RHS computation using the new diffusion base class
    m_diffusion->Diffuse(nVariables, m_fields, inarray, outarray);
}

/**
 * @brief Compute the projection for the unsteady diffusion problem.
 *
 * @param inarray    Given fields.
 * @param outarray   Calculated solution.
 * @param time       Time.
 */
void UnsteadyDiffusion::DoOdeProjection(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time)
{
    int i;
    int nvariables = inarray.size();
    SetBoundaryConditions(time);

    switch (m_projectionType)
    {
        case MultiRegions::eDiscontinuous:
        {
            // Just copy over array
            if (inarray != outarray)
            {
                int npoints = GetNpoints();

                for (i = 0; i < nvariables; ++i)
                {
                    Vmath::Vcopy(npoints, inarray[i], 1, outarray[i], 1);
                }
            }
            break;
        }
        case MultiRegions::eGalerkin:
        {
            Array<OneD, NekDouble> coeffs(m_fields[0]->GetNcoeffs());

            for (i = 0; i < nvariables; ++i)
            {
                m_fields[i]->FwdTrans(inarray[i], coeffs);
                m_fields[i]->BwdTrans(coeffs, outarray[i]);
            }
            break;
        }
        default:
        {
            ASSERTL0(false, "Unknown projection scheme");
            break;
        }
    }
}

/**
 * @brief Implicit solution of the unsteady diffusion problem.
 */
void UnsteadyDiffusion::DoImplicitSolve(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray,
    [[maybe_unused]] const NekDouble time, const NekDouble lambda)
{
    StdRegions::ConstFactorMap factors;

    int nvariables                     = inarray.size();
    int npoints                        = m_fields[0]->GetNpoints();
    factors[StdRegions::eFactorLambda] = 1.0 / lambda / m_epsilon;
    factors[StdRegions::eFactorTau]    = 1.0;

    if (m_useSpecVanVisc)
    {
        factors[StdRegions::eFactorSVVCutoffRatio] = m_sVVCutoffRatio;
        factors[StdRegions::eFactorSVVDiffCoeff]   = m_sVVDiffCoeff / m_epsilon;
    }

    // We solve ( \nabla^2 - HHlambda ) Y[i] = rhs [i]
    // inarray = input: \hat{rhs} -> output: \hat{Y}
    // outarray = output: nabla^2 \hat{Y}
    // where \hat = modal coeffs
    for (int i = 0; i < nvariables; ++i)
    {
        // Multiply 1.0/timestep/lambda
        Vmath::Smul(npoints, -factors[StdRegions::eFactorLambda], inarray[i], 1,
                    outarray[i], 1);

        // Solve a system of equations with Helmholtz solver
        m_fields[i]->HelmSolve(outarray[i], m_fields[i]->UpdateCoeffs(),
                               factors, m_varcoeff);

        m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(), outarray[i]);

        m_fields[i]->SetPhysState(false);
    }
}

/**
 * @brief Return the flux vector for the unsteady diffusion problem.
 */
void UnsteadyDiffusion::GetFluxVector(
    [[maybe_unused]] const Array<OneD, Array<OneD, NekDouble>> &inarray,
    const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &qfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscousTensor)
{
    unsigned int nDim              = qfield.size();
    unsigned int nConvectiveFields = qfield[0].size();
    unsigned int nPts              = qfield[0][0].size();

    NekDouble d[3] = {m_d00, m_d11, m_d22};

    for (unsigned int j = 0; j < nDim; ++j)
    {
        for (unsigned int i = 0; i < nConvectiveFields; ++i)
        {
            Vmath::Smul(nPts, m_epsilon * d[j], qfield[j][i], 1,
                        viscousTensor[j][i], 1);
        }
    }
}

/**
 * @brief Volume viscous flux for the InteriorPenalty operator.
 *
 * The same tensor GetFluxVector() builds, in the shape DiffusionIP asks for
 * it: indexed [direction][field], and reporting through @p nonZeroIndex which
 * fields carry a non-zero flux. For a scalar problem that is all of them -
 * the compressible solver leaves density out, which is what the index exists
 * for. Leaving it empty is not neutral: DiffusionIP loops over it to decide
 * what to write, so an unset index yields a silent zero.
 */
void UnsteadyDiffusion::GetFluxVectorCons(
    const int nDim,
    [[maybe_unused]] const Array<OneD, Array<OneD, NekDouble>> &inarray,
    const TensorOfArray3D<NekDouble> &qfield,
    TensorOfArray3D<NekDouble> &viscousTensor, Array<OneD, int> &nonZeroIndex,
    [[maybe_unused]] const Array<OneD, Array<OneD, NekDouble>> &normal)
{
    unsigned int nConvectiveFields = qfield[0].size();
    unsigned int nPts              = qfield[0][0].size();

    NekDouble d[3] = {m_d00, m_d11, m_d22};

    nonZeroIndex = Array<OneD, int>(nConvectiveFields);
    for (unsigned int i = 0; i < nConvectiveFields; ++i)
    {
        nonZeroIndex[i] = i;
    }

    for (int j = 0; j < nDim; ++j)
    {
        for (unsigned int i = 0; i < nConvectiveFields; ++i)
        {
            Vmath::Smul(nPts, m_epsilon * d[j], qfield[j][i], 1,
                        viscousTensor[j][i], 1);
        }
    }
}

/**
 * @brief The same flux on a trace, contracted with the trace normal.
 *
 * On the trace DiffusionIP wants n . (D grad u) rather than the tensor, so
 * the output carries a single direction. Binding the volume form here indexes
 * the output past that first direction and fails deep inside SolverUtils.
 */
void UnsteadyDiffusion::GetFluxVectorConsTrace(
    const int nDim,
    [[maybe_unused]] const Array<OneD, Array<OneD, NekDouble>> &inarray,
    const TensorOfArray3D<NekDouble> &qfield,
    TensorOfArray3D<NekDouble> &viscousTensor, Array<OneD, int> &nonZeroIndex,
    const Array<OneD, Array<OneD, NekDouble>> &normal)
{
    unsigned int nConvectiveFields = qfield[0].size();
    unsigned int nPts              = qfield[0][0].size();

    NekDouble d[3] = {m_d00, m_d11, m_d22};

    nonZeroIndex = Array<OneD, int>(nConvectiveFields);
    for (unsigned int i = 0; i < nConvectiveFields; ++i)
    {
        nonZeroIndex[i] = i;
    }

    for (unsigned int i = 0; i < nConvectiveFields; ++i)
    {
        Vmath::Zero(nPts, viscousTensor[0][i], 1);
        for (int j = 0; j < nDim; ++j)
        {
            for (unsigned int p = 0; p < nPts; ++p)
            {
                viscousTensor[0][i][p] +=
                    m_epsilon * d[j] * qfield[j][i][p] * normal[j][p];
            }
        }
    }
}

/**
 * @brief Symmetric flux D n [u] for the SIP/NIP variants.
 *
 * Not reached while IPSymmFluxCoeff is zero, which is the default, but
 * DiffusionIP is entitled to call it whenever that parameter is set and an
 * unbound functor would fail exactly then.
 */
void UnsteadyDiffusion::GetSymmFluxVectorCons(
    const int nDim,
    [[maybe_unused]] const Array<OneD, Array<OneD, NekDouble>> &inaverg,
    const Array<OneD, Array<OneD, NekDouble>> &injumps,
    TensorOfArray3D<NekDouble> &outarray, Array<OneD, int> &nonZeroIndex,
    const Array<OneD, Array<OneD, NekDouble>> &normal)
{
    unsigned int nConvectiveFields = injumps.size();
    unsigned int nPts              = injumps[0].size();

    NekDouble d[3] = {m_d00, m_d11, m_d22};

    nonZeroIndex = Array<OneD, int>(nConvectiveFields);
    for (unsigned int i = 0; i < nConvectiveFields; ++i)
    {
        nonZeroIndex[i] = i;
    }

    for (int j = 0; j < nDim; ++j)
    {
        for (unsigned int i = 0; i < nConvectiveFields; ++i)
        {
            Vmath::Vmul(nPts, injumps[i], 1, normal[j], 1, outarray[j][i], 1);
            Vmath::Smul(nPts, m_epsilon * d[j], outarray[j][i], 1,
                        outarray[j][i], 1);
        }
    }
}

/**
 * @brief Boundary fix-up hook. A scalar diffusion has nothing to correct -
 * this exists for the compressible solver, which resets the energy at a wall
 * - but DiffusionIP calls it unconditionally, so it has to be bound.
 */
void UnsteadyDiffusion::SpecialBndTreat(
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &consvar)
{
}

void UnsteadyDiffusion::v_GenerateSummary(SummaryList &s)
{
    UnsteadySystem::v_GenerateSummary(s);
    if (m_useSpecVanVisc)
    {
        std::stringstream ss;
        ss << "SVV (cut off = " << m_sVVCutoffRatio
           << ", coeff = " << m_sVVDiffCoeff << ")";
        AddSummaryItem(s, "Smoothing", ss.str());
    }
}

} // namespace Nektar
