///////////////////////////////////////////////////////////////////////////////
//
// File: ForcingNoise.cpp
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
// Description: White noise forcing
//
///////////////////////////////////////////////////////////////////////////////

#include <MultiRegions/ExpListHomogeneous1D.h>
#include <SolverCore/Forcing/ForcingNoise.h>

namespace Nektar::SolverCore
{

std::string ForcingNoise::className =
    GetForcingFactory().RegisterCreatorFunction("Noise", ForcingNoise::create,
                                                "White Noise Forcing");

// The planes this rank holds of a 3DH1 expansion, and one otherwise.
static unsigned int NumLocalPlanes(
    const MultiRegions::ExpListSharedPtr &expList)
{
    auto homoExpList =
        std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(expList);
    return homoExpList
               ? static_cast<unsigned int>(homoExpList->GetZIDs().size())
               : 1u;
}

/**
 * The seed is the rank, so that ranks draw different noise.
 */
ForcingNoise::ForcingNoise(const LibUtilities::SessionReaderSharedPtr &session,
                           const MultiRegions::ExpListSharedPtr &expList,
                           const std::vector<std::string> &variables)
    : Forcing(session, expList, variables),
      m_randomGenerator(session->GetComm()->GetRank()),
      m_noise(
          "ForcingNoise noise",
          MultiRegions::GetBlockAttributes<double, FieldState::Phys>(expList),
          variables, NumLocalPlanes(expList))
{
}

/**
 * Reads WHITENOISE, the standard deviation of the noise, and the optional
 * UPDATEFREQ and NSTEPS, and draws the first noise. An UPDATEFREQ of 0 never
 * redraws the noise, and an NSTEPS of 0 applies it throughout.
 */
void ForcingNoise::v_InitObject(const LibUtilities::SessionNode &force)
{
    ASSERTL0(force.DefinesChild("WHITENOISE"),
             "Requires WHITENOISE tag specifying magnitude of white noise "
             "force.");

    m_sigma      = force.GetValueAs<double>("WHITENOISE");
    m_updateFreq = force.GetValueAs<unsigned int>("UPDATEFREQ", 0);
    m_numSteps   = force.GetValueAs<unsigned int>("NSTEPS", 0);

    m_math.fillNormal(m_randomGenerator, 0.0, m_sigma, m_noise);
}

/**
 * Adds, or assigns without append, scale times the noise to out. Each call is
 * one application for UPDATEFREQ and NSTEPS.
 */
void ForcingNoise::v_Apply(
    [[maybe_unused]] LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out,
    [[maybe_unused]] const double time, const double scale)
{
    if (m_numSteps && m_index >= m_numSteps)
    {
        if (!m_append)
        {
            m_math.zero(out);
        }
        return;
    }

    if (m_updateFreq && m_index && m_index % m_updateFreq == 0)
    {
        m_math.fillNormal(m_randomGenerator, 0.0, m_sigma, m_noise);
    }

    if (m_append)
    {
        m_math.daxpy(scale, m_noise, out, out);
    }
    else
    {
        m_math.mul(scale, m_noise, out);
    }

    ++m_index;
}

void ForcingNoise::v_SetAppend(const bool append)
{
    m_append = append;
}

} // namespace Nektar::SolverCore
