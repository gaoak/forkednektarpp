///////////////////////////////////////////////////////////////////////////////
//
// File: AdjointBaseFlow.cpp
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
// Description: Base-flow reader / time interpolator for the temperature
//              adjoint solvers.
//
///////////////////////////////////////////////////////////////////////////////

#include <ADRSolver/Adjoint/AdjointBaseFlow.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

#include <LibUtilities/BasicUtils/Equation.h>
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/StringUtils.hpp>
#include <LibUtilities/BasicUtils/Vmath.hpp>

namespace Nektar
{

void AdjointBaseFlow::Init(
    const LibUtilities::SessionReaderSharedPtr &session,
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields)
{
    ASSERTL0(session, "AdjointBaseFlow::Init: null session.");
    ASSERTL0(fields.size() > 0, "AdjointBaseFlow::Init: no fields.");

    m_session = session;
    m_fields  = fields;

    m_spacedim = m_fields[0]->GetCoordim(0);
    m_npoints  = m_fields[0]->GetNpoints();
    ASSERTL0(m_spacedim >= 2 && m_spacedim <= 3,
             "AdjointBaseFlow only supports 2D and 3D fields.");

    ParseParameters();

    // Decide between file input and analytic input.  Nektar parameters are
    // numeric, so the file pattern is passed as a SolverInfo string property:
    //   <I PROPERTY="BaseFlowFile" VALUE="flow_%d.fld"/>
    bool hasFileInfo = m_session->DefinesSolverInfo("BaseFlowFile");
    bool hasFunction = m_session->DefinesFunction("BaseFlow");

    if (hasFileInfo)
    {
        m_filePattern = m_session->GetSolverInfo("BaseFlowFile");
    }
    else if (hasFunction)
    {
        m_analytic = true;
    }
    else
    {
        ASSERTL0(false,
                 "Temperature adjoint: define the SolverInfo property "
                 "'BaseFlowFile' or a session function 'BaseFlow' with "
                 "expressions.");
    }

    if (m_analytic)
    {
        ASSERTL0(m_nSlices <= 1,
                 "Analytic 'BaseFlow' cannot be used together with "
                 "N_slices > 1; use expressions of time and a single "
                 "base flow instead.");
        m_nSlices = 1;
        InitAnalyticMode();
    }
    else
    {
        InitFileMode();
    }

    m_initialised = true;

    // Load the velocity at the first available time so that GetVelocity()
    // and GetVelocityMinus() are valid before the first Update().
    Update(m_tStart);
}

void AdjointBaseFlow::ParseParameters()
{
    m_session->LoadParameter("N_slices", m_nSlices, 1);
    m_session->LoadParameter("N_start", m_start, 0);
    m_session->LoadParameter("N_skip", m_skip, 1);
    m_session->LoadParameter("BaseFlow_interporder", m_interpOrder, 2);
    m_session->LoadParameter("TimeSeriesStart", m_tStart, 0.0);
    m_session->LoadParameter("BaseFlowPeriodic", m_periodic, false);
    m_session->LoadParameter("Period", m_period, 0.0);

    ASSERTL0(m_nSlices >= 1, "N_slices must be >= 1.");
    ASSERTL0(m_skip >= 1, "N_skip must be >= 1.");
    ASSERTL0(m_interpOrder >= 1 && m_interpOrder <= 4,
             "BaseFlow_interporder must be in [1,4].");

    if (m_nSlices > 1)
    {
        ASSERTL0(m_interpOrder <= m_nSlices,
                 "BaseFlow_interporder must not exceed N_slices.");
        ASSERTL0(m_nSlices >= 2,
                 "A time series requires at least two slices.");
    }

    if (m_periodic)
    {
        ASSERTL0(m_period > 0.0,
                 "BaseFlowPeriodic requires a positive 'Period'.");
        ASSERTL0(m_interpOrder <= 2,
                 "Periodic base flow only supports interpolation order "
                 "1 (piecewise constant) or 2 (linear).");
        m_dt = m_period / m_nSlices;
    }
    else if (m_nSlices > 1)
    {
        ASSERTL0(m_session->DefinesParameter("TimeSeriesDt"),
                 "A non-periodic time series requires 'TimeSeriesDt'.");
        m_session->LoadParameter("TimeSeriesDt", m_dt, 0.0);
        ASSERTL0(m_dt > 0.0, "TimeSeriesDt must be positive.");
    }
    else
    {
        m_dt = 0.0;
    }

    m_cacheMax = std::max<size_t>(4, static_cast<size_t>(m_interpOrder) + 1);

    // Component names (SolverInfo string property, since parameters are
    // numeric): <I PROPERTY="BaseFlowVariables" VALUE="u,v"/>
    std::string varStr = "Vx,Vy,Vz";
    if (m_session->DefinesSolverInfo("BaseFlowVariables"))
    {
        varStr = m_session->GetSolverInfo("BaseFlowVariables");
    }
    std::vector<std::string> names;
    StringUtils::Split(names, varStr, StringUtils::IsAnyOf(", "),
                       StringUtils::eTokenCompressOn);

    // Drop empty tokens produced by leading/trailing separators.
    m_varNames.clear();
    for (auto &n : names)
    {
        if (!n.empty())
        {
            m_varNames.push_back(n);
        }
    }

    ASSERTL0(m_varNames.size() >= static_cast<size_t>(m_spacedim),
             "BaseFlowVariables must list at least " +
                 std::to_string(m_spacedim) + " component names.");
    m_varNames.resize(m_spacedim);
}

void AdjointBaseFlow::InitAnalyticMode()
{
    // ExpList::GetCoords does not allocate the output arrays.
    m_x = Array<OneD, NekDouble>(m_npoints, 0.0);
    m_y = Array<OneD, NekDouble>(m_npoints, 0.0);
    m_z = Array<OneD, NekDouble>(m_npoints, 0.0);
    m_fields[0]->GetCoords(m_x, m_y, m_z);

    m_equations.resize(m_spacedim);
    for (int d = 0; d < m_spacedim; ++d)
    {
        m_equations[d] = m_session->GetFunction("BaseFlow", m_varNames[d]);
        ASSERTL0(m_equations[d],
                 "Expression for BaseFlow component '" + m_varNames[d] +
                     "' is not defined.");
    }

    m_velPlus  = Array<OneD, Array<OneD, NekDouble>>(m_spacedim);
    m_velMinus = Array<OneD, Array<OneD, NekDouble>>(m_spacedim);
    for (int d = 0; d < m_spacedim; ++d)
    {
        m_velPlus[d]  = Array<OneD, NekDouble>(m_npoints, 0.0);
        m_velMinus[d] = Array<OneD, NekDouble>(m_npoints, 0.0);
    }
}

void AdjointBaseFlow::InitFileMode()
{
    if (m_nSlices > 1)
    {
        ASSERTL0(m_filePattern.find('%') != std::string::npos,
                 "BaseFlowFile must be a printf-style pattern containing "
                 "'%d' when N_slices > 1, e.g. flow_%d.fld.");
    }

    m_velPlus  = Array<OneD, Array<OneD, NekDouble>>(m_spacedim);
    m_velMinus = Array<OneD, Array<OneD, NekDouble>>(m_spacedim);
    for (int d = 0; d < m_spacedim; ++d)
    {
        m_velPlus[d]  = Array<OneD, NekDouble>(m_npoints, 0.0);
        m_velMinus[d] = Array<OneD, NekDouble>(m_npoints, 0.0);
    }

    // Load one slice now so that structural problems (missing variables,
    // incompatible mesh) are reported during initialisation.
    GetSlice(0);
}

std::string AdjointBaseFlow::SliceFilename(int sliceIdx) const
{
    if (m_nSlices <= 1)
    {
        return m_filePattern;
    }

    int index = m_start + sliceIdx * m_skip;

    // Support both a plain %d and a zero-padded width such as %05d.
    std::vector<char> buf(4096);
    std::snprintf(buf.data(), buf.size(), m_filePattern.c_str(), index);
    return std::string(buf.data());
}

void AdjointBaseFlow::LoadSliceFile(int sliceIdx, Slice &vel)
{
    const std::string filename = SliceFilename(sliceIdx);

    std::vector<LibUtilities::FieldDefinitionsSharedPtr> fieldDef;
    std::vector<std::vector<NekDouble>> fieldData;

    const size_t numExp = m_fields[0]->GetExpSize();
    Array<OneD, int> elementGIDs(numExp);
    for (size_t i = 0; i < numExp; ++i)
    {
        elementGIDs[i] =
            m_fields[0]->GetExp(i)->GetGeom()->GetGlobalID();
    }

    LibUtilities::FieldIOSharedPtr fld =
        LibUtilities::FieldIO::CreateForFile(m_session, filename);
    fld->Import(filename, fieldDef, fieldData,
                LibUtilities::NullFieldMetaDataMap, elementGIDs);

    ASSERTL0(fieldDef.size() > 0,
             "No field definitions found in base-flow file '" + filename +
                 "'.");

    vel.resize(m_spacedim);
    Array<OneD, NekDouble> coeffs(m_fields[0]->GetNcoeffs(), 0.0);
    for (int d = 0; d < m_spacedim; ++d)
    {
        vel[d] = Array<OneD, NekDouble>(m_npoints, 0.0);

        bool found = false;
        Vmath::Zero(coeffs.size(), coeffs, 1);
        for (size_t i = 0; i < fieldDef.size(); ++i)
        {
            auto it = std::find(fieldDef[i]->m_fields.begin(),
                                fieldDef[i]->m_fields.end(), m_varNames[d]);
            if (it == fieldDef[i]->m_fields.end())
            {
                continue;
            }

            found = true;
            std::string varName = *it;
            m_fields[0]->ExtractDataToCoeffs(fieldDef[i], fieldData[i],
                                             varName, coeffs);
        }

        if (!found)
        {
            std::string available;
            for (size_t i = 0; i < fieldDef[0]->m_fields.size(); ++i)
            {
                available += (i ? ", " : "") + fieldDef[0]->m_fields[i];
            }
            ASSERTL0(false,
                     "Base-flow file '" + filename + "' does not contain "
                     "variable '" + m_varNames[d] + "'. Available: " +
                     available + ".  Set 'BaseFlowVariables' accordingly.");
        }

        bool oldWaveSpace = m_fields[0]->GetWaveSpace();
        m_fields[0]->SetWaveSpace(false);
        m_fields[0]->BwdTrans(coeffs, vel[d]);
        m_fields[0]->SetWaveSpace(oldWaveSpace);
    }
}

const AdjointBaseFlow::Slice &AdjointBaseFlow::GetSlice(int sliceIdx)
{
    ASSERTL0(sliceIdx >= 0 && sliceIdx < m_nSlices,
             "Base-flow slice index out of range.");

    auto it = m_cache.find(sliceIdx);
    if (it == m_cache.end())
    {
        Slice s;
        LoadSliceFile(sliceIdx, s);
        it = m_cache.emplace(sliceIdx, std::move(s)).first;
    }

    // Touch in the LRU list.
    auto pos = std::find(m_lruOrder.begin(), m_lruOrder.end(), sliceIdx);
    if (pos != m_lruOrder.end())
    {
        m_lruOrder.erase(pos);
    }
    m_lruOrder.push_back(sliceIdx);

    // Evict least-recently-used slices while the cache is too large, never
    // evicting the slice just requested.
    while (m_cache.size() > m_cacheMax && !m_lruOrder.empty())
    {
        int victim = m_lruOrder.front();
        if (victim == sliceIdx)
        {
            m_lruOrder.pop_front();
            m_lruOrder.push_back(victim);
            continue;
        }
        m_lruOrder.pop_front();
        m_cache.erase(victim);
    }

    return m_cache.find(sliceIdx)->second;
}

void AdjointBaseFlow::PruneCache(const std::set<int> &keep)
{
    while (m_cache.size() > m_cacheMax && !m_lruOrder.empty())
    {
        int victim = m_lruOrder.front();
        m_lruOrder.pop_front();
        if (keep.find(victim) != keep.end())
        {
            m_lruOrder.push_back(victim);
            continue;
        }
        m_cache.erase(victim);
    }
}

void AdjointBaseFlow::Interpolate(
    NekDouble time, Array<OneD, Array<OneD, NekDouble>> &out)
{
    const int p = std::min(m_interpOrder, m_nSlices);

    NekDouble xf;
    if (m_periodic)
    {
        NekDouble tt = std::fmod(time - m_tStart, m_period);
        if (tt < 0.0)
        {
            tt += m_period;
        }
        xf = tt / m_dt;
        if (xf >= m_nSlices)
        {
            xf -= m_nSlices;
        }
    }
    else
    {
        xf = (time - m_tStart) / m_dt;
    }

    // Clamp outside the sampled window (non-periodic).
    if (!m_periodic)
    {
        if (xf <= 0.0)
        {
            const Slice &s = GetSlice(0);
            for (int d = 0; d < m_spacedim; ++d)
            {
                Vmath::Vcopy(m_npoints, s[d], 1, out[d], 1);
            }
            return;
        }
        if (xf >= static_cast<NekDouble>(m_nSlices - 1))
        {
            const Slice &s = GetSlice(m_nSlices - 1);
            for (int d = 0; d < m_spacedim; ++d)
            {
                Vmath::Vcopy(m_npoints, s[d], 1, out[d], 1);
            }
            return;
        }
    }

    int i0 = static_cast<int>(std::floor(xf));
    NekDouble frac;
    if (m_periodic)
    {
        // Wrap to a valid interval.  Do not clamp to nSlices-2 here: the
        // wrap-around interval starts at the last slice.
        i0   = ((i0 % m_nSlices) + m_nSlices) % m_nSlices;
        frac = xf - std::floor(xf);
    }
    else
    {
        i0   = std::max(0, std::min(i0, m_nSlices - 2));
        frac = xf - static_cast<NekDouble>(i0);
    }

    if (p == 1)
    {
        // Piecewise constant: value of the left sample.
        int idx = std::max(0, std::min(static_cast<int>(std::floor(xf)),
                                      m_nSlices - 1));
        const Slice &s = GetSlice(idx);
        for (int d = 0; d < m_spacedim; ++d)
        {
            Vmath::Vcopy(m_npoints, s[d], 1, out[d], 1);
        }
        return;
    }

    if (m_periodic)
    {
        // Linear interpolation with wrap-around.
        const Slice &s0 = GetSlice(i0);
        const Slice &s1 = GetSlice((i0 + 1) % m_nSlices);
        for (int d = 0; d < m_spacedim; ++d)
        {
            Vmath::Vcopy(m_npoints, s0[d], 1, out[d], 1);
            Vmath::Svtvp(m_npoints, frac, s1[d], 1, out[d], 1, out[d], 1);
            Vmath::Svtvp(m_npoints, -frac, s0[d], 1, out[d], 1, out[d], 1);
        }
        return;
    }

    // Non-periodic Lagrange interpolation on p consecutive samples.
    int padLeft  = p / 2 - 1;
    int padRight = p - 1 - padLeft;
    if (padLeft > i0)
    {
        padLeft = i0;
    }
    if (padRight > m_nSlices - 1 - i0)
    {
        padRight = m_nSlices - 1 - i0;
    }
    padLeft = p - 1 - padRight;

    std::vector<Slice> slices(p);
    for (int j = 0; j < p; ++j)
    {
        // Copy (shallow) so that later cache evictions cannot invalidate it.
        slices[j] = GetSlice(i0 - padLeft + j);
    }

    const NekDouble s = xf - static_cast<NekDouble>(i0 - padLeft);
    std::vector<NekDouble> coeff(p, 1.0);
    for (int i = 0; i < p; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            if (i != j)
            {
                coeff[i] *= (s - j) / static_cast<NekDouble>(i - j);
            }
        }
    }

    for (int d = 0; d < m_spacedim; ++d)
    {
        Vmath::Zero(m_npoints, out[d], 1);
        for (int j = 0; j < p; ++j)
        {
            Vmath::Svtvp(m_npoints, coeff[j], slices[j][d], 1, out[d], 1,
                         out[d], 1);
        }
    }
}

void AdjointBaseFlow::Update(NekDouble time)
{
    ASSERTL0(m_initialised,
             "AdjointBaseFlow::Update called before Init().");

    if (m_analytic)
    {
        for (int d = 0; d < m_spacedim; ++d)
        {
            m_equations[d]->Evaluate(m_x, m_y, m_z, time, m_velPlus[d]);
        }
    }
    else if (m_nSlices == 1)
    {
        const Slice &s = GetSlice(0);
        for (int d = 0; d < m_spacedim; ++d)
        {
            Vmath::Vcopy(m_npoints, s[d], 1, m_velPlus[d], 1);
        }
    }
    else
    {
        Interpolate(time, m_velPlus);
    }

    for (int d = 0; d < m_spacedim; ++d)
    {
        Vmath::Vcopy(m_npoints, m_velPlus[d], 1, m_velMinus[d], 1);
        Vmath::Neg(m_npoints, m_velMinus[d], 1);
    }
}

NekDouble AdjointBaseFlow::GetTimeEnd() const
{
    if (m_nSlices <= 1)
    {
        return m_tStart;
    }
    if (m_periodic)
    {
        return m_tStart + m_period;
    }
    return m_tStart + (m_nSlices - 1) * m_dt;
}

std::string AdjointBaseFlow::GetSummary() const
{
    std::ostringstream ss;
    if (m_analytic)
    {
        ss << "analytic BaseFlow (" << m_varNames[0];
        for (int d = 1; d < m_spacedim; ++d)
        {
            ss << "," << m_varNames[d];
        }
        ss << ")";
    }
    else if (m_nSlices <= 1)
    {
        ss << "file " << m_filePattern;
    }
    else
    {
        ss << m_nSlices << " slices from " << SliceFilename(0) << " to "
           << SliceFilename(m_nSlices - 1) << ", t in [" << m_tStart << ", "
           << GetTimeEnd() << "]";
        if (m_periodic)
        {
            ss << " (periodic, period " << m_period << ")";
        }
        ss << ", interpolation order " << m_interpOrder;
    }
    return ss.str();
}

} // namespace Nektar
