///////////////////////////////////////////////////////////////////////////////
//
// File: AdjointBaseFlow.h
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
//              adjoint solvers.  Provides -u(x,t) for the adjoint
//              advection velocity and +u(x,t) for boundary-condition
//              post-processing (the forward normal velocity).
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_ADRSOLVER_ADJOINT_ADJOINTBASEFLOW_H
#define NEKTAR_SOLVERS_ADRSOLVER_ADJOINT_ADJOINTBASEFLOW_H

#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <MultiRegions/ExpList.h>

namespace Nektar
{

/**
 * @brief Read a steady base flow or a time series of base-flow slices and
 *        provide the negated velocity required by the temperature adjoint
 *        equation, V = -u.
 *
 * Input modes
 * -----------
 *  - File: SolverInfo property @c BaseFlowFile, containing a Nektar field
 *    file or a printf-style pattern such as @c flow_%d.fld when
 *    @c N_slices > 1.  (Nektar parameters are numeric, so this is a
 *    SolverInfo string: @c \<I PROPERTY="BaseFlowFile" VALUE="..."/\>.)
 *  - Analytic: a session function @c BaseFlow written as expressions for the
 *    components listed in the SolverInfo property @c BaseFlowVariables
 *    (default @c Vx,Vy,Vz).
 *
 * The field file must use the same mesh and expansion as the adjoint field.
 * No runtime mesh interpolation is performed (use FieldConvert offline).
 *
 * Time series parameters (only used when @c N_slices > 1)
 * -------------------------------------------------------
 *  - @c N_start, @c N_skip : slice index of the first file and stride.
 *  - @c TimeSeriesStart     : physical time of the first slice (default 0).
 *  - @c TimeSeriesDt        : uniform time spacing (required unless periodic).
 *  - @c BaseFlow_interporder: number of interpolation points:
 *       1 = piecewise constant, 2 = linear (default), 3/4 = Lagrange.
 *  - @c BaseFlowPeriodic / @c Period : periodic sampling.  Slices are assumed
 *    equally spaced over @c Period with spacing Period/N_slices.
 *
 * Only at most max(4, interporder+1) slices are kept in memory at any time.
 */
class AdjointBaseFlow
{
public:
    AdjointBaseFlow() = default;
    ~AdjointBaseFlow() = default;

    /// Read parameters and the first base-flow data.
    void Init(const LibUtilities::SessionReaderSharedPtr &session,
              const Array<OneD, MultiRegions::ExpListSharedPtr> &fields);

    /// Interpolate the base flow at physical time @p time and update
    /// GetVelocityMinus()/GetVelocity().  Times outside the sampled range are
    /// clamped (non-periodic).
    void Update(NekDouble time);

    /// Adjoint advection velocity, V = -u(t).  Updated by Update().
    const Array<OneD, Array<OneD, NekDouble>> &GetVelocityMinus() const
    {
        return m_velMinus;
    }

    /// Forward velocity u(t).  Updated by Update().
    const Array<OneD, Array<OneD, NekDouble>> &GetVelocity() const
    {
        return m_velPlus;
    }

    /// Number of spatial dimensions of the returned velocity (2 or 3).
    int GetSpaceDim() const
    {
        return m_spacedim;
    }

    /// True if a single (time-independent) base flow was supplied.
    bool IsSteady() const
    {
        return m_nSlices <= 1;
    }

    /// True if the velocity comes from session expressions rather than files.
    bool IsAnalytic() const
    {
        return m_analytic;
    }

    int GetNumSlices() const
    {
        return m_nSlices;
    }

    NekDouble GetTimeStart() const
    {
        return m_tStart;
    }

    /// Last sampled time (non-periodic) / period end (periodic).  Returns the
    /// start time for a steady base flow.
    NekDouble GetTimeEnd() const;

    /// Human-readable description of the input mode, for summary output.
    std::string GetSummary() const;

private:
    /// A loaded slice: physical velocity components on the local mesh.
    using Slice = std::vector<Array<OneD, NekDouble>>;

    void ParseParameters();
    void InitFileMode();
    void InitAnalyticMode();
    void CheckVariableNames(const std::vector<std::string> &fileVars) const;

    /// Return a slice from the cache, loading it if necessary.
    const Slice &GetSlice(int sliceIdx);
    /// Read one slice from disk into @p vel (local elements only).
    void LoadSliceFile(int sliceIdx, Slice &vel);
    std::string SliceFilename(int sliceIdx) const;
    /// Drop cached slices that are not in @p keep.
    void PruneCache(const std::set<int> &keep);
    /// Interpolate the loaded slices at time @p time into @p out.
    void Interpolate(NekDouble time,
                     Array<OneD, Array<OneD, NekDouble>> &out);

    LibUtilities::SessionReaderSharedPtr m_session;
    Array<OneD, MultiRegions::ExpListSharedPtr> m_fields;

    int m_spacedim = 0;
    int m_npoints  = 0;
    std::vector<std::string> m_varNames;

    bool m_analytic = false;
    bool m_periodic = false;
    bool m_initialised = false;

    std::string m_filePattern;
    int m_nSlices = 1;
    int m_start   = 0;
    int m_skip    = 1;

    NekDouble m_tStart    = 0.0;
    NekDouble m_dt        = 0.0;
    NekDouble m_period    = 0.0;
    int m_interpOrder     = 2;

    /// Analytic mode data.
    Array<OneD, NekDouble> m_x, m_y, m_z;
    std::vector<LibUtilities::EquationSharedPtr> m_equations;

    /// Loaded slices, keyed by slice index.  m_lruOrder front = least recent.
    std::map<int, Slice> m_cache;
    std::deque<int> m_lruOrder;
    size_t m_cacheMax = 4;

    Array<OneD, Array<OneD, NekDouble>> m_velPlus;
    Array<OneD, Array<OneD, NekDouble>> m_velMinus;
};

} // namespace Nektar

#endif
