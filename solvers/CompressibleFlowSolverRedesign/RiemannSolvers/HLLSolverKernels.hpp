///////////////////////////////////////////////////////////////////////////////
//
// File: HLLSolverKernels.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/SafeDivide.hpp>

#include "RiemannSolvers/RoeAverage.hpp"

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::detail
{

/**
 * @brief The HLL flux through the trace normal, in Cartesian components (as
 * the legacy HLLSolver), with the Einfeldt wave speeds from the Roe average.
 */
template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM>
struct HLLSolverKernel
{
    /// @p normal: the unit trace normal, @p NDIM components at the states'
    /// stride @p blksize, in TData (a scalar or a SIMD vector of points).
    template <typename TData>
    NEK_HOSTDEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                          const size_t blksize,
                                          const TData *fwd, const TData *bwd,
                                          const TData *normal, TData *flux)
    {
        using std::max;
        using std::min;

        // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
        const auto l = fwd;
        const auto r = bwd;

        // The two sides' primitive quantities and the Roe averages.
        RoeAverageState<NDIM, TData> s;
        RoeAverages<NDIM>(EoS, l, r, blksize, s);
        const TData rhoL = l[0], rhoR = r[0];
        const TData EL = s.EL, ER = s.ER, pL = s.pL, pR = s.pR;
        const auto &uL = s.uL, &uR = s.uR, &uRoe = s.uRoe;
        const TData cRoe = s.cRoe;
        const TData cL   = GetSoundSpeed(EoS, rhoL, s.eL);
        const TData cR   = GetSoundSpeed(EoS, rhoR, s.eR);

        TData n[NDIM];
        TData unL = TData(0.0), unR = TData(0.0), unRoe = TData(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            n[d] = normal[d * blksize];
            unL += n[d] * uL[d];
            unR += n[d] * uR[d];
            unRoe += n[d] * uRoe[d];
        }

        // The largest wave speeds, clamped at zero: SL- = min(SL, 0) and
        // SR+ = max(SR, 0). With them the three cases of HLL - the left
        // state's flux where SL >= 0, the right's where SR <= 0, and the HLL
        // flux between - are the one expression
        //   F = (SR+ FL - SL- FR + SR+ SL- (UR - UL)) / (SR+ - SL-),
        // so every lane takes it without a branch. SR+ - SL- is zero only
        // when both speeds are, where the flux is taken as zero.
        const TData SLm  = min(min(unL - cL, unRoe - cRoe), TData(0.0));
        const TData SRp  = max(max(unR + cR, unRoe + cRoe), TData(0.0));
        const TData invS = SafeDivide(TData(1.0), SRp - SLm);
        const TData SLR  = SRp * SLm;

        // Jumps: density and momentum.
        const TData dRho = rhoR - rhoL;

        flux[0] = (SRp * rhoL * unL - SLm * rhoR * unR + SLR * dRho) * invS;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const TData rhouL = l[(1u + d) * blksize],
                        rhouR = r[(1u + d) * blksize];
            const TData dm    = rhouR - rhouL;
            flux[(1u + d) * blksize] =
                (SRp * (rhouL * unL + pL * n[d]) -
                 SLm * (rhouR * unR + pR * n[d]) + SLR * dm) *
                invS;
        }
        flux[(1u + NDIM) * blksize] =
            (SRp * unL * (EL + pL) - SLm * unR * (ER + pR) + SLR * (ER - EL)) *
            invS;
    }
};

} // namespace Nektar::detail
