///////////////////////////////////////////////////////////////////////////////
//
// File: HLLCSolverKernels.hpp
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

#include <LibUtilities/Backends/Backends.hpp>

#include <CompressibleFlowSolverRedesign/RiemannSolvers/RoeAverage.hpp>

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::detail
{

/**
 * @brief The HLLC flux through the trace normal, in Cartesian components (as
 * the legacy HLLCSolver), with the Einfeldt wave speeds from the Roe average.
 */
template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM>
struct HLLCSolverKernel
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

        // Mass flux through the normal from each side, and the largest wave
        // speeds.
        const TData mnL = rhoL * unL;
        const TData mnR = rhoR * unR;
        const TData SL  = min(unL - cL, unRoe - cRoe);
        const TData SR  = max(unR + cR, unRoe + cRoe);

        // HLLC: the star state on the side the contact wave SM leaves
        // behind. Only the normal velocity changes there, to SM; the
        // tangential velocity is advected unaltered, which in Cartesian
        // components is u + (SM - u.n) n.
        //
        // Every point, or lane of a vector, forms the one-sided fluxes and the
        // star flux, and takes the one its wave speeds select: the left flux
        // where SL >= 0, the right where SR <= 0, the star flux otherwise. A
        // flux not taken may hold any value.
        const TData SM = (pR - pL + mnL * (SL - unL) - mnR * (SR - unR)) /
                         (rhoL * (SL - unL) - rhoR * (SR - unR));
        const auto left  = SL >= TData(0.0);
        const auto right = SR <= TData(0.0);
        const auto side  = SM >= TData(0.0);
        const TData rho  = Select(side, rhoL, rhoR);
        const TData un   = Select(side, unL, unR);
        const TData p    = Select(side, pL, pR);
        const TData E    = Select(side, EL, ER);
        const TData S    = Select(side, SL, SR);
        const TData mn   = rho * un;
        const TData rhoM = rho * (S - un) / (S - SM);
        const TData EM =
            rhoM * (E / rho + (SM - un) * (SM + p / (rho * (S - un))));

        flux[0] = Select(left, mnL, Select(right, mnR, mn + S * (rhoM - rho)));
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const TData rhouL = l[(1u + d) * blksize],
                        rhouR = r[(1u + d) * blksize];
            const TData u     = Select(side, uL[d], uR[d]);
            const TData rhou  = Select(side, rhouL, rhouR);
            const TData mM    = rhoM * (u + (SM - un) * n[d]);
            flux[(1u + d) * blksize] =
                Select(left, rhouL * unL + pL * n[d],
                       Select(right, rhouR * unR + pR * n[d],
                              rhou * un + p * n[d] + S * (mM - rhou)));
        }
        flux[(1u + NDIM) * blksize] =
            Select(left, unL * (EL + pL),
                   Select(right, unR * (ER + pR), un * (E + p) + S * (EM - E)));
    }
};

} // namespace Nektar::detail
