///////////////////////////////////////////////////////////////////////////////
//
// File: RoeSolverKernels.hpp
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

#include <CompressibleFlowSolverRedesign/RiemannSolvers/RoeAverage.hpp>

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::detail
{

/**
 * @brief The Roe flux through the trace normal, in Cartesian components (as
 * the legacy RoeSolver).
 *
 * The averages, wave strengths and eigenvectors of the Roe linearisation,
 * written with the normal n alone: the normal components are dot products
 * with n, and the shear waves, whose eigenvectors are the tangent
 * directions, enter only through their sum, the tangential projection
 * w - (n . w) n of w = [rho u] - u~ [rho]. In one dimension that projection
 * vanishes.
 */
template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM>
struct RoeSolverKernel
{
    /**
     * @param blksize  Component stride, in TData (a scalar or a SIMD vector
     *                 of points): component c of the states, the normal and
     *                 the flux lies at offset `c * blksize`.
     * @param normal   The unit trace normal, pointing from @p fwd to @p bwd,
     *                 @p NDIM components.
     */
    template <typename TData>
    NEK_HOSTDEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                          const size_t blksize,
                                          const TData *fwd, const TData *bwd,
                                          const TData *normal, TData *flux)
    {
        using std::abs;

        const TData half = TData(0.5);

        // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
        const auto l = fwd;
        const auto r = bwd;

        // The two sides' primitive quantities and the Roe averages.
        RoeAverageState<NDIM, TData> s;
        RoeAverages<NDIM>(EoS, l, r, blksize, s);
        const TData rhoL = l[0], rhoR = r[0];
        const TData EL = s.EL, ER = s.ER, pL = s.pL, pR = s.pR;
        const auto &uL = s.uL, &uR = s.uR, &uRoe = s.uRoe;
        const TData hRoe = s.hRoe, cRoe = s.cRoe, URoe2 = s.URoe2;

        TData n[NDIM];
        TData unL = TData(0.0), unR = TData(0.0), unRoe = TData(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            n[d] = normal[d * blksize];
            unL += n[d] * uL[d];
            unR += n[d] * uR[d];
            unRoe += n[d] * uRoe[d];
        }
        const TData invC = TData(1.0) / cRoe;

        // Jumps: density, momentum (vector and normal), energy; and the shear
        // vector w = [rho u] - u~ [rho] with its normal part removed.
        const TData dRho = rhoR - rhoL;
        TData dm[NDIM], w[NDIM];
        TData dmn = TData(0.0), wn = TData(0.0), wDotU = TData(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            dm[d] = r[(1u + d) * blksize] - l[(1u + d) * blksize];
            w[d]  = dm[d] - uRoe[d] * dRho;
            dmn += n[d] * dm[d];
            wn += n[d] * w[d];
            wDotU += w[d] * uRoe[d];
        }
        // The energy jump less the tangential kinetic part: jumpbar = [E] -
        // sum_t w_t u~_t.
        const TData jumpbar = (ER - EL) - (wDotU - wn * unRoe);

        const TData gamma = TData(EoS.gamma());
        const TData alpha1 =
            (gamma - TData(1.0)) *
            (dRho * (hRoe - unRoe * unRoe) + unRoe * dmn - jumpbar) *
            (invC * invC);
        const TData alpha0 =
            (dRho * (unRoe + cRoe) - dmn - cRoe * alpha1) * (half * invC);
        const TData alpha4 = dRho - (alpha0 + alpha1);

        const TData lam0 = abs(unRoe - cRoe), lam1 = abs(unRoe),
                    lam4 = abs(unRoe + cRoe);
        const TData t0 = half * alpha0 * lam0, t1 = half * alpha1 * lam1,
                    t4 = half * alpha4 * lam4, ts = half * lam1;

        // Central flux through n, then the dissipation of the wave groups.
        flux[0] = half * (rhoL * unL + rhoR * unR) - (t0 + t1 + t4);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            flux[(1u + d) * blksize] =
                half * (l[(1u + d) * blksize] * unL +
                        r[(1u + d) * blksize] * unR + (pL + pR) * n[d]) -
                (t0 * (uRoe[d] - cRoe * n[d]) + t1 * uRoe[d] +
                 t4 * (uRoe[d] + cRoe * n[d]) + ts * (w[d] - wn * n[d]));
        }
        flux[(1u + NDIM) * blksize] =
            half * (unL * (EL + pL) + unR * (ER + pR)) -
            (t0 * (hRoe - unRoe * cRoe) + t1 * half * URoe2 +
             t4 * (hRoe + unRoe * cRoe) + ts * (wDotU - wn * unRoe));
    }
};

} // namespace Nektar::detail
