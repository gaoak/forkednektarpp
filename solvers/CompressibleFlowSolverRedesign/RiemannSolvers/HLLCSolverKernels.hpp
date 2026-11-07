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

#include "RiemannSolvers/RoeAverage.hpp"

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
    /// stride @p blksize.
    template <typename TScalar>
    NEK_HOSTDEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                          const size_t blksize,
                                          const TScalar *fwd,
                                          const TScalar *bwd,
                                          const TScalar *normal, TScalar *flux)
    {
        using std::max;
        using std::min;

        // Explicit vectorisation for AVX backend, vec_t =
        // tinysimd::simd<TScalar> for AVX, vec_t = TScalar otherwise.
        using vec_t =
            typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                                  TScalar>::type;
        constexpr unsigned int vec_width =
            (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                ? tinysimd::simd<TScalar>::width
                : 1;

        // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
        const size_t gs = blksize / vec_width;
        const auto l    = reinterpret_cast<const vec_t *>(fwd);
        const auto r    = reinterpret_cast<const vec_t *>(bwd);
        const auto nvec = reinterpret_cast<const vec_t *>(normal);
        auto f          = reinterpret_cast<vec_t *>(flux);

        // The two sides' primitive quantities and the Roe averages.
        RoeAverageState<NDIM, vec_t> s;
        RoeAverages<NDIM>(EoS, l, r, gs, s);
        const vec_t rhoL = l[0], rhoR = r[0];
        const vec_t EL = s.EL, ER = s.ER, pL = s.pL, pR = s.pR;
        const auto &uL = s.uL, &uR = s.uR, &uRoe = s.uRoe;
        const vec_t cRoe = s.cRoe;
        const vec_t cL   = GetSoundSpeed(EoS, rhoL, s.eL);
        const vec_t cR   = GetSoundSpeed(EoS, rhoR, s.eR);

        vec_t n[NDIM];
        vec_t unL = vec_t(0.0), unR = vec_t(0.0), unRoe = vec_t(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            n[d] = nvec[d * gs];
            unL += n[d] * uL[d];
            unR += n[d] * uR[d];
            unRoe += n[d] * uRoe[d];
        }

        // Mass flux through the normal from each side, and the largest wave
        // speeds.
        const vec_t mnL = rhoL * unL;
        const vec_t mnR = rhoR * unR;
        const vec_t SL  = min(unL - cL, unRoe - cRoe);
        const vec_t SR  = max(unR + cR, unRoe + cRoe);

        // HLLC: the star state on the side the contact wave SM leaves
        // behind. Only the normal velocity changes there, to SM; the
        // tangential velocity is advected unaltered, which in Cartesian
        // components is u + (SM - u.n) n.
        //
        // Every point, or lane of a vector, forms the one-sided fluxes and the
        // star flux, and takes the one its wave speeds select: the left flux
        // where SL >= 0, the right where SR <= 0, the star flux otherwise. A
        // flux not taken may hold any value.
        const vec_t SM = (pR - pL + mnL * (SL - unL) - mnR * (SR - unR)) /
                         (rhoL * (SL - unL) - rhoR * (SR - unR));
        const auto left  = SL >= vec_t(0.0);
        const auto right = SR <= vec_t(0.0);
        const auto side  = SM >= vec_t(0.0);
        const vec_t rho  = Select(side, rhoL, rhoR);
        const vec_t un   = Select(side, unL, unR);
        const vec_t p    = Select(side, pL, pR);
        const vec_t E    = Select(side, EL, ER);
        const vec_t S    = Select(side, SL, SR);
        const vec_t mn   = rho * un;
        const vec_t rhoM = rho * (S - un) / (S - SM);
        const vec_t EM =
            rhoM * (E / rho + (SM - un) * (SM + p / (rho * (S - un))));

        f[0] = Select(left, mnL, Select(right, mnR, mn + S * (rhoM - rho)));
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const vec_t rhouL = l[(1u + d) * gs], rhouR = r[(1u + d) * gs];
            const vec_t u    = Select(side, uL[d], uR[d]);
            const vec_t rhou = Select(side, rhouL, rhouR);
            const vec_t mM   = rhoM * (u + (SM - un) * n[d]);
            f[(1u + d) * gs] =
                Select(left, rhouL * unL + pL * n[d],
                       Select(right, rhouR * unR + pR * n[d],
                              rhou * un + p * n[d] + S * (mM - rhou)));
        }
        f[(1u + NDIM) * gs] =
            Select(left, unL * (EL + pL),
                   Select(right, unR * (ER + pR), un * (E + p) + S * (EM - E)));
    }
};

} // namespace Nektar::detail
