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

        // The largest wave speeds, clamped at zero: SL- = min(SL, 0) and
        // SR+ = max(SR, 0). With them the three cases of HLL - the left
        // state's flux where SL >= 0, the right's where SR <= 0, and the HLL
        // flux between - are the one expression
        //   F = (SR+ FL - SL- FR + SR+ SL- (UR - UL)) / (SR+ - SL-),
        // so every lane takes it without a branch. SR+ - SL- is zero only
        // when both speeds are, where the flux is taken as zero.
        const vec_t SLm  = min(min(unL - cL, unRoe - cRoe), vec_t(0.0));
        const vec_t SRp  = max(max(unR + cR, unRoe + cRoe), vec_t(0.0));
        const vec_t invS = SafeDivide(vec_t(1.0), SRp - SLm);
        const vec_t SLR  = SRp * SLm;

        // Jumps: density and momentum.
        const vec_t dRho = rhoR - rhoL;

        f[0] = (SRp * rhoL * unL - SLm * rhoR * unR + SLR * dRho) * invS;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const vec_t rhouL = l[(1u + d) * gs], rhouR = r[(1u + d) * gs];
            const vec_t dm   = rhouR - rhouL;
            f[(1u + d) * gs] = (SRp * (rhouL * unL + pL * n[d]) -
                                SLm * (rhouR * unR + pR * n[d]) + SLR * dm) *
                               invS;
        }
        f[(1u + NDIM) * gs] =
            (SRp * unL * (EL + pL) - SLm * unR * (ER + pR) + SLR * (ER - EL)) *
            invS;
    }
};

} // namespace Nektar::detail
