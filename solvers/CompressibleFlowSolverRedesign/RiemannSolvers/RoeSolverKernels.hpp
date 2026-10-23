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

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::detail
{

/**
 * @brief The two sides' velocities, internal energies, pressures and
 * enthalpies at one point, and the Roe averages of the velocity, the
 * enthalpy and the sound speed: what every form of the Roe flux starts from.
 */
template <unsigned int NDIM, typename vec_t> struct RoeAverageState
{
    vec_t invRhoL, invRhoR;
    vec_t uL[NDIM], uR[NDIM];
    vec_t EL, ER, eL, eR, pL, pR, hL, hR;
    vec_t srL, srR, srLR;
    vec_t uRoe[NDIM];
    vec_t URoe2, hRoe, cRoe;
};

/**
 * @brief Fill @p s from the two conserved states @p l and @p r, whose
 * components lie @p stride vectors apart.
 */
template <unsigned int NDIM, typename EqnOfStParams, typename vec_t>
NEK_HOSTDEVICE_INLINE static void RoeAverages(const EqnOfStParams &EoS,
                                              const vec_t *l, const vec_t *r,
                                              const size_t stride,
                                              RoeAverageState<NDIM, vec_t> &s)
{
    using std::sqrt;
    const vec_t half = vec_t(0.5);

    const vec_t rhoL = l[0], rhoR = r[0];
    // One reciprocal per density and per Roe denominator in place of a
    // division at every use.
    s.invRhoL = vec_t(1.0) / rhoL;
    s.invRhoR = vec_t(1.0) / rhoR;
    vec_t qL2 = vec_t(0.0), qR2 = vec_t(0.0);
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        const vec_t rhouL = l[(1u + d) * stride];
        const vec_t rhouR = r[(1u + d) * stride];
        s.uL[d]           = rhouL * s.invRhoL;
        s.uR[d]           = rhouR * s.invRhoR;
        qL2 += rhouL * s.uL[d];
        qR2 += rhouR * s.uR[d];
    }
    s.EL = l[(1u + NDIM) * stride];
    s.ER = r[(1u + NDIM) * stride];
    s.eL = (s.EL - half * qL2) * s.invRhoL;
    s.eR = (s.ER - half * qR2) * s.invRhoR;
    s.pL = GetPressure(EoS, rhoL, s.eL);
    s.pR = GetPressure(EoS, rhoR, s.eR);
    s.hL = (s.EL + s.pL) * s.invRhoL;
    s.hR = (s.ER + s.pR) * s.invRhoR;

    s.srL               = sqrt(rhoL);
    s.srR               = sqrt(rhoR);
    s.srLR              = s.srL + s.srR;
    const vec_t invSrLR = vec_t(1.0) / s.srLR;
    s.URoe2             = vec_t(0.0);
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        s.uRoe[d] = (s.srL * s.uL[d] + s.srR * s.uR[d]) * invSrLR;
        s.URoe2 += s.uRoe[d] * s.uRoe[d];
    }
    s.hRoe = (s.srL * s.hL + s.srR * s.hR) * invSrLR;
    s.cRoe = GetRoeSoundSpeed(EoS, rhoL, s.pL, s.eL, s.hL, s.srL, rhoR, s.pR,
                              s.eR, s.hR, s.srR, s.hRoe, s.URoe2, s.srLR);
}

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
     * @param blksize  Points in the block; component c of the states, the
     *                 normal and the flux lies at offset `c * blksize`.
     * @param normal   The unit trace normal, pointing from @p fwd to @p bwd,
     *                 @p NDIM components.
     */
    template <typename TScalar>
    NEK_HOSTDEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                          const size_t blksize,
                                          const TScalar *fwd,
                                          const TScalar *bwd,
                                          const TScalar *normal, TScalar *flux)
    {
        using std::abs;

        // Explicit vectorisation for AVX backend, vec_t =
        // tinysimd::simd<TScalar> for AVX, vec_t = TScalar otherwise.
        using vec_t =
            typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                                  TScalar>::type;
        constexpr unsigned int vec_width =
            (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                ? tinysimd::simd<TScalar>::width
                : 1;

        const vec_t half = vec_t(0.5);

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
        const vec_t hRoe = s.hRoe, cRoe = s.cRoe, URoe2 = s.URoe2;

        vec_t n[NDIM];
        vec_t unL = vec_t(0.0), unR = vec_t(0.0), unRoe = vec_t(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            n[d] = nvec[d * gs];
            unL += n[d] * uL[d];
            unR += n[d] * uR[d];
            unRoe += n[d] * uRoe[d];
        }
        const vec_t invC = vec_t(1.0) / cRoe;

        // Jumps: density, momentum (vector and normal), energy; and the shear
        // vector w = [rho u] - u~ [rho] with its normal part removed.
        const vec_t dRho = rhoR - rhoL;
        vec_t dm[NDIM], w[NDIM];
        vec_t dmn = vec_t(0.0), wn = vec_t(0.0), wDotU = vec_t(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            dm[d] = r[(1u + d) * gs] - l[(1u + d) * gs];
            w[d]  = dm[d] - uRoe[d] * dRho;
            dmn += n[d] * dm[d];
            wn += n[d] * w[d];
            wDotU += w[d] * uRoe[d];
        }
        // The energy jump less the tangential kinetic part: jumpbar = [E] -
        // sum_t w_t u~_t.
        const vec_t jumpbar = (ER - EL) - (wDotU - wn * unRoe);

        const vec_t gamma = vec_t(EoS.gamma());
        const vec_t alpha1 =
            (gamma - vec_t(1.0)) *
            (dRho * (hRoe - unRoe * unRoe) + unRoe * dmn - jumpbar) *
            (invC * invC);
        const vec_t alpha0 =
            (dRho * (unRoe + cRoe) - dmn - cRoe * alpha1) * (half * invC);
        const vec_t alpha4 = dRho - (alpha0 + alpha1);

        const vec_t lam0 = abs(unRoe - cRoe), lam1 = abs(unRoe),
                    lam4 = abs(unRoe + cRoe);
        const vec_t t0 = half * alpha0 * lam0, t1 = half * alpha1 * lam1,
                    t4 = half * alpha4 * lam4, ts = half * lam1;

        // Central flux through n, then the dissipation of the wave groups.
        f[0] = half * (rhoL * unL + rhoR * unR) - (t0 + t1 + t4);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            f[(1u + d) * gs] =
                half * (l[(1u + d) * gs] * unL + r[(1u + d) * gs] * unR +
                        (pL + pR) * n[d]) -
                (t0 * (uRoe[d] - cRoe * n[d]) + t1 * uRoe[d] +
                 t4 * (uRoe[d] + cRoe * n[d]) + ts * (w[d] - wn * n[d]));
        }
        f[(1u + NDIM) * gs] =
            half * (unL * (EL + pL) + unR * (ER + pR)) -
            (t0 * (hRoe - unRoe * cRoe) + t1 * half * URoe2 +
             t4 * (hRoe + unRoe * cRoe) + ts * (wDotU - wn * unRoe));
    }
};

} // namespace Nektar::detail