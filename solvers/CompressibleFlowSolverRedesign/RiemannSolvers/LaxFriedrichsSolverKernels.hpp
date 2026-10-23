///////////////////////////////////////////////////////////////////////////////
//
// File: LaxFriedrichsSolverKernels.hpp
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
 * @brief The Lax-Friedrichs flux through the trace normal, in Cartesian
 * components (as the legacy LaxFriedrichsSolver): the central flux of the two
 * states through @p normal less the jump scaled by the largest normal wave
 * speed of the Roe-averaged state.
 */
template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM>
struct LaxFriedrichsSolverKernel
{
    /**
     * @param blksize  Points in the block; component c of the states, the
     *                 normal and the flux lies at offset `c * blksize`.
     * @param normal   The unit trace normal, @p NDIM components.
     */
    template <typename TScalar>
    NEK_HOSTDEVICE_INLINE void operator()(const EqnOfStParams EoS,
                                          const size_t blksize,
                                          const TScalar *fwd,
                                          const TScalar *bwd,
                                          const TScalar *normal, TScalar *flux)
    {
        using std::abs;
        using std::sqrt;

        // Explicit vectorisation for AVX backend, vec_t =
        // tinysimd::simd<TScalar> for AVX, vec_t = TScalar otherwise.
        using vec_t =
            typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                                  TScalar>::type;
        constexpr unsigned int vec_width =
            (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                ? tinysimd::simd<TScalar>::width
                : 1;

        const vec_t oneHalf = 0.5;

        // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
        const size_t groupsize = blksize / vec_width;
        const auto fwdvec      = reinterpret_cast<const vec_t *>(fwd);
        const auto bwdvec      = reinterpret_cast<const vec_t *>(bwd);
        const auto nvec        = reinterpret_cast<const vec_t *>(normal);
        auto fluxvec           = reinterpret_cast<vec_t *>(flux);

        const vec_t rhoL = fwdvec[0];
        const vec_t rhoR = bwdvec[0];

        // Velocities, normal velocities and kinetic energy terms.
        vec_t n[NDIM], mL[NDIM], mR[NDIM], uL[NDIM], uR[NDIM];
        vec_t unL = 0.0, unR = 0.0, qL2 = 0.0, qR2 = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            n[d]  = nvec[d * groupsize];
            mL[d] = fwdvec[(1u + d) * groupsize];
            mR[d] = bwdvec[(1u + d) * groupsize];
            uL[d] = mL[d] / rhoL;
            uR[d] = mR[d] / rhoR;
            unL += uL[d] * n[d];
            unR += uR[d] * n[d];
            qL2 += mL[d] * uL[d];
            qR2 += mR[d] * uR[d];
        }

        // Internal energy per unit mass, pressure and enthalpy.
        const vec_t EL = fwdvec[(1u + NDIM) * groupsize];
        const vec_t ER = bwdvec[(1u + NDIM) * groupsize];
        const vec_t eL = (EL - oneHalf * qL2) / rhoL;
        const vec_t eR = (ER - oneHalf * qR2) / rhoR;
        const vec_t pL = GetPressure(EoS, rhoL, eL);
        const vec_t pR = GetPressure(EoS, rhoR, eR);
        const vec_t hL = (EL + pL) / rhoL;
        const vec_t hR = (ER + pR) / rhoR;

        // Roe averages.
        const vec_t srL  = sqrt(rhoL);
        const vec_t srR  = sqrt(rhoR);
        const vec_t srLR = srL + srR;
        vec_t unRoe = 0.0, URoe2 = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const vec_t uRoe = (srL * uL[d] + srR * uR[d]) / srLR;
            unRoe += uRoe * n[d];
            URoe2 += uRoe * uRoe;
        }
        const vec_t hRoe = (srL * hL + srR * hR) / srLR;
        const vec_t cRoe = GetRoeSoundSpeed(EoS, rhoL, pL, eL, hL, srL, rhoR,
                                            pR, eR, hR, srR, hRoe, URoe2, srLR);

        // Largest wave speed through the normal.
        const vec_t a = abs(unRoe) + cRoe;

        // Mass, momentum and energy: the central flux through n less the
        // dissipation of the jump.
        fluxvec[0] = oneHalf * (rhoL * unL + rhoR * unR - a * (rhoR - rhoL));
#pragma unroll
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            fluxvec[(1u + d) * groupsize] =
                oneHalf * (mL[d] * unL + mR[d] * unR + (pL + pR) * n[d] -
                           a * (mR[d] - mL[d]));
        }
        fluxvec[(1u + NDIM) * groupsize] =
            oneHalf * (unL * (EL + pL) + unR * (ER + pR) - a * (ER - EL));
    }
};

} // namespace Nektar::detail
