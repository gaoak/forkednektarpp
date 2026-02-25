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

#include "Operators/LoopExecution/LoopExecution.hpp"

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::Operators::detail
{

template <typename ExecSpace, unsigned int NDIM> struct RoeSolverKernel
{
    template <typename TScalar>
    NEK_FORCE_INLINE void operator()(const size_t blksize, const TScalar *fwd,
                                     const TScalar *bwd, TScalar *flux)
    {
        // Explicit vectorisation for AVX backend, vec_t =
        // tinysimd::simd<TScalar> for AVX, vec_t = TScalar otherwise.
        using vec_t =
            typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                                  TScalar>::type;
        const unsigned int vec_width =
            (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                ? tinysimd::simd<TScalar>::width
                : 1;

        // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
        const size_t groupsize = blksize / vec_width;
        const auto fwdvec      = reinterpret_cast<const vec_t *>(fwd);
        const auto bwdvec      = reinterpret_cast<const vec_t *>(bwd);
        auto fluxvec           = reinterpret_cast<vec_t *>(flux);
        Nektar::parallel_for<ExecSpace>(
            0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
                using std::sqrt;
                using std::abs;

                const vec_t oneHalf = 0.5;
                const vec_t gamma   = 1.4;

                // Density
                const vec_t rhoL = fwdvec[i];
                const vec_t rhoR = bwdvec[i];

                // Velocities and kinetic energy terms
                vec_t uL[NDIM];
                vec_t uR[NDIM];
                vec_t qL2 = 0.0;
                vec_t qR2 = 0.0;
#pragma unroll
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    const vec_t rhouL = fwdvec[(1u + d) * groupsize + i];
                    const vec_t rhouR = bwdvec[(1u + d) * groupsize + i];

                    uL[d] = rhouL / rhoL;
                    uR[d] = rhouR / rhoR;

                    qL2 += rhouL * uL[d];
                    qR2 += rhouR * uR[d];
                }

                // Internal energy per unit mass
                const vec_t EL = fwdvec[(1u + NDIM) * groupsize + i];
                const vec_t ER = bwdvec[(1u + NDIM) * groupsize + i];
                const vec_t eL = (EL - oneHalf * qL2) / rhoL;
                const vec_t eR = (ER - oneHalf * qR2) / rhoR;

                // Pressure
                const vec_t pL = GetPressure(rhoL, eL);
                const vec_t pR = GetPressure(rhoR, eR);

                // Enthalpy
                const vec_t hL = (EL + pL) / rhoL;
                const vec_t hR = (ER + pR) / rhoR;

                // Roe averages
                const vec_t srL  = sqrt(rhoL);
                const vec_t srR  = sqrt(rhoR);
                const vec_t srLR = srL + srR;

                vec_t uRoe[NDIM];
                vec_t URoe2 = 0;
#pragma unroll
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    uRoe[d] = (srL * uL[d] + srR * uR[d]) / srLR;
                    URoe2 += uRoe[d] * uRoe[d];
                }

                const vec_t hRoe = (srL * hL + srR * hR) / srLR;

                const vec_t cRoe =
                    GetRoeSoundSpeed(rhoL, pL, eL, hL, srL, rhoR, pR, eR, hR,
                                     srR, hRoe, URoe2, srLR);

                // Compute eigenvectors (equation 11.59).
                vec_t k[NDIM + 2][NDIM + 2];
                k[0][0] = 1.0;
                k[0][1] = uRoe[0] - cRoe;
                if constexpr (NDIM > 1)
                {
                    k[0][2] = uRoe[1];
                }
                if constexpr (NDIM > 2)
                {
                    k[0][3] = uRoe[2];
                }
                k[0][NDIM + 1] = hRoe - uRoe[0] * cRoe;

                k[1][0] = 1.0;
                k[1][1] = uRoe[0];
                if constexpr (NDIM > 1)
                {
                    k[1][2] = uRoe[1];
                }
                if constexpr (NDIM > 2)
                {
                    k[1][3] = uRoe[2];
                }
                k[1][NDIM + 1] = 0.5 * URoe2;

                if constexpr (NDIM > 1)
                {
                    k[2][0] = 0.0;
                    k[2][1] = 0.0;
                    k[2][2] = 1.0;
                    if constexpr (NDIM > 2)
                    {
                        k[2][3] = 0.0;
                    }
                    k[2][NDIM + 1] = uRoe[1];
                }

                if constexpr (NDIM > 2)
                {
                    k[3][0]        = 0.0;
                    k[3][1]        = 0.0;
                    k[3][2]        = 0.0;
                    k[3][3]        = 1.0;
                    k[3][NDIM + 1] = uRoe[2];
                }

                k[NDIM + 1][0] = 1.0;
                k[NDIM + 1][1] = uRoe[0] + cRoe;
                if constexpr (NDIM > 1)
                {
                    k[NDIM + 1][2] = uRoe[1];
                }
                if constexpr (NDIM > 2)
                {
                    k[NDIM + 1][3] = uRoe[2];
                }
                k[NDIM + 1][NDIM + 1] = hRoe + uRoe[0] * cRoe;

                // Calculate jumps \Delta u_i (defined preceding
                // equation 11.67).
                vec_t jump[NDIM + 2];
                jump[0] = rhoR - rhoL;
                jump[1] = rhoR * uR[0] - rhoL * uL[0];
                if constexpr (NDIM > 1)
                {
                    jump[2] = rhoR * uR[1] - rhoL * uL[1];
                }
                if constexpr (NDIM > 2)
                {
                    jump[3] = rhoR * uR[2] - rhoL * uL[2];
                }
                jump[NDIM + 1] = ER - EL;

                // Define \Delta u_5 (equation 11.70).
                vec_t jumpbar = jump[NDIM + 1];
                if constexpr (NDIM > 1)
                {
                    jumpbar -= (jump[2] - uRoe[1] * jump[0]) * uRoe[1];
                }
                if constexpr (NDIM > 2)
                {
                    jumpbar -= (jump[3] - uRoe[2] * jump[0]) * uRoe[2];
                }

                // Compute wave amplitudes (equations 11.68, 11.69).
                vec_t alpha[NDIM + 2];
                alpha[1] = (gamma - 1.0) *
                           (jump[0] * (hRoe - uRoe[0] * uRoe[0]) +
                            uRoe[0] * jump[1] - jumpbar) /
                           (cRoe * cRoe);
                alpha[0] =
                    (jump[0] * (uRoe[0] + cRoe) - jump[1] - cRoe * alpha[1]) /
                    (2.0 * cRoe);
                if constexpr (NDIM > 1)
                {
                    alpha[2] = jump[2] - uRoe[1] * jump[0];
                }
                if constexpr (NDIM > 2)
                {
                    alpha[3] = jump[3] - uRoe[2] * jump[0];
                }
                alpha[NDIM + 1] = jump[0] - (alpha[0] + alpha[1]);

                // Compute average of left and right fluxes needed for
                // equation 11.29.
                vec_t fluxtmp[NDIM + 2];
                fluxtmp[0] = oneHalf * (rhoL * uL[0] + rhoR * uR[0]);
                fluxtmp[1] = oneHalf * (rhoL * uL[0] * uL[0] + pL +
                                        rhoR * uR[0] * uR[0] + pR);
                if constexpr (NDIM > 1)
                {
                    fluxtmp[2] =
                        oneHalf * (rhoL * uL[0] * uL[1] + rhoR * uR[0] * uR[1]);
                }
                if constexpr (NDIM > 2)
                {
                    fluxtmp[3] =
                        oneHalf * (rhoL * uL[0] * uL[2] + rhoR * uR[0] * uR[2]);
                }
                fluxtmp[1u + NDIM] =
                    oneHalf * (uL[0] * (EL + pL) + uR[0] * (ER + pR));

                // Compute eigenvalues \lambda_i (equation 11.58).
                const vec_t uRoeAbs = abs(uRoe[0]);
                vec_t lambda[NDIM + 2];
                lambda[0] = abs(uRoe[0] - cRoe);
                lambda[1] = uRoeAbs;
                if constexpr (NDIM > 1)
                {
                    lambda[2] = uRoeAbs;
                }
                if constexpr (NDIM > 2)
                {
                    lambda[3] = uRoeAbs;
                }
                lambda[NDIM + 1] = abs(uRoe[0] + cRoe);

                for (unsigned int i = 0; i < NDIM + 2; ++i)
                {
                    const vec_t tmp = oneHalf * alpha[i] * lambda[i];

                    fluxtmp[0] -= tmp * k[i][0];
                    fluxtmp[1] -= tmp * k[i][1];
                    if constexpr (NDIM > 1)
                    {
                        fluxtmp[2] -= tmp * k[i][2];
                    }
                    if constexpr (NDIM > 2)
                    {
                        fluxtmp[3] -= tmp * k[i][3];
                    }
                    fluxtmp[1u + NDIM] -= tmp * k[i][1u + NDIM];
                }

                // Save flux
                fluxvec[i]                  = fluxtmp[0];
                fluxvec[1u * groupsize + i] = fluxtmp[1];
                if constexpr (NDIM > 1)
                {
                    fluxvec[2u * groupsize + i] = fluxtmp[2];
                }
                if constexpr (NDIM > 2)
                {
                    fluxvec[3u * groupsize + i] = fluxtmp[3];
                }
                fluxvec[(1u + NDIM) * groupsize + i] = fluxtmp[1u + NDIM];
            });
    }
};

} // namespace Nektar::Operators::detail
