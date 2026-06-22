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

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM>
struct HLLSolverKernel
{
    template <typename TScalar>
    NEK_DEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                      const size_t blksize, const TScalar *fwd,
                                      const TScalar *bwd, TScalar *flux)
    {
        constexpr unsigned int vec_width =
            (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                ? tinysimd::simd<TScalar>::width
                : 1;

        using std::abs;
        using std::sqrt;

        const TScalar oneHalf = 0.5;

        // Currently not directly vectorisable due to branching
        for (unsigned int k = 0; k < vec_width; k++)
        {
            // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment
            // exist.

            // Density
            const TScalar rhoL = fwd[0];
            const TScalar rhoR = bwd[0];

            // Velocities and kinetic energy terms
            TScalar uL[NDIM];
            TScalar uR[NDIM];
            TScalar qL2 = 0.0;
            TScalar qR2 = 0.0;
#pragma unroll
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const TScalar rhouL = fwd[(1u + d) * blksize];
                const TScalar rhouR = bwd[(1u + d) * blksize];

                uL[d] = rhouL / rhoL;
                uR[d] = rhouR / rhoR;

                qL2 += rhouL * uL[d];
                qR2 += rhouR * uR[d];
            }

            // Internal energy per unit mass
            const TScalar EL = fwd[(1u + NDIM) * blksize];
            const TScalar ER = bwd[(1u + NDIM) * blksize];
            const TScalar eL = (EL - oneHalf * qL2) / rhoL;
            const TScalar eR = (ER - oneHalf * qR2) / rhoR;

            // Pressure
            const TScalar pL = GetPressure(EoS, rhoL, eL);
            const TScalar pR = GetPressure(EoS, rhoR, eR);

            // Speed of sound
            const TScalar cL = GetSoundSpeed(EoS, rhoL, eL);
            const TScalar cR = GetSoundSpeed(EoS, rhoR, eR);

            // Enthalpy
            const TScalar hL = (EL + pL) / rhoL;
            const TScalar hR = (ER + pR) / rhoR;

            // Roe averages
            const TScalar srL  = sqrt(rhoL);
            const TScalar srR  = sqrt(rhoR);
            const TScalar srLR = srL + srR;

            TScalar uRoe[NDIM];
            TScalar URoe2 = 0;
#pragma unroll
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                uRoe[d] = (srL * uL[d] + srR * uR[d]) / srLR;
                URoe2 += uRoe[d] * uRoe[d];
            }

            const TScalar hRoe = (srL * hL + srR * hR) / srLR;

            const TScalar cRoe =
                GetRoeSoundSpeed(EoS, rhoL, pL, eL, hL, srL, rhoR, pR, eR, hR,
                                 srR, hRoe, URoe2, srLR);

            // Maximum wave speeds
            const TScalar SL = std::min(uL[0] - cL, uRoe[0] - cRoe);
            const TScalar SR = std::max(uR[0] + cR, uRoe[0] + cRoe);

            // HLL Riemann fluxes (positive case)
            if (SL >= 0)
            {
                flux[0]            = rhoL * uL[0];
                flux[1u * blksize] = rhoL * uL[0] * uL[0] + pL;
#pragma unroll
                for (unsigned int d = 1; d < NDIM; ++d)
                {
                    flux[(1u + d) * blksize] = rhoL * uL[0] * uL[d];
                }
                flux[(1u + NDIM) * blksize] = uL[0] * (EL + pL);
            }
            // HLL Riemann fluxes (negative case)
            else if (SR <= 0)
            {
                flux[0]            = rhoR * uR[0];
                flux[1u * blksize] = rhoR * uR[0] * uR[0] + pR;
#pragma unroll
                for (unsigned int d = 1; d < NDIM; ++d)
                {
                    flux[(1u + d) * blksize] = rhoR * uR[0] * uR[d];
                }
                flux[(1u + NDIM) * blksize] = uR[0] * (ER + pR);
            }
            // HLL Riemann fluxes (general case (SL < 0 | SR > 0)
            else
            {
                TScalar tmp1 = 1.0 / (SR - SL);
                TScalar tmp2 = SR * SL;
                flux[0]      = (SR * rhoL * uL[0] - SL * rhoR * uR[0] +
                           tmp2 * (rhoR - rhoL)) *
                          tmp1;
                flux[1u * blksize] = (SR * (rhoL * uL[0] * uL[0] + pL) -
                                      SL * (rhoR * uR[0] * uR[0] + pR) +
                                      tmp2 * (rhoR * uR[0] - rhoL * uL[0])) *
                                     tmp1;
#pragma unroll
                for (unsigned int d = 1; d < NDIM; ++d)
                {
                    flux[(1u + d) * blksize] =
                        (SR * rhoL * uL[0] * uL[d] - SL * rhoR * uR[0] * uR[d] +
                         tmp2 * (rhoR * uR[d] - rhoL * uL[d])) *
                        tmp1;
                }
                flux[(1u + NDIM) * blksize] =
                    (SR * uL[0] * (EL + pL) - SL * uR[0] * (ER + pR) +
                     tmp2 * (ER - EL)) *
                    tmp1;
            }

            fwd++;
            bwd++;
            flux++;
        }
    }
};

} // namespace Nektar::Operators::detail
