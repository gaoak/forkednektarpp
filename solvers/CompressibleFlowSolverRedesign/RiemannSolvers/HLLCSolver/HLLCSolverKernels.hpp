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

#include "Operators/LoopExecution/LoopExecution.hpp"

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::Operators::detail
{

template <typename ExecSpace, unsigned int NDIM> struct HLLCSolverKernel
{
    template <typename TScalar>
    NEK_DEVICE_INLINE void operator()(const size_t blksize, const TScalar *fwd,
                                      const TScalar *bwd, TScalar *flux)
    {
        // Explicit vectorisation for AVX backend, vec_t =
        // tinysimd::simd<TScalar> for AVX, vec_t = TScalar otherwise.
        // using vec_t =
        //    typename data_type_if<std::is_same_v<ExecSpace,
        //    NektarSpaces::AVX>,
        //                          TScalar>::type;
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
            const TScalar pL = GetPressure(rhoL, eL);
            const TScalar pR = GetPressure(rhoR, eR);

            // Speed of sound
            const TScalar cL = GetSoundSpeed(rhoL, eL);
            const TScalar cR = GetSoundSpeed(rhoR, eR);

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
                GetRoeSoundSpeed(rhoL, pL, eL, hL, srL, rhoR, pR, eR, hR, srR,
                                 hRoe, URoe2, srLR);

            // Maximum wave speeds
            const TScalar SL = std::min(uL[0] - cL, uRoe[0] - cRoe);
            const TScalar SR = std::max(uR[0] + cR, uRoe[0] + cRoe);

            // HLLC Riemann fluxes (positive case)
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
            // HLLC Riemann fluxes (negative case)
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
                TScalar rhouML[NDIM];
                TScalar SM = (pR - pL + rhoL * uL[0] * (SL - uL[0]) -
                              rhoR * uR[0] * (SR - uR[0])) /
                             (rhoL * (SL - uL[0]) - rhoR * (SR - uR[0]));
                TScalar rhoML = rhoL * (SL - uL[0]) / (SL - SM);
                rhouML[0]     = rhoML * SM;
#pragma unroll
                for (unsigned int d = 1; d < NDIM; ++d)
                {
                    rhouML[d] = rhoML * uL[d];
                }
                TScalar EML =
                    rhoML * (EL / rhoL +
                             (SM - uL[0]) * (SM + pL / (rhoL * (SL - uL[0]))));

                TScalar rhouMR[NDIM];
                TScalar rhoMR = rhoR * (SR - uR[0]) / (SR - SM);
                rhouMR[0]     = rhoMR * SM;
#pragma unroll
                for (unsigned int d = 1; d < NDIM; ++d)
                {
                    rhouMR[d] = rhoMR * uR[d];
                }
                TScalar EMR =
                    rhoMR * (ER / rhoR +
                             (SM - uR[0]) * (SM + pR / (rhoR * (SR - uR[0]))));

                // Conditional assignment
                const bool cond        = SL < 0.0 && SM >= 0.0;
                const TScalar &rhoUp   = (cond) ? rhoL : rhoR;
                const TScalar &pUp     = (cond) ? pL : pR;
                const TScalar *uUp     = (cond) ? uL : uR;
                const TScalar &EUp     = (cond) ? EL : ER;
                const TScalar &rhoMUp  = (cond) ? rhoML : rhoMR;
                const TScalar *rhouMUp = (cond) ? rhouML : rhouMR;
                const TScalar &SUp     = (cond) ? SL : SR;
                const TScalar &EMUp    = (cond) ? EML : EMR;

                // Compute flux
                flux[0]            = rhoUp * uUp[0] + SUp * (rhoMUp - rhoUp);
                flux[1u * blksize] = rhoUp * uUp[0] * uUp[0] + pUp +
                                     SL * (rhouMUp[0] - rhoUp * uUp[0]);
#pragma unroll
                for (unsigned int d = 1; d < NDIM; ++d)
                {
                    flux[(1u + d) * blksize] =
                        rhoUp * uUp[0] * uUp[d] +
                        SUp * (rhouMUp[d] - rhoUp * uUp[d]);
                }
                flux[(1u + NDIM) * blksize] =
                    uUp[0] * (EUp + pUp) + SUp * (EMUp - EUp);
            }

            fwd++;
            bwd++;
            flux++;
        }
    }
};

} // namespace Nektar::Operators::detail
