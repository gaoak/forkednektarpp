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
    template <typename TData>
    NEK_FORCE_INLINE void operator()(const size_t blksize, const TData *fwd,
                                     const TData *bwd, TData *flux)
    {
        // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
        Nektar::parallel_for<ExecSpace>(
            0u, blksize, NEKTAR_LAMBDA(const size_t i) {
                using std::sqrt;
                using std::abs;

                const TData oneHalf = 0.5;

                // Density
                const TData rhoL = fwd[i];
                const TData rhoR = bwd[i];

                // Velocities and kinetic energy terms
                TData uL[NDIM];
                TData uR[NDIM];
                TData qL2 = 0.0;
                TData qR2 = 0.0;
#pragma unroll
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    const TData rhouL = fwd[(1u + d) * blksize + i];
                    const TData rhouR = bwd[(1u + d) * blksize + i];

                    uL[d] = rhouL / rhoL;
                    uR[d] = rhouR / rhoR;

                    qL2 += rhouL * uL[d];
                    qR2 += rhouR * uR[d];
                }

                // Internal energy per unit mass
                const TData EL = fwd[(1u + NDIM) * blksize + i];
                const TData ER = bwd[(1u + NDIM) * blksize + i];
                const TData eL = (EL - oneHalf * qL2) / rhoL;
                const TData eR = (ER - oneHalf * qR2) / rhoR;

                // Pressure
                const TData pL = GetPressure(rhoL, eL);
                const TData pR = GetPressure(rhoR, eR);

                // Speed of sound
                const TData cL = GetSoundSpeed(rhoL, eL);
                const TData cR = GetSoundSpeed(rhoR, eR);

                // Enthalpy
                const TData hL = (EL + pL) / rhoL;
                const TData hR = (ER + pR) / rhoR;

                // Roe averages
                const TData srL  = sqrt(rhoL);
                const TData srR  = sqrt(rhoR);
                const TData srLR = srL + srR;

                TData uRoe[NDIM];
                TData URoe2 = 0;
#pragma unroll
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    uRoe[d] = (srL * uL[d] + srR * uR[d]) / srLR;
                    URoe2 += uRoe[d] * uRoe[d];
                }

                const TData hRoe = (srL * hL + srR * hR) / srLR;

                const TData cRoe =
                    GetRoeSoundSpeed(rhoL, pL, eL, hL, srL, rhoR, pR, eR, hR,
                                     srR, hRoe, URoe2, srLR);

                // Maximum wave speeds
                const TData SL = std::min(uL[0] - cL, uRoe[0] - cRoe);
                const TData SR = std::max(uR[0] + cR, uRoe[0] + cRoe);

                // HLLC Riemann fluxes (positive case)
                if (SL >= 0)
                {
                    flux[i]                = rhoL * uL[0];
                    flux[1u * blksize + i] = rhoL * uL[0] * uL[0] + pL;
#pragma unroll
                    for (unsigned int d = 1; d < NDIM; ++d)
                    {
                        flux[(1u + d) * blksize + i] = rhoL * uL[0] * uL[d];
                    }
                    flux[(1u + NDIM) * blksize + i] = uL[0] * (EL + pL);
                }
                // HLLC Riemann fluxes (negative case)
                else if (SR <= 0)
                {
                    flux[i]                = rhoR * uR[0];
                    flux[1u * blksize + i] = rhoR * uR[0] * uR[0] + pR;
#pragma unroll
                    for (unsigned int d = 1; d < NDIM; ++d)
                    {
                        flux[(1u + d) * blksize + i] = rhoR * uR[0] * uR[d];
                    }
                    flux[(1u + NDIM) * blksize + i] = uR[0] * (ER + pR);
                }
                // HLL Riemann fluxes (general case (SL < 0 | SR > 0)
                else
                {
                    TData rhouML[NDIM];
                    TData SM = (pR - pL + rhoL * uL[0] * (SL - uL[0]) -
                                rhoR * uR[0] * (SR - uR[0])) /
                               (rhoL * (SL - uL[0]) - rhoR * (SR - uR[0]));
                    TData rhoML = rhoL * (SL - uL[0]) / (SL - SM);
                    rhouML[0]   = rhoML * SM;
#pragma unroll
                    for (unsigned int d = 1; d < NDIM; ++d)
                    {
                        rhouML[d] = rhoML * uL[d];
                    }
                    TData EML = rhoML * (EL / rhoL +
                                         (SM - uL[0]) *
                                             (SM + pL / (rhoL * (SL - uL[0]))));

                    TData rhouMR[NDIM];
                    TData rhoMR = rhoR * (SR - uR[0]) / (SR - SM);
                    rhouMR[0]   = rhoMR * SM;
#pragma unroll
                    for (unsigned int d = 1; d < NDIM; ++d)
                    {
                        rhouMR[d] = rhoMR * uR[d];
                    }
                    TData EMR = rhoMR * (ER / rhoR +
                                         (SM - uR[0]) *
                                             (SM + pR / (rhoR * (SR - uR[0]))));

                    // Conditional assignment
                    const bool cond      = SL < 0.0 && SM >= 0.0;
                    const TData &rhoUp   = (cond) ? rhoL : rhoR;
                    const TData &pUp     = (cond) ? pL : pR;
                    const TData *uUp     = (cond) ? uL : uR;
                    const TData &EUp     = (cond) ? EL : ER;
                    const TData &rhoMUp  = (cond) ? rhoML : rhoMR;
                    const TData *rhouMUp = (cond) ? rhouML : rhouMR;
                    const TData &SUp     = (cond) ? SL : SR;
                    const TData &EMUp    = (cond) ? EML : EMR;

                    // Compute flux
                    flux[i] = rhoUp * uUp[0] + SUp * (rhoMUp - rhoUp);
                    flux[1u * blksize + i] = rhoUp * uUp[0] * uUp[0] + pUp +
                                             SL * (rhouMUp[0] - rhoUp * uUp[0]);
#pragma unroll
                    for (unsigned int d = 1; d < NDIM; ++d)
                    {
                        flux[(1u + d) * blksize + i] =
                            rhoUp * uUp[0] * uUp[d] +
                            SUp * (rhouMUp[d] - rhoUp * uUp[d]);
                    }
                    flux[(1u + NDIM) * blksize + i] =
                        uUp[0] * (EUp + pUp) + SUp * (EMUp - EUp);
                }
            });
    }
};

} // namespace Nektar::Operators::detail
