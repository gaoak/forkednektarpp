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
     * @param blksize  Component stride, in TData (a scalar or a SIMD vector
     *                 of points): component c of the states, the normal and
     *                 the flux lies at offset `c * blksize`.
     * @param normal   The unit trace normal, @p NDIM components.
     */
    template <typename TData>
    NEK_HOSTDEVICE_INLINE void operator()(const EqnOfStParams EoS,
                                          const size_t blksize,
                                          const TData *fwd, const TData *bwd,
                                          const TData *normal, TData *flux)
    {
        using std::abs;
        using std::sqrt;

        const TData oneHalf = 0.5;

        // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
        const TData rhoL = fwd[0];
        const TData rhoR = bwd[0];

        // Velocities, normal velocities and kinetic energy terms.
        TData n[NDIM], mL[NDIM], mR[NDIM], uL[NDIM], uR[NDIM];
        TData unL = 0.0, unR = 0.0, qL2 = 0.0, qR2 = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            n[d]  = normal[d * blksize];
            mL[d] = fwd[(1u + d) * blksize];
            mR[d] = bwd[(1u + d) * blksize];
            uL[d] = mL[d] / rhoL;
            uR[d] = mR[d] / rhoR;
            unL += uL[d] * n[d];
            unR += uR[d] * n[d];
            qL2 += mL[d] * uL[d];
            qR2 += mR[d] * uR[d];
        }

        // Internal energy per unit mass, pressure and enthalpy.
        const TData EL = fwd[(1u + NDIM) * blksize];
        const TData ER = bwd[(1u + NDIM) * blksize];
        const TData eL = (EL - oneHalf * qL2) / rhoL;
        const TData eR = (ER - oneHalf * qR2) / rhoR;
        const TData pL = GetPressure(EoS, rhoL, eL);
        const TData pR = GetPressure(EoS, rhoR, eR);
        const TData hL = (EL + pL) / rhoL;
        const TData hR = (ER + pR) / rhoR;

        // Roe averages.
        const TData srL  = sqrt(rhoL);
        const TData srR  = sqrt(rhoR);
        const TData srLR = srL + srR;
        TData unRoe = 0.0, URoe2 = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const TData uRoe = (srL * uL[d] + srR * uR[d]) / srLR;
            unRoe += uRoe * n[d];
            URoe2 += uRoe * uRoe;
        }
        const TData hRoe = (srL * hL + srR * hR) / srLR;
        const TData cRoe = GetRoeSoundSpeed(EoS, rhoL, pL, eL, hL, srL, rhoR,
                                            pR, eR, hR, srR, hRoe, URoe2, srLR);

        // Largest wave speed through the normal.
        const TData a = abs(unRoe) + cRoe;

        // Mass, momentum and energy: the central flux through n less the
        // dissipation of the jump.
        flux[0] = oneHalf * (rhoL * unL + rhoR * unR - a * (rhoR - rhoL));
#pragma unroll
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            flux[(1u + d) * blksize] =
                oneHalf * (mL[d] * unL + mR[d] * unR + (pL + pR) * n[d] -
                           a * (mR[d] - mL[d]));
        }
        flux[(1u + NDIM) * blksize] =
            oneHalf * (unL * (EL + pL) + unR * (ER + pR) - a * (ER - EL));
    }
};

} // namespace Nektar::detail
