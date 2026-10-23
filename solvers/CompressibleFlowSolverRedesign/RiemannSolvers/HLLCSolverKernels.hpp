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
        constexpr unsigned int vec_width =
            (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                ? tinysimd::simd<TScalar>::width
                : 1;

        using std::sqrt;

        const TScalar oneHalf = 0.5;

        // Currently not directly vectorisable due to branching
        for (unsigned int k = 0; k < vec_width; k++)
        {
            // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment
            // exist; the normal [n0 | n1 | n2] alike.
            const TScalar rhoL = fwd[0];
            const TScalar rhoR = bwd[0];

            // Velocities, normal velocities and kinetic energy terms.
            TScalar n[NDIM], mL[NDIM], mR[NDIM], uL[NDIM], uR[NDIM];
            TScalar unL = 0.0, unR = 0.0, qL2 = 0.0, qR2 = 0.0;
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

            // Internal energy per unit mass, pressure, sound speed, enthalpy.
            const TScalar EL = fwd[(1u + NDIM) * blksize];
            const TScalar ER = bwd[(1u + NDIM) * blksize];
            const TScalar eL = (EL - oneHalf * qL2) / rhoL;
            const TScalar eR = (ER - oneHalf * qR2) / rhoR;
            const TScalar pL = GetPressure(EoS, rhoL, eL);
            const TScalar pR = GetPressure(EoS, rhoR, eR);
            const TScalar cL = GetSoundSpeed(EoS, rhoL, eL);
            const TScalar cR = GetSoundSpeed(EoS, rhoR, eR);
            const TScalar hL = (EL + pL) / rhoL;
            const TScalar hR = (ER + pR) / rhoR;

            // Roe averages.
            const TScalar srL  = sqrt(rhoL);
            const TScalar srR  = sqrt(rhoR);
            const TScalar srLR = srL + srR;
            TScalar unRoe = 0.0, URoe2 = 0.0;
#pragma unroll
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const TScalar uRoe = (srL * uL[d] + srR * uR[d]) / srLR;
                unRoe += uRoe * n[d];
                URoe2 += uRoe * uRoe;
            }
            const TScalar hRoe = (srL * hL + srR * hR) / srLR;
            const TScalar cRoe =
                GetRoeSoundSpeed(EoS, rhoL, pL, eL, hL, srL, rhoR, pR, eR, hR,
                                 srR, hRoe, URoe2, srLR);

            // Mass flux through the normal from each side, and the largest
            // wave speeds.
            const TScalar mnL = rhoL * unL;
            const TScalar mnR = rhoR * unR;
            const TScalar SL  = std::min(unL - cL, unRoe - cRoe);
            const TScalar SR  = std::max(unR + cR, unRoe + cRoe);

            if (SL >= 0)
            {
                // The left state's flux through n.
                flux[0] = mnL;
#pragma unroll
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    flux[(1u + d) * blksize] = mnL * uL[d] + pL * n[d];
                }
                flux[(1u + NDIM) * blksize] = unL * (EL + pL);
            }
            else if (SR <= 0)
            {
                // The right state's flux through n.
                flux[0] = mnR;
#pragma unroll
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    flux[(1u + d) * blksize] = mnR * uR[d] + pR * n[d];
                }
                flux[(1u + NDIM) * blksize] = unR * (ER + pR);
            }
            else
            {
                // HLLC: the star state on the side the contact wave SM leaves
                // behind. Only the normal velocity changes there, to SM; the
                // tangential velocity is advected unaltered, which in
                // Cartesian components is u + (SM - u.n) n.
                const TScalar SM =
                    (pR - pL + mnL * (SL - unL) - mnR * (SR - unR)) /
                    (rhoL * (SL - unL) - rhoR * (SR - unR));

                const bool cond    = SL < 0.0 && SM >= 0.0;
                const TScalar rho  = cond ? rhoL : rhoR;
                const TScalar un   = cond ? unL : unR;
                const TScalar p    = cond ? pL : pR;
                const TScalar E    = cond ? EL : ER;
                const TScalar S    = cond ? SL : SR;
                const TScalar *u   = cond ? uL : uR;
                const TScalar *m   = cond ? mL : mR;
                const TScalar mn   = rho * un;
                const TScalar rhoM = rho * (S - un) / (S - SM);
                const TScalar EM =
                    rhoM * (E / rho + (SM - un) * (SM + p / (rho * (S - un))));

                flux[0] = mn + S * (rhoM - rho);
#pragma unroll
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    const TScalar mM = rhoM * (u[d] + (SM - un) * n[d]);
                    flux[(1u + d) * blksize] =
                        mn * u[d] + p * n[d] + S * (mM - m[d]);
                }
                flux[(1u + NDIM) * blksize] = un * (E + p) + S * (EM - E);
            }

            fwd++;
            bwd++;
            normal++;
            flux++;
        }
    }
};

} // namespace Nektar::detail
