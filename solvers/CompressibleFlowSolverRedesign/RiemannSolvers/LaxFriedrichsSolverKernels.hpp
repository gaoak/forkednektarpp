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

template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM>
struct LaxFriedrichsSolverKernel
{
    template <typename TScalar>
    NEK_DEVICE_INLINE void operator()(const EqnOfStParams EoS,
                                      const size_t blksize, const TScalar *fwd,
                                      const TScalar *bwd, TScalar *flux)
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
        auto fluxvec           = reinterpret_cast<vec_t *>(flux);

        // Density
        const vec_t rhoL = fwdvec[0];
        const vec_t rhoR = bwdvec[0];

        // Physical fluxes in normal direction (note: uL[0] is normal
        // velocity)
        const vec_t Fn_rho_L = fwdvec[1u * groupsize];
        const vec_t Fn_rho_R = bwdvec[1u * groupsize];

        // Velocities and kinetic energy terms (in rotated frame: m0 is
        // normal)
        vec_t uL[NDIM];
        vec_t uR[NDIM];
        uL[0]     = Fn_rho_L / rhoL;
        uR[0]     = Fn_rho_R / rhoR;
        vec_t qL2 = Fn_rho_L * uL[0];
        vec_t qR2 = Fn_rho_R * uR[0];
#pragma unroll
        for (unsigned int d = 1; d < NDIM; ++d)
        {
            const vec_t rhouL = fwdvec[(1u + d) * groupsize];
            const vec_t rhouR = bwdvec[(1u + d) * groupsize];

            uL[d] = rhouL / rhoL;
            uR[d] = rhouR / rhoR;

            qL2 += rhouL * uL[d];
            qR2 += rhouR * uR[d];
        }

        // Internal energy per unit mass
        const vec_t EL = fwdvec[(1u + NDIM) * groupsize];
        const vec_t ER = bwdvec[(1u + NDIM) * groupsize];
        const vec_t eL = (EL - oneHalf * qL2) / rhoL;
        const vec_t eR = (ER - oneHalf * qR2) / rhoR;

        // Pressure
        const vec_t pL = GetPressure(EoS, rhoL, eL);
        const vec_t pR = GetPressure(EoS, rhoR, eR);

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

        const vec_t cRoe = GetRoeSoundSpeed(EoS, rhoL, pL, eL, hL, srL, rhoR,
                                            pR, eR, hR, srR, hRoe, URoe2, srLR);

        // Max eigenvalue in normal direction
        const vec_t a = abs(uRoe[0]) + cRoe;

        // Mass
        fluxvec[0] = oneHalf * (Fn_rho_L + Fn_rho_R - a * (rhoR - rhoL));

        // Normal momentum (d=0): p + m0*u0
        const vec_t Fm0_L = pL + Fn_rho_L * uL[0];
        const vec_t Fm0_R = pR + Fn_rho_R * uR[0];
        fluxvec[1u * groupsize] =
            oneHalf * (Fm0_L + Fm0_R - a * (Fn_rho_R - Fn_rho_L));

        // Tangential momentum(s): m0*u_t
#pragma unroll
        for (unsigned int d = 1; d < NDIM; ++d)
        {
            const vec_t Fmd_L = Fn_rho_L * uL[d];
            const vec_t Fmd_R = Fn_rho_R * uR[d];
            fluxvec[(1u + d) * groupsize] =
                oneHalf * (Fmd_L + Fmd_R - a * (rhoR * uR[d] - rhoL * uL[d]));
        }

        // Energy: u0*(E+p)
        const vec_t FE_L = uL[0] * (EL + pL);
        const vec_t FE_R = uR[0] * (ER + pR);
        fluxvec[(1u + NDIM) * groupsize] =
            oneHalf * (FE_L + FE_R - a * (ER - EL));
    }
};

} // namespace Nektar::detail
