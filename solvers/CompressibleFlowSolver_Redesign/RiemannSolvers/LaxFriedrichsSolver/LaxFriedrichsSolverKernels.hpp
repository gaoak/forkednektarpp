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

#include "Operators/LoopExecution/LoopExecution.hpp"

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::Operators::detail
{

// TODO: move to Common/Spaces.hpp and tidy.
template <bool B, typename TData> struct data_type_if
{
    typedef TData type;
};

template <typename TData> struct data_type_if<true, TData>
{
    typedef tinysimd::simd<TData> type;
};

template <typename TData>
NEK_DEVICE_INLINE TData GetPressure(const TData &rho, const TData &e)
{
    // Ideal gas law: P = (gamma - 1) * rho * e
    const TData gamma = 1.4; // Specific heat ratio for air
    return (gamma - 1) * rho * e;
}

template <typename TData>
NEK_DEVICE_INLINE TData GetRoeSoundSpeed(
    [[maybe_unused]] const TData &rhoL, [[maybe_unused]] const TData &pL,
    [[maybe_unused]] const TData &eL, [[maybe_unused]] const TData &HL,
    [[maybe_unused]] const TData &srL, [[maybe_unused]] const TData &rhoR,
    [[maybe_unused]] const TData &pR, [[maybe_unused]] const TData &eR,
    [[maybe_unused]] const TData &HR, [[maybe_unused]] const TData &srR,
    const TData &HRoe, const TData &URoe2, [[maybe_unused]] const TData &srLR)
{
    // Specific heat ratio for air
    const TData gamma = 1.4;
    // Calculate sound speed using ideal gas relation
#if defined(_MSC_VER)
    return std::sqrt((gamma - 1.0) * (HRoe - 0.5 * URoe2));
#else
    return sqrt((gamma - 1.0) * (HRoe - 0.5 * URoe2));
#endif
}

template <typename ExecSpace, typename TScalar, unsigned int NDIM>
NEK_FORCE_INLINE static void LaxFriedrichsSolverKernel(const size_t blksize,
                                                       const TScalar *fwd,
                                                       const TScalar *bwd,
                                                       TScalar *flux)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    const unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TScalar>::width
            : 1;

    // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
    Nektar::parallel_for<ExecSpace>(
        0u, blksize / vec_width, NEKTAR_LAMBDA(const size_t i) {
            const vec_t oneHalf = 0.5;

            // Density
            const vec_t rhoL = reinterpret_cast<const vec_t *>(fwd)[i];
            const vec_t rhoR = reinterpret_cast<const vec_t *>(bwd)[i];

            // Physical fluxes in normal direction (note: uL[0] is normal
            // velocity)
            const vec_t Fn_rho_L =
                reinterpret_cast<const vec_t *>(fwd + 1u * blksize)[i];
            const vec_t Fn_rho_R =
                reinterpret_cast<const vec_t *>(bwd + 1u * blksize)[i];

            // Velocities and kinetic energy terms (in rotated frame: m0 is
            // normal)
            vec_t uL[3] = {0, 0, 0};
            vec_t uR[3] = {0, 0, 0};
            uL[0]       = Fn_rho_L / rhoL;
            uR[0]       = Fn_rho_R / rhoR;
            vec_t qL2   = Fn_rho_L * uL[0];
            vec_t qR2   = Fn_rho_R * uR[0];
            for (unsigned int d = 1; d < NDIM; ++d)
            {
                const vec_t rhouL = reinterpret_cast<const vec_t *>(
                    fwd + (1u + d) * blksize)[i];
                const vec_t rhouR = reinterpret_cast<const vec_t *>(
                    bwd + (1u + d) * blksize)[i];

                uL[d] = rhouL / rhoL;
                uR[d] = rhouR / rhoR;

                qL2 += rhouL * uL[d];
                qR2 += rhouR * uR[d];
            }

            // Internal energy per unit mass
            const vec_t EL =
                reinterpret_cast<const vec_t *>(fwd + (1u + NDIM) * blksize)[i];
            const vec_t ER =
                reinterpret_cast<const vec_t *>(bwd + (1u + NDIM) * blksize)[i];
            const vec_t eL = (EL - oneHalf * qL2) / rhoL;
            const vec_t eR = (ER - oneHalf * qR2) / rhoR;

            // Pressure + enthalpy
            const vec_t pL = GetPressure(rhoL, eL);
            const vec_t pR = GetPressure(rhoR, eR);
            const vec_t HL = (EL + pL) / rhoL;
            const vec_t HR = (ER + pR) / rhoR;

        // Roe averages (same as your code)
#if defined(_MSC_VER)
            const vec_t srL = std::sqrt(rhoL);
            const vec_t srR = std::sqrt(rhoR);
#else
            const vec_t srL  = sqrt(rhoL);
            const vec_t srR  = sqrt(rhoR);
#endif
            const vec_t srLR = srL + srR;

            vec_t uRoe[3] = {0, 0, 0};
            vec_t URoe2   = 0;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                uRoe[d] = (srL * uL[d] + srR * uR[d]) / srLR;
                URoe2 += uRoe[d] * uRoe[d];
            }

            const vec_t HRoe = (srL * HL + srR * HR) / srLR;

            const vec_t cRoe = GetRoeSoundSpeed(rhoL, pL, eL, HL, srL, rhoR, pR,
                                                eR, HR, srR, HRoe, URoe2, srLR);

        // Max eigenvalue in normal direction
#if defined(_MSC_VER)
            const vec_t a = std::abs(uRoe[0]) + cRoe;
#else
            const vec_t a = abs(uRoe[0]) + cRoe;
#endif

            // Mass
            reinterpret_cast<vec_t *>(flux)[i] =
                oneHalf * (Fn_rho_L + Fn_rho_R - a * (rhoR - rhoL));

            // Normal momentum (d=0): p + m0*u0
            const vec_t Fm0_L = pL + Fn_rho_L * uL[0];
            const vec_t Fm0_R = pR + Fn_rho_R * uR[0];
            reinterpret_cast<vec_t *>(flux + 1u * blksize)[i] =
                oneHalf * (Fm0_L + Fm0_R - a * (Fn_rho_R - Fn_rho_L));

            // Tangential momentum(s): m0*u_t
            for (unsigned int d = 1; d < NDIM; ++d)
            {
                const vec_t Fmd_L = Fn_rho_L * uL[d];
                const vec_t Fmd_R = Fn_rho_R * uR[d];
                reinterpret_cast<vec_t *>(flux + (1u + d) * blksize)[i] =
                    oneHalf *
                    (Fmd_L + Fmd_R - a * (rhoR * uR[d] - rhoL * uL[d]));
            }

            // Energy: u0*(E+p)
            const vec_t FE_L = uL[0] * (EL + pL);
            const vec_t FE_R = uR[0] * (ER + pR);
            reinterpret_cast<vec_t *>(flux + (1u + NDIM) * blksize)[i] =
                oneHalf * (FE_L + FE_R - a * (ER - EL));
        });
}

} // namespace Nektar::Operators::detail
