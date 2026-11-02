///////////////////////////////////////////////////////////////////////////////
//
// File: AUSMSolverKernels.hpp
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

#include "LibUtilities/LoopExecution/LoopExecution.hpp"
#include <LibUtilities/Backends/Backends.hpp>

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::detail
{

// The split Mach and pressure functions of the AUSM family, for A = 0 the
// part carried from the left and for A = 1 that from the right. TData is a
// scalar or a SIMD vector: a vector takes the branch on |M| lane by lane, with
// every lane forming both sides and keeping the one its |M| selects.
template <typename TData> NEK_HOSTDEVICE_INLINE TData M1Function(int A, TData M)
{
    using std::abs;

    TData out;

    if (A == 0)
    {
        out = TData(0.5) * (M + abs(M));
    }
    else
    {
        out = TData(0.5) * (M - abs(M));
    }

    return out;
}

template <typename TData> NEK_HOSTDEVICE_INLINE TData M2Function(int A, TData M)
{
    TData out;

    if (A == 0)
    {
        out = TData(0.25) * (M + TData(1.0)) * (M + TData(1.0));
    }
    else
    {
        out = TData(-0.25) * (M - TData(1.0)) * (M - TData(1.0));
    }

    return out;
}

template <typename TData>
NEK_HOSTDEVICE_INLINE TData M4Function(int A, TData beta, TData M)
{
    using std::abs;

    TData out;

    if constexpr (std::is_arithmetic_v<TData>)
    {
        if (abs(M) >= TData(1.0))
        {
            out = M1Function(A, M);
        }
        else
        {
            out = M2Function(A, M);

            if (A == 0)
            {
                out *= TData(1.0) - TData(16.0) * beta * M2Function(1, M);
            }
            else
            {
                out *= TData(1.0) + TData(16.0) * beta * M2Function(0, M);
            }
        }
    }
    else
    {
        TData sub = M2Function(A, M);

        if (A == 0)
        {
            sub = sub * (TData(1.0) - TData(16.0) * beta * M2Function(1, M));
        }
        else
        {
            sub = sub * (TData(1.0) + TData(16.0) * beta * M2Function(0, M));
        }

        out = select(abs(M) >= TData(1.0), M1Function(A, M), sub);
    }

    return out;
}

template <typename TData>
NEK_HOSTDEVICE_INLINE TData P5Function(int A, TData alpha, TData M)
{
    using std::abs;

    TData out;

    if constexpr (std::is_arithmetic_v<TData>)
    {
        if (abs(M) >= TData(1.0))
        {
            out = (TData(1.0) / M) * M1Function(A, M);
        }
        else
        {
            out = M2Function(A, M);

            if (A == 0)
            {
                out *= (TData(2.0) - M) -
                       TData(16.0) * alpha * M * M2Function(1, M);
            }
            else
            {
                out *= (TData(-2.0) - M) +
                       TData(16.0) * alpha * M * M2Function(0, M);
            }
        }
    }
    else
    {
        // The supersonic side divides by M, which may be zero in a lane
        // that takes the subsonic one.
        TData sub = M2Function(A, M);

        if (A == 0)
        {
            sub = sub * ((TData(2.0) - M) -
                         TData(16.0) * alpha * M * M2Function(1, M));
        }
        else
        {
            sub = sub * ((TData(-2.0) - M) +
                         TData(16.0) * alpha * M * M2Function(0, M));
        }

        out = select(abs(M) >= TData(1.0), (TData(1.0) / M) * M1Function(A, M),
                     sub);
    }

    return out;
}

/**
 * @brief The AUSM flux through the trace normal, in Cartesian components (as
 * the legacy AUSM solvers): the interface Mach number and pressure from the
 * normal Mach numbers by @p AUSMUpwinding, the upwind state advected through
 * the normal and the pressure acting along it.
 */
template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM,
          typename AUSMUpwinding>
struct AUSMSolverKernel
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

        // Velocities, normal velocities and kinetic energy terms, with one
        // reciprocal per density in place of a division at every use.
        const vec_t rhoL = l[0], rhoR = r[0];
        const vec_t invRhoL = vec_t(1.0) / rhoL;
        const vec_t invRhoR = vec_t(1.0) / rhoR;
        vec_t n[NDIM], uL[NDIM], uR[NDIM];
        vec_t unL = vec_t(0.0), unR = vec_t(0.0);
        vec_t qL2 = vec_t(0.0), qR2 = vec_t(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            n[d]              = nvec[d * gs];
            const vec_t rhouL = l[(1u + d) * gs];
            const vec_t rhouR = r[(1u + d) * gs];
            uL[d]             = rhouL * invRhoL;
            uR[d]             = rhouR * invRhoR;
            unL += n[d] * uL[d];
            unR += n[d] * uR[d];
            qL2 += rhouL * uL[d];
            qR2 += rhouR * uR[d];
        }

        // Internal energy per unit mass, pressure and sound speed.
        const vec_t EL = l[(1u + NDIM) * gs];
        const vec_t ER = r[(1u + NDIM) * gs];
        const vec_t eL = (EL - half * qL2) * invRhoL;
        const vec_t eR = (ER - half * qR2) * invRhoR;
        const vec_t pL = GetPressure(EoS, rhoL, eL);
        const vec_t pR = GetPressure(EoS, rhoR, eR);
        const vec_t cL = GetSoundSpeed(EoS, rhoL, eL);
        const vec_t cR = GetSoundSpeed(EoS, rhoR, eR);

        // Average speed of sound and the normal Mach numbers.
        const vec_t cA   = half * (cL + cR);
        const vec_t invC = vec_t(1.0) / cA;
        const vec_t ML   = unL * invC;
        const vec_t MR   = unR * invC;

        // Interface Mach number and pressure, and the mass flux they carry.
        vec_t pbar, Mbar;
        AUSMUpwinding()(cA, rhoL, rhoR, pL, pR, ML, MR, pbar, Mbar);
        const vec_t mn = cA * Mbar;

        // The upwind state, lane by lane for a vector.
        const auto cond   = Mbar >= vec_t(0.0);
        const vec_t rhoUp = Select(cond, rhoL, rhoR);
        const vec_t pUp   = Select(cond, pL, pR);
        const vec_t EUp   = Select(cond, EL, ER);

        f[0] = mn * rhoUp;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            f[(1u + d) * gs] =
                mn * rhoUp * Select(cond, uL[d], uR[d]) + pbar * n[d];
        }
        f[(1u + NDIM) * gs] = mn * (EUp + pUp);
    }
};

} // namespace Nektar::detail
