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

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/LoopExecution/LoopExecution.hpp>

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
    /// stride @p blksize, in TData (a scalar or a SIMD vector of points).
    template <typename TData>
    NEK_HOSTDEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                          const size_t blksize,
                                          const TData *fwd, const TData *bwd,
                                          const TData *normal, TData *flux)
    {
        const TData half = TData(0.5);

        // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
        const auto l = fwd;
        const auto r = bwd;

        // Velocities, normal velocities and kinetic energy terms, with one
        // reciprocal per density in place of a division at every use.
        const TData rhoL = l[0], rhoR = r[0];
        const TData invRhoL = TData(1.0) / rhoL;
        const TData invRhoR = TData(1.0) / rhoR;
        TData n[NDIM], uL[NDIM], uR[NDIM];
        TData unL = TData(0.0), unR = TData(0.0);
        TData qL2 = TData(0.0), qR2 = TData(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            n[d]              = normal[d * blksize];
            const TData rhouL = l[(1u + d) * blksize];
            const TData rhouR = r[(1u + d) * blksize];
            uL[d]             = rhouL * invRhoL;
            uR[d]             = rhouR * invRhoR;
            unL += n[d] * uL[d];
            unR += n[d] * uR[d];
            qL2 += rhouL * uL[d];
            qR2 += rhouR * uR[d];
        }

        // Internal energy per unit mass, pressure and sound speed.
        const TData EL = l[(1u + NDIM) * blksize];
        const TData ER = r[(1u + NDIM) * blksize];
        const TData eL = (EL - half * qL2) * invRhoL;
        const TData eR = (ER - half * qR2) * invRhoR;
        const TData pL = GetPressure(EoS, rhoL, eL);
        const TData pR = GetPressure(EoS, rhoR, eR);
        const TData cL = GetSoundSpeed(EoS, rhoL, eL);
        const TData cR = GetSoundSpeed(EoS, rhoR, eR);

        // Average speed of sound and the normal Mach numbers.
        const TData cA   = half * (cL + cR);
        const TData invC = TData(1.0) / cA;
        const TData ML   = unL * invC;
        const TData MR   = unR * invC;

        // Interface Mach number and pressure, and the mass flux they carry.
        TData pbar, Mbar;
        AUSMUpwinding()(cA, rhoL, rhoR, pL, pR, ML, MR, pbar, Mbar);
        const TData mn = cA * Mbar;

        // The upwind state, lane by lane for a vector.
        const auto cond   = Mbar >= TData(0.0);
        const TData rhoUp = Select(cond, rhoL, rhoR);
        const TData pUp   = Select(cond, pL, pR);
        const TData EUp   = Select(cond, EL, ER);

        flux[0] = mn * rhoUp;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            flux[(1u + d) * blksize] =
                mn * rhoUp * Select(cond, uL[d], uR[d]) + pbar * n[d];
        }
        flux[(1u + NDIM) * blksize] = mn * (EUp + pUp);
    }
};

} // namespace Nektar::detail
