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

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::detail
{

template <typename TData> NEK_HOSTDEVICE_INLINE TData M1Function(int A, TData M)
{
    TData out;

    if (A == 0)
    {
        out = 0.5 * (M + fabs(M));
    }
    else
    {
        out = 0.5 * (M - fabs(M));
    }

    return out;
}

template <typename TData> NEK_HOSTDEVICE_INLINE TData M2Function(int A, TData M)
{
    TData out;

    if (A == 0)
    {
        out = 0.25 * (M + 1.0) * (M + 1.0);
    }
    else
    {
        out = -0.25 * (M - 1.0) * (M - 1.0);
    }

    return out;
}

template <typename TData>
NEK_HOSTDEVICE_INLINE TData M4Function(int A, TData beta, TData M)
{
    TData out;

    if (fabs(M) >= 1.0)
    {
        out = M1Function(A, M);
    }
    else
    {
        out = M2Function(A, M);

        if (A == 0)
        {
            out *= 1.0 - 16.0 * beta * M2Function(1, M);
        }
        else
        {
            out *= 1.0 + 16.0 * beta * M2Function(0, M);
        }
    }

    return out;
}

template <typename TData>
NEK_HOSTDEVICE_INLINE TData P5Function(int A, TData alpha, TData M)
{
    TData out;

    if (fabs(M) >= 1.0)
    {
        out = (1.0 / M) * M1Function(A, M);
    }
    else
    {
        out = M2Function(A, M);

        if (A == 0)
        {
            out *= (2.0 - M) - 16.0 * alpha * M * M2Function(1, M);
        }
        else
        {
            out *= (-2.0 - M) + 16.0 * alpha * M * M2Function(0, M);
        }
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
        constexpr unsigned int vec_width =
            (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                ? tinysimd::simd<TScalar>::width
                : 1;

        const TScalar oneHalf = 0.5;

        // Currently not directly vectorisable due to branching
        for (unsigned int k = 0; k < vec_width; k++)
        {
            // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment
            // exist; the normal [n0 | n1 | n2] alike.
            const TScalar rhoL = fwd[0];
            const TScalar rhoR = bwd[0];

            // Velocities, normal velocities and kinetic energy terms.
            TScalar n[NDIM], uL[NDIM], uR[NDIM];
            TScalar unL = 0.0, unR = 0.0, qL2 = 0.0, qR2 = 0.0;
#pragma unroll
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                n[d]                = normal[d * blksize];
                const TScalar rhouL = fwd[(1u + d) * blksize];
                const TScalar rhouR = bwd[(1u + d) * blksize];
                uL[d]               = rhouL / rhoL;
                uR[d]               = rhouR / rhoR;
                unL += uL[d] * n[d];
                unR += uR[d] * n[d];
                qL2 += rhouL * uL[d];
                qR2 += rhouR * uR[d];
            }

            // Internal energy per unit mass, pressure and sound speed.
            const TScalar EL = fwd[(1u + NDIM) * blksize];
            const TScalar ER = bwd[(1u + NDIM) * blksize];
            const TScalar eL = (EL - oneHalf * qL2) / rhoL;
            const TScalar eR = (ER - oneHalf * qR2) / rhoR;
            const TScalar pL = GetPressure(EoS, rhoL, eL);
            const TScalar pR = GetPressure(EoS, rhoR, eR);
            const TScalar cL = GetSoundSpeed(EoS, rhoL, eL);
            const TScalar cR = GetSoundSpeed(EoS, rhoR, eR);

            // Average speed of sound and the normal Mach numbers.
            const TScalar cA = 0.5 * (cL + cR);
            const TScalar ML = unL / cA;
            const TScalar MR = unR / cA;

            // Interface Mach number and pressure.
            TScalar pbar, Mbar;
            AUSMUpwinding()(cA, rhoL, rhoR, pL, pR, ML, MR, pbar, Mbar);

            // The upwind state.
            const bool cond      = Mbar >= 0.0;
            const TScalar &rhoUp = (cond) ? rhoL : rhoR;
            const TScalar &pUp   = (cond) ? pL : pR;
            const TScalar *uUp   = (cond) ? uL : uR;
            const TScalar &EUp   = (cond) ? EL : ER;

            flux[0] = cA * Mbar * rhoUp;
#pragma unroll
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                flux[(1u + d) * blksize] =
                    cA * Mbar * rhoUp * uUp[d] + pbar * n[d];
            }
            flux[(1u + NDIM) * blksize] = cA * Mbar * (EUp + pUp);

            fwd++;
            bwd++;
            normal++;
            flux++;
        }
    }
};

} // namespace Nektar::detail
