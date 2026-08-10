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

namespace Nektar::Operators::detail
{

template <typename TData> NEK_DEVICE_INLINE TData M1Function(int A, TData M)
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

template <typename TData> NEK_DEVICE_INLINE TData M2Function(int A, TData M)
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
NEK_DEVICE_INLINE TData M4Function(int A, TData beta, TData M)
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
NEK_DEVICE_INLINE TData P5Function(int A, TData alpha, TData M)
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

template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM,
          typename AUSMUpwinding>
struct AUSMSolverKernel
{
    template <typename TScalar>
    NEK_DEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                      const size_t blksize, const TScalar *fwd,
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
            const TScalar pL = GetPressure(EoS, rhoL, eL);
            const TScalar pR = GetPressure(EoS, rhoR, eR);

            // Speed of sound
            const TScalar cL = GetSoundSpeed(EoS, rhoL, eL);
            const TScalar cR = GetSoundSpeed(EoS, rhoR, eR);

            // Average speeds of sound
            TScalar cA = 0.5 * (cL + cR);

            // Local Mach numbers
            TScalar ML = uL[0] / cA;
            TScalar MR = uR[0] / cA;

            // Parameters for specify the upwinding
            TScalar pbar, Mbar;
            AUSMUpwinding()(cA, rhoL, rhoR, pL, pR, ML, MR, pbar, Mbar);

            // Conditional assignment
            const bool cond      = Mbar >= 0.0;
            const TScalar &rhoUp = (cond) ? rhoL : rhoR;
            const TScalar &pUp   = (cond) ? pL : pR;
            const TScalar *uUp   = (cond) ? uL : uR;
            const TScalar &EUp   = (cond) ? EL : ER;

            // Compute flux
            flux[0]            = cA * Mbar * rhoUp;
            flux[1u * blksize] = cA * Mbar * rhoUp * uUp[0] + pbar;
#pragma unroll
            for (unsigned int d = 1; d < NDIM; ++d)
            {
                flux[(1u + d) * blksize] = cA * Mbar * rhoUp * uUp[d];
            }
            flux[(1u + NDIM) * blksize] = cA * Mbar * (EUp + pUp);

            fwd++;
            bwd++;
            flux++;
        }
    }
};

} // namespace Nektar::Operators::detail
