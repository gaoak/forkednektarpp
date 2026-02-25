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

#include "Operators/LoopExecution/LoopExecution.hpp"

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

template <typename ExecSpace, unsigned int NDIM, typename AUSMUpwinding>
struct AUSMSolverKernel
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

                // Average speeds of sound
                TData cA = 0.5 * (cL + cR);

                // Local Mach numbers
                TData ML = uL[0] / cA;
                TData MR = uR[0] / cA;

                // Parameters for specify the upwinding
                TData pbar, Mbar;
                AUSMUpwinding()(cA, rhoL, rhoR, pL, pR, ML, MR, pbar, Mbar);

                // Conditional assignment
                const bool cond    = Mbar >= 0.0;
                const TData &rhoUp = (cond) ? rhoL : rhoR;
                const TData &pUp   = (cond) ? pL : pR;
                const TData *uUp   = (cond) ? uL : uR;
                const TData &EUp   = (cond) ? EL : ER;

                // Compute flux
                flux[i]                = cA * Mbar * rhoUp;
                flux[1u * blksize + i] = cA * Mbar * rhoUp * uUp[0] + pbar;
#pragma unroll
                for (unsigned int d = 1; d < NDIM; ++d)
                {
                    flux[(1u + d) * blksize + i] = cA * Mbar * rhoUp * uUp[d];
                }
                flux[(1u + NDIM) * blksize + i] = cA * Mbar * (EUp + pUp);
            });
    }
};

} // namespace Nektar::Operators::detail
