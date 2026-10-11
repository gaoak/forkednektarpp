///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionSerialAVXStdMatKernels.hpp
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
// Description: Pointwise derivative-factor and velocity kernel of the
// Serial/AVX StdMat advection block operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionSerialAVXStdMatKernels.hpp
 * @brief Pointwise derivative-factor/velocity kernel of the Serial/AVX
 * StdMat advection block operator (see AdvectionSerialAVXStdMat.hpp).
 */

#pragma once

namespace Nektar::MultiRegions::detail
{

/**
 * @brief From reference-space derivatives, form the Cartesian gradient and
 * contract it with the advection velocity:
 * out = scale * sum_k V_k sum_d df(k,d) du/dxi_d, evaluated per quadrature
 * point with one element per SIMD lane.
 *
 * The derivative-factor entry k * dimension + d maps the
 * reference-direction-d derivative into the Cartesian-k gradient
 * component. For regular geometry one factor set is stored per element
 * group; for deformed geometry one per quadrature point.
 *
 * @tparam ExecSpace Execution space; unused here, kept so the call sites
 *                   match the Device kernel's signature.
 * @tparam APPEND    Accumulate onto @p outptr instead of overwriting it.
 * @tparam DEFORMED  Read per-point rather than per-element derivative
 *                   factors.
 * @tparam TData     SIMD vector type of the data (the width-one
 *                   tinysimd::scalarT wrapper on the Serial execution
 *                   space).
 * @tparam TScalar   Type of the scale factor.
 *
 * @param   nqTot     Quadrature points per element.
 * @param   ncoord    Cartesian gradient/velocity components.
 * @param   dimension Reference directions covered by @p inptr.
 * @param   nelmt     Element groups to process (the block operator passes
 *                    1 and advances the pointers itself).
 * @param   inoffset  Stride, in SIMD vectors, between the
 *                    reference-direction planes of @p inptr.
 * @param   dfptr     Derivative factors: ncoord * dimension vectors per
 *                    element group (regular) or point (deformed).
 * @param   advVel    Velocity data; component k starts at k * advelsize
 *                    vectors.
 * @param   advelsize Stride, in SIMD vectors, between velocity components.
 * @param   inptr     Reference-space derivatives, one plane of nqTot
 *                    vectors per direction.
 * @param   outptr    Output plane of nqTot vectors per element group.
 * @param   scale     Scalar factor applied to the result.
 */
template <typename ExecSpace, bool APPEND, bool DEFORMED, typename TData,
          typename TScalar>
NEK_FORCE_INLINE static void MultiplyByDerivFactorAndAdvecVelKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const TData *dfptr, const TData *advVel, const size_t advelsize,
    const TData *inptr, TData *outptr, const TScalar scale)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    TData tmp[3], tmp0;
    if constexpr (DEFORMED)
    {
        for (size_t idx = 0; idx < nsize; idx++)
        {
            tmp0 = 0.0;
            for (unsigned int k = 0; k < ncoord; k++)
            {
                tmp[k] = dfptr[ndf * idx + k * dimension] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k].fma(dfptr[ndf * idx + k * dimension + d],
                               inptr[d * inoffset + idx]);
                }
                tmp0.fma(advVel[k * advelsize + idx], tmp[k]);
            }

            if constexpr (APPEND)
            {
                outptr[idx].fma(scale, tmp0);
            }
            else
            {
                outptr[idx] = scale * tmp0;
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned int i = 0; i < nqTot; i++)
            {
                tmp0 = 0.0;
                for (unsigned int k = 0; k < ncoord; k++)
                {
                    tmp[k] =
                        dfptr[ndf * e + k * dimension] * inptr[nqTot * e + i];
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        tmp[k].fma(dfptr[ndf * e + k * dimension + d],
                                   inptr[nqTot * e + i + d * inoffset]);
                    }
                    tmp0.fma(advVel[k * advelsize + i], tmp[k]);
                }

                if constexpr (APPEND)
                {
                    outptr[nqTot * e + i].fma(scale, tmp0);
                }
                else
                {
                    outptr[nqTot * e + i] = scale * tmp0;
                }
            }
        }
    }
}

} // namespace Nektar::MultiRegions::detail
