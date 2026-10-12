///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialAVXStdMatKernels.hpp
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
// Description: Serial/AVX StdMat kernels of the inner product with
// the gradient of the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTDerivBaseSerialAVXStdMatKernels.hpp
 * @brief Serial/AVX StdMat kernels of the inner product with the gradient
 * of the basis: the geometric stage that turns the input vector into one
 * physical-space field per reference direction.
 *
 * @details
 * Both kernels below evaluate, for every quadrature point of every element
 * of a group,
 * \f[ g_d = \mathrm{scale} \cdot J \cdot
 *           \sum_k \frac{\partial \xi_d}{\partial x_k} f_k , \f]
 * the field the StdMat implementation then contracts against the
 * standard-region matrices (see IProductWRTDerivBaseSerialAVXStdMat.hpp
 * for the two stages, and IProductWRTDerivBaseOp.hpp for what \f$g_d\f$
 * means). They differ only in whether the quadrature weights are applied
 * as well: the Coeff path leaves them to the standard-region matrix, the
 * Phys path does not have them there.
 *
 * @p TData is instantiated with the SIMD vector type, so a single index
 * steps over whole groups of interleaved elements and the .fma() calls are
 * vector fused multiply-adds; on the Serial execution space the vector
 * type is scalar.
 *
 * The derivative factors are read untransposed: the ncoord * dimension
 * factors belonging to one point are contiguous, indexed
 * k * dimension + d. On a regular (non-deformed) block there is one such
 * set and one Jacobian per element; on a deformed block there is one per
 * quadrature point.
 */

#pragma once

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Multiply the input vector by the derivative factors and the
 * Jacobian, without the quadrature weights (Coeff output path).
 *
 * @tparam ExecSpace   Execution space of the calling implementation
 *                     (Serial or AVX); unused in the body.
 * @tparam DEFORMED    True when the geometric factors vary within an
 *                     element.
 * @tparam TData       SIMD vector type of one interleaved element group.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Coordinate dimension: number of input components
 *                      read.
 * @param   dimension   Reference dimension: number of output fields
 *                      written.
 * @param   nelmt       Element groups processed by this call (the block
 *                      implementation passes 1).
 * @param   inoffset    Stride, in vectors, between the input's coordinate
 *                      components.
 * @param   outsize     Stride, in vectors, between the output fields of
 *                      successive reference directions.
 * @param   jacptr      Jacobians of the group.
 * @param   dfptr       Derivative factors of the group.
 * @param   inptr       Input vector \f${\bf f}\f$; component k starts at
 *                      inptr + k * inoffset.
 * @param   outptr      Receives the @p dimension fields \f$g_d\f$, field d
 *                      starting at outptr + d * outsize.
 * @param   scale       Factor applied to every value written.
 */
template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void JacobianDerivFactorKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outsize, const TData *jacptr, const TData *dfptr,
    const TData *inptr, TData *outptr, const TData scale)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    TData tmp[3];
    if constexpr (DEFORMED)
    {
        for (size_t idx = 0; idx < nsize; idx++)
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] = dfptr[ndf * idx + d] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d].fma(dfptr[ndf * idx + k * dimension + d],
                               inptr[k * inoffset + idx]);
                }
            }
            for (unsigned int d = 0; d < dimension; d++)
            {
                outptr[d * outsize + idx] = scale * tmp[d] * jacptr[idx];
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned int i = 0; i < nqTot; i++)
            {
                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[ndf * e + d] * inptr[nqTot * e + i];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp[d].fma(dfptr[ndf * e + k * dimension + d],
                                   inptr[nqTot * e + i + k * inoffset]);
                    }
                }
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outsize + nqTot * e + i] =
                        scale * tmp[d] * jacptr[e];
                }
            }
        }
    }
}

/**
 * @brief Multiply the input vector by the derivative factors, the Jacobian
 * and the quadrature weights (Phys output path).
 *
 * Identical to JacobianDerivFactorKernel except that each value is
 * additionally weighted by @p weights, the element's quadrature weight at
 * that point. The Phys path needs this because its standard-region matrix
 * (ePhysDerivStdMat) carries no weights.
 *
 * @tparam ExecSpace   Execution space of the calling implementation
 *                     (Serial or AVX); unused in the body.
 * @tparam DEFORMED    True when the geometric factors vary within an
 *                     element.
 * @tparam TData       SIMD vector type of one interleaved element group.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Coordinate dimension: number of input components
 *                      read.
 * @param   dimension   Reference dimension: number of output fields
 *                      written.
 * @param   nelmt       Element groups processed by this call.
 * @param   inoffset    Stride, in vectors, between the input's coordinate
 *                      components.
 * @param   outsize     Stride, in vectors, between the output fields of
 *                      successive reference directions.
 * @param   jacptr      Jacobians of the group.
 * @param   dfptr       Derivative factors of the group.
 * @param   weights     Quadrature weight per point of the element.
 * @param   inptr       Input vector \f${\bf f}\f$.
 * @param   outptr      Receives the @p dimension weighted fields.
 * @param   scale       Factor applied to every value written.
 */
template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void JacobianDerivFactorWeightsKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outsize, const TData *jacptr, const TData *dfptr,
    const TData *weights, const TData *inptr, TData *outptr, const TData scale)
{
    const auto ndf = ncoord * dimension;

    TData tmp[3];
    if constexpr (DEFORMED)
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned i = 0; i < nqTot; i++)
            {
                size_t idx = e * nqTot + i;

                auto wj = jacptr[idx] * weights[i];

                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[ndf * idx + d] * inptr[idx];
                    for (unsigned int k = 1; k < ncoord; ++k)
                    {
                        tmp[d].fma(dfptr[ndf * idx + k * dimension + d],
                                   inptr[k * inoffset + idx]);
                    }
                }
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outsize + idx] = scale * tmp[d] * wj;
                }
            }
        }
    }
    else
    {
        for (size_t e = 0; e < nelmt; e++)
        {
            for (unsigned i = 0; i < nqTot; i++)
            {
                auto wj = jacptr[e] * weights[i];

                for (unsigned d = 0; d < dimension; d++)
                {
                    tmp[d] = dfptr[ndf * e + d] * inptr[nqTot * e + i];
                    for (unsigned k = 1; k < ncoord; ++k)
                    {
                        tmp[d].fma(dfptr[ndf * e + k * dimension + d],
                                   inptr[nqTot * e + i + k * inoffset]);
                    }
                }
                for (unsigned d = 0; d < dimension; d++)
                {
                    outptr[d * outsize + nqTot * e + i] = scale * tmp[d] * wj;
                }
            }
        }
    }
}

} // namespace Nektar::MultiRegions::detail
