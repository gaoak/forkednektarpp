///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionBlockOpSerialAVXStdMatKernels.hpp
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
// Description: Pointwise metric kernel of the Serial/AVX StdMat linearised
// advection-diffusion-reaction block operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionBlockOpSerialAVXStdMatKernels.hpp
 * @brief The pointwise metric kernel of the Serial/AVX StdMat linearised
 * advection-diffusion-reaction operator.
 *
 * @details
 * Everything the StdMat path does that is not a dense matrix multiply
 * happens here: the chain rule from reference to Cartesian derivatives,
 * the advection and reaction terms, the diffusivity tensor, and the
 * Jacobian weighting. The kernel sits between the two derivative and
 * backward-transform multiplies and the two inner-product multiplies of
 * detail::LinAdvDiffReactionBlockOpImpl::v_Apply
 * (LinAdvDiffReactionBlockOpSerialAVXStdMat.hpp), which is its only caller.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Apply the geometry and the three operator coefficients at
 * every quadrature point of an interleaved group of elements.
 *
 * @details
 * Reading the reference-space derivatives
 * \f$\partial u/\partial\xi_d\f$ from @p inptr and the physical values
 * \f$u\f$ from @p bwdptr, the kernel produces
 * - in @p bwdptr, the integrand of the advection and reaction terms,
 *   \f$J\,(\mathbf{V}\cdot\nabla u + \mathrm{scale}\cdot u)\f$, where the
 *   Cartesian derivative is recovered by the chain rule
 *   \f$\partial u/\partial x_k = \sum_d (\partial\xi_d/\partial x_k)\,
 *   \partial u/\partial\xi_d\f$ from @p dfptr; @p bwdptr is not read
 *   when @p scale is zero;
 * - in @p outptr, one component per reference direction \f$d\f$, the
 *   diffusion flux pulled back to that direction,
 *   \f$J \sum_{l,k} (\partial\xi_d/\partial x_l)\, D_{lk}\,
 *   \partial u/\partial x_k\f$, ready for the inner product with the
 *   basis gradient.
 *
 * @p outptr may alias @p inptr, as it does in the only caller: for a
 * given point all of @p inptr's components are read into the local
 * Cartesian derivatives before any component of @p outptr is written.
 * When the geometry is regular the reference-space diffusion metric is
 * constant over an element and is computed once, at the element's first
 * point; when it is deformed it is recomputed per point. The fixed-size
 * local arrays limit both @p ncoord and @p dimension to three.
 *
 * @tparam ExecSpace   Execution space of the calling block operator;
 *                     not used by the body.
 * @tparam DEFORMED    Geometry is deformed, so the Jacobian and the
 *                     derivative factors vary point by point.
 * @tparam TData       SIMD vector type of the interleaved group (a
 *                     scalar type on the Serial space).
 * @tparam TScalar     Scalar element type of @p TData.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Coordinate dimension: velocity components and
 *                      rank of the diffusivity tensor.
 * @param   dimension   Reference-element dimension: number of
 *                      derivative components in @p inptr and of flux
 *                      components in @p outptr.
 * @param   nelmt       Element groups covered by this call (one, in the
 *                      current caller).
 * @param   insize      Stride between the derivative components of
 *                      @p inptr, in vectors.
 * @param   outsize     Stride between the flux components of
 *                      @p outptr, in vectors.
 * @param   advelsize   Stride between the velocity components of
 *                      @p advVel, in vectors.
 * @param   diffCoeff   Packed lower triangle of the symmetric
 *                      diffusivity tensor, indexed through
 *                      LibUtilities::GetDiffCoeffMap (see
 *                      LinAdvDiffReactionOp::SetDiffCoeff).
 * @param   jacptr      Jacobians: one value per point when deformed,
 *                      one per element otherwise.
 * @param   dfptr       Derivative factors in point-major order, entry
 *                      \f$\partial\xi_d/\partial x_k\f$ at
 *                      `k * dimension + d` within each point's (or, for
 *                      regular geometry, each element's) block of
 *                      `ncoord * dimension` values.
 * @param   advVel      Advection velocity at the quadrature points,
 *                      @p ncoord components.
 * @param   inptr       Reference-space derivatives of \f$u\f$,
 *                      @p dimension components.
 * @param   outptr      Diffusion flux out, @p dimension components.
 * @param[in,out] bwdptr    In: \f$u\f$ at the quadrature points; out:
 *                          the advection-plus-reaction integrand.
 * @param   scale       Reaction coefficient \f$\lambda\f$.
 */
template <typename ExecSpace, bool DEFORMED, typename TData, typename TScalar>
NEK_FORCE_INLINE static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t insize,
    const size_t outsize, const size_t advelsize, const TScalar *diffCoeff,
    const TData *jacptr, const TData *dfptr, const TData *advVel,
    const TData *inptr, TData *outptr, TData *bwdptr, const TScalar scale)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    if constexpr (DEFORMED)
    {
        TData tmp[3], tmp0, metric[3];
        for (size_t idx = 0; idx < nsize; idx++)
        {
            auto jac = jacptr[idx];

            tmp0 = 0.0;
            for (unsigned int k = 0; k < ncoord; ++k)
            {
                tmp[k] = dfptr[ndf * idx + k * dimension] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k].fma(dfptr[ndf * idx + k * dimension + d],
                               inptr[d * insize + idx]);
                }
                tmp0.fma(advVel[k * advelsize + idx], tmp[k]);
            }

            // Write. bwdptr is left unset when scale is zero.
            if (scale != 0.0)
            {
                tmp0 += scale * bwdptr[idx];
            }
            bwdptr[idx] = tmp0 * jac;

            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[k] =
                        dfptr[ndf * idx + d] *
                        diffCoeff[LibUtilities::GetDiffCoeffMap(ncoord, k)];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k].fma(dfptr[ndf * idx + l * dimension + d],
                                      diffCoeff[LibUtilities::GetDiffCoeffMap(
                                          ncoord, l * ncoord + k)]);
                    }
                }
                tmp0 = metric[0] * tmp[0];
                for (unsigned int k = 1; k < ncoord; k++)
                {
                    tmp0.fma(metric[k], tmp[k]);
                }

                // Write.
                outptr[d * outsize + idx] = tmp0 * jac;
            }
        }
    }
    else
    {
        TData tmp[3], tmp0, metric[9];
        for (size_t e = 0; e < nelmt; e++)
        {
            auto jac = jacptr[e];
            for (unsigned int i = 0; i < nqTot; i++)
            {
                auto idx = nqTot * e + i;
                tmp0     = 0.0;
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    tmp[k] = dfptr[ndf * e + k * dimension] * inptr[idx];
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        tmp[k].fma(dfptr[ndf * e + k * dimension + d],
                                   inptr[d * insize + idx]);
                    }
                    tmp0.fma(advVel[k * advelsize + nqTot * e + i], tmp[k]);
                }

                // Write. bwdptr is left unset when scale is zero.
                if (scale != 0.0)
                {
                    tmp0 += scale * bwdptr[idx];
                }
                bwdptr[idx] = tmp0 * jac;

                // Compute metric.
                for (unsigned int d = 0; d < dimension; d++)
                {
                    if (i == 0)
                    {
                        for (unsigned int k = 0; k < ncoord; ++k)
                        {
                            metric[d * ncoord + k] =
                                dfptr[ndf * e + d] *
                                diffCoeff[LibUtilities::GetDiffCoeffMap(ncoord,
                                                                        k)];
                            for (unsigned int l = 1; l < ncoord; ++l)
                            {
                                metric[d * ncoord + k].fma(
                                    dfptr[ndf * e + l * dimension + d],
                                    diffCoeff[LibUtilities::GetDiffCoeffMap(
                                        ncoord, l * ncoord + k)]);
                            }
                        }
                    }
                    tmp0 = metric[d * ncoord] * tmp[0];
                    for (unsigned int k = 1; k < ncoord; k++)
                    {
                        tmp0.fma(metric[d * ncoord + k], tmp[k]);
                    }

                    // Write.
                    outptr[d * outsize + idx] = tmp0 * jac;
                }
            }
        }
    }
}

} // namespace Nektar::MultiRegions::detail
