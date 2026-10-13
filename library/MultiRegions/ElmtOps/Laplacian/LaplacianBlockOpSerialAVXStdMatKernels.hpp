///////////////////////////////////////////////////////////////////////////////
//
// File: LaplacianBlockOpSerialAVXStdMatKernels.hpp
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
// Description: Metric kernel of the Serial/AVX StdMat weak Laplacian:
// turns reference derivatives into metric-contracted quantities.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LaplacianBlockOpSerialAVXStdMatKernels.hpp
 * @brief Metric kernel of the Serial/AVX StdMat weak Laplacian: turns
 * reference derivatives into the metric-contracted, Jacobian-weighted
 * quantities the inner product is taken against.
 *
 * @details
 * This is the middle step of the three described in
 * LaplacianBlockOpSerialAVXStdMat.hpp; the outer two are dense matrix
 * multiplies issued by the block operator itself, so this header holds
 * a single kernel.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Apply \f$J\,G = J\,M^T D M\f$ to the reference derivatives of
 * an element group.
 *
 * @details
 * With @p inptr holding the reference derivatives
 * \f$\partial u/\partial\xi_e\f$ (direction e at offset
 * `e * insize`), the kernel writes to @p outptr, at offset
 * `d * outsize`,
 * \f[ J \sum_{k,l,e} M_{kd} \, D_{kl} \, M_{le} \,
 *     \frac{\partial u}{\partial\xi_e}, \f]
 * the quantity LaplacianOp.hpp derives; in place is allowed and is how
 * the block operator calls it. The two branches differ only in where
 * the geometry is constant:
 * - DEFORMED: the factors vary point by point, so per point the
 *   physical gradient `tmp[k]` is formed first and then contracted with
 *   the row \f$\sum_l M_{ld} D_{lk}\f$ -- one pass over the points, no
 *   metric stored;
 * - regular: the full symmetric metric `sum[d][k]` is built once per
 *   element from the element's single set of factors and then applied
 *   to every point.
 *
 * The diffusion tensor is read through GetDiffCoeffMap(), which maps a
 * full index pair onto the packed storage described in
 * LaplacianOp::SetDiffCoeff.
 *
 * @tparam ExecSpace  Execution space; unused, present for uniformity
 *                    with the other kernels.
 * @tparam DEFORMED   Block geometry is deformed.
 * @tparam TData      Value type the kernel computes in: the SIMD vector
 *                    type of the interleaved element group, one lane
 *                    per element.
 * @tparam TScalar    Scalar type of the diffusion tensor entries.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Coordinate dimension: the range of the physical
 *                      indices k, l.
 * @param   dimension   Reference dimension: the range of d, e.
 * @param   nelmt       Element groups covered by one call (1 from the
 *                      block operator).
 * @param   insize      Stride between reference directions in
 *                      @p inptr, in SIMD vectors.
 * @param   outsize     Stride between reference directions in
 *                      @p outptr.
 * @param   diffCoeff   Packed symmetric diffusion tensor.
 * @param   jacptr      Jacobians: one value per element group when
 *                      regular, one per quadrature point when deformed.
 * @param   dfptr       Derivative factors in point-major order, as
 *                      described in the block operator's @c m_dfptr.
 * @param   inptr       Reference derivatives of the group.
 * @param   outptr      Destination; may alias @p inptr.
 */
template <typename ExecSpace, bool DEFORMED, typename TData, typename TScalar>
NEK_FORCE_INLINE static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t insize,
    const size_t outsize, const TScalar *diffCoeff, const TData *jacptr,
    const TData *dfptr, const TData *inptr, TData *outptr)
{
    const auto ndf   = ncoord * dimension;
    const auto nsize = nqTot * nelmt;

    if constexpr (DEFORMED)
    {
        TData tmp[3], tmp0, metric[3];
        for (size_t idx = 0; idx < nsize; idx++)
        {
            auto jac = jacptr[idx];

            for (unsigned int k = 0; k < ncoord; ++k)
            {
                tmp[k] = dfptr[ndf * idx + k * dimension] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k].fma(dfptr[ndf * idx + k * dimension + d],
                               inptr[d * insize + idx]);
                }
            }

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
        TData tmp[3], sum[9], metric[3];
        for (size_t e = 0; e < nelmt; e++)
        {
            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[k] =
                        dfptr[ndf * e + d] *
                        diffCoeff[LibUtilities::GetDiffCoeffMap(ncoord, k)];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k].fma(dfptr[ndf * e + l * dimension + d],
                                      diffCoeff[LibUtilities::GetDiffCoeffMap(
                                          ncoord, l * ncoord + k)]);
                    }
                }
                for (unsigned int k = 0; k < dimension; ++k)
                {
                    sum[d * dimension + k] = metric[0] * dfptr[ndf * e + k];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        sum[d * dimension + k].fma(
                            metric[l], dfptr[ndf * e + l * dimension + k]);
                    }
                }
            }

            auto jac = jacptr[e];
            for (unsigned int i = 0; i < nqTot; i++)
            {
                for (unsigned int d = 0; d < dimension; d++)
                {
                    tmp[d] = sum[d * dimension] * inptr[nqTot * e + i];
                    for (unsigned int k = 1; k < dimension; k++)
                    {
                        tmp[d].fma(sum[d * dimension + k],
                                   inptr[nqTot * e + i + k * insize]);
                    }
                }

                // Write.
                for (unsigned int d = 0; d < dimension; d++)
                {
                    outptr[d * outsize + nqTot * e + i] = tmp[d] * jac;
                }
            }
        }
    }
}
} // namespace Nektar::MultiRegions::detail
