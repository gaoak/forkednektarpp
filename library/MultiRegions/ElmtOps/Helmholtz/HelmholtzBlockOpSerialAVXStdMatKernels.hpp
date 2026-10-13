///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzBlockOpSerialAVXStdMatKernels.hpp
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

/**
 * @file HelmholtzBlockOpSerialAVXStdMatKernels.hpp
 * @brief Geometry and diffusion-tensor kernel of the Serial/AVX StdMat
 * weak Helmholtz operator.
 *
 * @details
 * The single kernel below is the third of the five steps described in
 * HelmholtzBlockOpSerialAVXStdMat.hpp: it turns the reference derivatives of
 * the expansion into the quantities the two inner products are taken
 * of, and scales the physical values by the mass term's weight. It is
 * the only place in that implementation where the geometry, the
 * diffusion tensor and lambda appear; the surrounding steps are pure
 * matrix multiplications against standard-region matrices.
 *
 * @see HelmholtzOp.hpp for the metric \f$G\f$ the kernel forms.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Apply the Jacobian, the derivative factors and the diffusion
 * tensor to one group's reference derivatives, and scale its physical
 * values by `lambda * J`.
 *
 * @details
 * With \f$M_{kd} = \partial\xi_d/\partial x_k\f$ the derivative factors
 * and \f$D\f$ the packed symmetric diffusion tensor, the kernel writes,
 * for every quadrature point,
 * \f[ \mathrm{out}_d = J \sum_{e} G_{de}
 *     \frac{\partial u}{\partial \xi_e}, \qquad
 *     G_{de} = \sum_{k,l} M_{kd} D_{kl} M_{le}, \f]
 * so that a subsequent inner product with the derivative of the basis
 * yields the weak Laplacian term, and multiplies @p bwdptr in place by
 * `scale * J`, so that an inner product with the basis yields the mass
 * term; @p bwdptr is left untouched when @p scale is zero. The quadrature
 * weights are not applied here -- the standard-region inner-product matrices of
 * the StdMat implementation already carry them.
 *
 * The two branches differ only in where the geometry is read. On a
 * deformed block the contraction is assembled per quadrature point,
 * one factor at a time (first \f$\partial u/\partial x_k\f$, then
 * \f$\sum_l M_{ld} D_{lk}\f$ against it). On a regular block \f$M\f$
 * and \f$J\f$ are constant within an element, so the full \f$G\f$ is
 * formed once per element and then applied at every point.
 *
 * @p outptr may alias @p inptr -- the block operator passes the same
 * workspace for both, and every point's outputs are computed from that
 * point's inputs before they are written.
 *
 * @tparam ExecSpace  Execution space of the calling implementation;
 *                    part of the signature the call sites share, not
 *                    used in the body.
 * @tparam DEFORMED   Geometry varies within an element.
 * @tparam TData      Value type the arrays are addressed as: one SIMD
 *                    vector per quadrature point, holding that point's
 *                    value for each element of the interleaved group.
 * @tparam TScalar    Scalar type of @p diffCoeff and @p scale.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Coordinate dimension of the element.
 * @param   dimension   Dimension of the reference element.
 * @param   nelmt       Element groups covered by one call; the
 *                      Serial/AVX block operator passes 1.
 * @param   insize      Stride between successive reference-derivative
 *                      directions in @p inptr, in TData units.
 * @param   outsize     Same, for @p outptr.
 * @param   diffCoeff   Packed symmetric diffusion tensor, indexed
 *                      through LibUtilities::GetDiffCoeffMap().
 * @param   jacptr      Jacobian: one value per element group, or one
 *                      per quadrature point when @p DEFORMED.
 * @param   dfptr       Derivative factors, `ncoord * dimension` per
 *                      element group -- or per quadrature point when
 *                      @p DEFORMED -- with `dfptr[k * dimension + d]`
 *                      = \f$M_{kd}\f$.
 * @param   inptr       Reference derivatives of the expansion,
 *                      @p dimension planes of nqTot values.
 * @param   outptr      Receives \f$J G \nabla_\xi u\f$, same layout.
 * @param   bwdptr      Physical values of the expansion, scaled in
 *                      place by `scale * J` unless @p scale is zero.
 * @param   scale       Mass-term weight, i.e. lambda.
 */
template <typename ExecSpace, bool DEFORMED, typename TData, typename TScalar>
NEK_FORCE_INLINE static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t insize,
    const size_t outsize, const TScalar *diffCoeff, const TData *jacptr,
    const TData *dfptr, const TData *inptr, TData *outptr, TData *bwdptr,
    const TScalar scale)
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

            // Write. bwdptr is left unset when scale is zero.
            if (scale != 0.0)
            {
                bwdptr[idx] *= scale * jac;
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
                // bwdptr is left unset when scale is zero.
                if (scale != 0.0)
                {
                    bwdptr[nqTot * e + i] *= scale * jac;
                }
            }
        }
    }
}
} // namespace Nektar::MultiRegions::detail
