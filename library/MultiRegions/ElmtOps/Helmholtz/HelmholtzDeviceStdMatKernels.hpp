///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzDeviceStdMatKernels.hpp
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
 * @file HelmholtzDeviceStdMatKernels.hpp
 * @brief Geometry and diffusion-tensor kernel of the device StdMat weak
 * Helmholtz operator.
 *
 * @details
 * The third step of the five-stage decomposition (see
 * HelmholtzDeviceStdMat.hpp), expressed as a grid-strided kernel over
 * the quadrature points of the block, one variable and plane per index
 * of the second grid dimension. It is compiled only into device
 * translation units of a device-enabled build (NEKTAR_ENABLE_DEVICE and
 * DEVICE_COMPILE_ONLY).
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Apply the Jacobian, the derivative factors and the diffusion
 * tensor to the block's reference derivatives, and scale its physical
 * values by `scale * J`.
 *
 * @details
 * One thread handles one quadrature point of one element, for one
 * component-and-homogeneous-mode plane, striding over the plane by the
 * grid size, and writes for its point
 * \f[ \mathrm{out}_d = J \sum_e G_{de}
 *     \frac{\partial u}{\partial\xi_e}, \qquad
 *     G_{de} = \sum_{k,l} M_{kd} D_{kl} M_{le}, \f]
 * as well as `bwd *= scale * J`, which is skipped when @p scale is zero;
 * see HelmholtzSerialAVXStdMatKernels.hpp for the meaning of the two terms
 * and HelmholtzOp.hpp for the metric. The quadrature weights are
 * carried by the standard-region inner-product matrices, not applied
 * here.
 *
 * The variables and planes run together on the second grid dimension:
 * block `c` there works on the `c`-th set of `nqTot * nelmt` values of
 * @p in, @p out and @p bwd, so every plane reuses the same element
 * geometry. On a regular block the metric is formed from the element's
 * constant derivative factors; on a deformed block it is formed from the
 * point's own, indexed with the transposed layout this path requests from
 * DerivFactorKey (factor `k * dimension + d` of point `i` at
 * `ndf * nqTot * e + (k * dimension + d) * nqTot + i`).
 *
 * @p out may alias @p in -- the block operator passes the same
 * workspace for both, and a thread reads all of its point's inputs
 * before writing its outputs.
 *
 * @tparam DEFORMED      Geometry varies within an element.
 * @tparam TthreadBlock  Back-end thread-block handle supplying the
 *                       global index and range.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Coordinate dimension of the element.
 * @param   dimension   Dimension of the reference element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   inoffset    Stride between successive reference-derivative
 *                      directions in @p in.
 * @param   outoffset   Same, for @p out.
 * @param   diffCoeff   Packed symmetric diffusion tensor, indexed
 *                      through LibUtilities::GetDiffCoeffMap().
 * @param   jacptr      Jacobian: one value per element, or one per
 *                      quadrature point when @p DEFORMED.
 * @param   dfptr       Derivative factors in the transposed layout
 *                      described above, with `dfptr[k * dimension + d]`
 *                      = \f$M_{kd} = \partial\xi_d/\partial x_k\f$ per
 *                      point.
 * @param   in          Reference derivatives of the expansion,
 *                      @p dimension directions @p inoffset apart, each
 *                      holding `nqTot * nelmt` values per plane.
 * @param   out         Receives \f$J G \nabla_\xi u\f$, same layout.
 * @param   bwd         Physical values of the expansion, scaled in
 *                      place by `scale * J` unless @p scale is zero.
 * @param   scale       Mass-term weight, i.e. lambda.
 * @param   threadBlock Back-end thread-block handle.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outoffset, const TData *diffCoeff, const TData *jacptr,
    const TData *dfptr, const TData *in, TData *out, TData *bwd,
    const TData scale, const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx           = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    const TData *inptr   = in + nsize * c;
    TData *outptr        = out + nsize * c;
    TData *bwdptr        = bwd + nsize * c;
    while (idx < nsize)
    {
        if constexpr (DEFORMED)
        {
            size_t e     = idx / nqTot;
            size_t i     = idx - e * nqTot;
            TData tmp[3] = {0.0}, tmp0 = 0.0, metric[3];

            auto jac = jacptr[idx];

            for (unsigned int k = 0; k < ncoord; ++k)
            {
                tmp[k] = dfptr[ndf * nqTot * e + (k * dimension) * nqTot + i] *
                         inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[ndf * nqTot * e +
                                    (k * dimension + d) * nqTot + i] *
                              inptr[idx + d * inoffset];
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
                        dfptr[ndf * nqTot * e + d * nqTot + i] *
                        diffCoeff[LibUtilities::GetDiffCoeffMap(ncoord, k)];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k] += dfptr[ndf * nqTot * e +
                                           (l * dimension + d) * nqTot + i] *
                                     diffCoeff[LibUtilities::GetDiffCoeffMap(
                                         ncoord, l * ncoord + k)];
                    }
                }
                tmp0 = metric[0] * tmp[0];
                for (unsigned int k = 1; k < ncoord; k++)
                {
                    tmp0 += metric[k] * tmp[k];
                }

                // Write.
                outptr[d * outoffset + idx] = tmp0 * jac;
            }
        }
        else
        {
            size_t e     = idx / nqTot;
            TData tmp[3] = {0.0}, metric[3];

            // Compute metric.
            for (unsigned int d = 0; d < dimension; d++)
            {
                for (unsigned int k = 0; k < ncoord; ++k)
                {
                    metric[k] =
                        dfptr[(ndf * e + d)] *
                        diffCoeff[LibUtilities::GetDiffCoeffMap(ncoord, k)];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k] += dfptr[(ndf * e + l * dimension + d)] *
                                     diffCoeff[LibUtilities::GetDiffCoeffMap(
                                         ncoord, l * ncoord + k)];
                    }
                }
                for (unsigned int k = 0; k < dimension; ++k)
                {
                    TData sum = 0.0;
                    for (unsigned int l = 0; l < ncoord; ++l)
                    {
                        sum += metric[l] * dfptr[ndf * e + l * dimension + k];
                    }
                    tmp[d] += sum * inptr[idx + k * inoffset];
                }
            }

            // Write.
            auto jac = jacptr[e];
            for (unsigned int d = 0; d < dimension; d++)
            {
                outptr[d * outoffset + idx] = tmp[d] * jac;
            }
            // bwdptr is left unset when scale is zero.
            if (scale != 0.0)
            {
                bwdptr[idx] *= scale * jac;
            }
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif
} // namespace Nektar::MultiRegions::detail
