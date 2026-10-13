///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionBlockOpDeviceStdMatKernels.hpp
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
// Description: Pointwise metric kernel of the Device StdMat linearised
// advection-diffusion-reaction block operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionBlockOpDeviceStdMatKernels.hpp
 * @brief The pointwise metric kernel of the Device StdMat linearised
 * advection-diffusion-reaction operator.
 *
 * @details
 * The device counterpart of
 * LinAdvDiffReactionBlockOpSerialAVXStdMatKernels.hpp: it does everything the
 * StdMat path does that is not a dense matrix multiply, and sits between
 * the derivative and inner-product GEMMs of
 * detail::LinAdvDiffReactionBlockOpImpl::v_Apply
 * (LinAdvDiffReactionBlockOpDeviceStdMat.hpp), its only caller. Compiled only
 * into device translation units of a device-enabled build
 * (NEKTAR_ENABLE_DEVICE and DEVICE_COMPILE_ONLY).
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Apply the geometry and the three operator coefficients at
 * every quadrature point of the block.
 *
 * @details
 * The variables and planes run together on the second grid dimension:
 * its index \f$c\f$ picks one (variable, plane) slice of
 * `nqTot * nelmt` points, `nqTot * nelmt` values apart in @p in, @p out
 * and @p bwd, over which the first dimension grid-strides. Within a slice
 * the index runs fastest over the points of an element, then over the
 * elements, so it addresses the geometry directly. The velocity is shared
 * by the variables but not by the planes, and is read on plane
 * `c % nhomo`.
 *
 * Reading the reference-space derivatives
 * \f$\partial u/\partial\xi_d\f$ from @p in and the physical values
 * \f$u\f$ from @p bwd, each thread produces
 * - in @p bwd, the integrand of the advection and reaction terms,
 *   \f$J\,(\mathbf{V}\cdot\nabla u + \mathrm{scale}\cdot u)\f$, where the
 *   Cartesian derivative is recovered by the chain rule
 *   \f$\partial u/\partial x_k = \sum_d (\partial\xi_d/\partial x_k)\,
 *   \partial u/\partial\xi_d\f$ from @p dfptr; @p bwd is not read when
 *   @p scale is zero;
 * - in @p out, one component per reference direction \f$d\f$, the
 *   diffusion flux pulled back to that direction,
 *   \f$J \sum_{l,k} (\partial\xi_d/\partial x_l)\, D_{lk}\,
 *   \partial u/\partial x_k\f$.
 *
 * @p out may alias @p in, as it does in the only caller: a thread
 * reads all of @p in's components for its point before writing any
 * of @p out's. The fixed-size local arrays limit both @p ncoord and
 * @p dimension to three.
 *
 * @tparam DEFORMED        Geometry is deformed, so the Jacobian and the
 *                         derivative factors vary point by point and the
 *                         factors are laid out factor-major within each
 *                         element.
 * @tparam TthreadBlock    Back-end thread-block handle type.
 * @tparam TData           Floating-point type of the field data.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Coordinate dimension: velocity components and
 *                      rank of the diffusivity tensor.
 * @param   dimension   Reference-element dimension: number of
 *                      derivative components in @p in and of flux
 *                      components in @p out.
 * @param   nelmt       Elements in the block, padding included.
 * @param   nhomo       Homogeneous planes carried by the field.
 * @param   inoffset    Stride between the derivative components of
 *                      @p in.
 * @param   outoffset   Stride between the flux components of
 *                      @p out.
 * @param   adveloffset Stride between the velocity components of
 *                      @p advVel.
 * @param   diffCoeff   Packed lower triangle of the symmetric
 *                      diffusivity tensor, indexed through
 *                      LibUtilities::GetDiffCoeffMap (see
 *                      LinAdvDiffReactionOp::SetDiffCoeff).
 * @param   jacptr      Jacobians: nqTot values per element when
 *                      deformed, one per element otherwise.
 * @param   dfptr       Derivative factors: when deformed, per element a
 *                      block of `ncoord * dimension` runs of nqTot
 *                      values, the run for coordinate k and reference
 *                      direction d being at `k * dimension + d`; when
 *                      regular, one such value per element instead of a
 *                      run.
 * @param   advVel      Advection velocity at the quadrature points,
 *                      @p ncoord components, each spanning every plane.
 * @param   in          Reference-space derivatives of \f$u\f$,
 *                      @p dimension components.
 * @param   out         Diffusion flux out, @p dimension components.
 * @param[in,out] bwd   In: \f$u\f$ at the quadrature points; out:
 *                      the advection-plus-reaction integrand.
 * @param   scale       Reaction coefficient \f$\lambda\f$.
 * @param   threadBlock Back-end thread-block handle supplying the
 *                      global thread index and grid stride.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void ApplyMetricKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const unsigned int nhomo,
    const size_t inoffset, const size_t outoffset, const size_t adveloffset,
    const TData *diffCoeff, const TData *jacptr, const TData *dfptr,
    const TData *advVel, const TData *in, TData *out, TData *bwd,
    const TData scale, const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx             = getGlobalIdx<0>(threadBlock);
    const unsigned int c   = getBlockIdx<1>(threadBlock);
    const TData *inptr     = in + nsize * c;
    TData *outptr          = out + nsize * c;
    TData *bwdptr          = bwd + nsize * c;
    const TData *advVelPtr = advVel + nsize * (c % nhomo);
    while (idx < nsize)
    {
        if constexpr (DEFORMED)
        {
            size_t e     = idx / nqTot;
            size_t i     = idx - e * nqTot;
            TData tmp[3] = {0.0}, tmp0 = 0.0, metric[3];

            auto jac = jacptr[idx];

            tmp0 = 0.0;
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
                tmp0 += advVelPtr[k * adveloffset + idx] * tmp[k];
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
            TData tmp[3] = {0.0}, tmp0 = 0.0, metric[3];

            auto jac = jacptr[e];

            tmp0 = 0.0;
            for (unsigned int k = 0; k < ncoord; ++k)
            {
                tmp[k] = dfptr[ndf * e + k * dimension] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[ndf * e + k * dimension + d] *
                              inptr[idx + d * inoffset];
                }
                tmp0 += advVelPtr[k * adveloffset + idx] * tmp[k];
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
                        dfptr[(ndf * e + d)] *
                        diffCoeff[LibUtilities::GetDiffCoeffMap(ncoord, k)];
                    for (unsigned int l = 1; l < ncoord; ++l)
                    {
                        metric[k] += dfptr[(ndf * e + l * dimension + d)] *
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

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::MultiRegions::detail
