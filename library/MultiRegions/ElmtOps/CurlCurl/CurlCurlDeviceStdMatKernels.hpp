///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlDeviceStdMatKernels.hpp
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
 * @file CurlCurlDeviceStdMatKernels.hpp
 * @brief Pointwise curl kernels of the Device StdMat curl-curl block
 * operator (see CurlCurlDeviceStdMat.hpp).
 *
 * @details
 * These kernels chain-rule reference-space derivatives into Cartesian ones
 * through the derivative factors and combine them into a curl. They run as
 * device kernels over all quadrature points of the block, launched by the
 * block operator on its device stream, and they read the derivative planes
 * the batched GEMM lays out direction-major. The 2D kernels take the
 * homogeneous planes of a 3DH1 block on the second grid dimension of the
 * same launch.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Scalar curl of a two-dimensional vector field,
 * omega = d(u1)/dx0 - d(u0)/dx1, at every quadrature point of every
 * element of the block.
 *
 * The derivative-factor entry k * 2 + d maps the reference-direction-d
 * derivative into the Cartesian-k gradient component. The factors are read
 * in the layout produced by DerivFactorKey with the transpose flag set: for
 * regular geometry one set of four values per element, for deformed
 * geometry one such set per element in which each factor component is
 * contiguous over the element's nqTot points. The chain rule and the curl
 * are applied in a single sweep. The second grid dimension picks the
 * homogeneous plane, nqTot * nelmt values apart in @p deriv and @p omega.
 *
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   nelmt       Elements in the block, including padding.
 * @param   derivoffset Stride between the planes of @p deriv,
 *                      nqTot * nelmt times the number of homogeneous
 *                      planes.
 * @param   dfptr       Derivative factors of the block, laid out as
 *                      described above.
 * @param   deriv       Reference-space derivatives of the two components,
 *                      ordered as deriv[(d * 2 + c) * derivoffset + idx]
 *                      for direction d and component c.
 * @param   omega       Output plane of nqTot * nelmt values.
 * @param   threadBlock Thread-block handle supplied by the launcher
 *                      macro.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void Curl2DScalarStdMatKernel(
    const unsigned int nqTot, const size_t nelmt, const size_t derivoffset,
    const TData *dfptr, const TData *deriv, TData *omega,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 4u;
    const size_t nsize         = nqTot * nelmt;

    size_t idx            = getGlobalIdx<0>(threadBlock);
    const unsigned int p  = getBlockIdx<1>(threadBlock);
    const TData *derivptr = deriv + nsize * p;
    TData *omegaptr       = omega + nsize * p;
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        const size_t dfbase =
            DEFORMED ? ndf * nqTot * e + (idx - e * nqTot) : ndf * e;
        const unsigned int dfstride = DEFORMED ? nqTot : 1u;

        const TData d0u = derivptr[0u * derivoffset + idx];
        const TData d0v = derivptr[1u * derivoffset + idx];
        const TData d1u = derivptr[2u * derivoffset + idx];
        const TData d1v = derivptr[3u * derivoffset + idx];

        const TData dvdx = d0v * dfptr[dfbase + 0u * dfstride] +
                           d1v * dfptr[dfbase + 1u * dfstride];
        const TData dudy = d0u * dfptr[dfbase + 2u * dfstride] +
                           d1u * dfptr[dfbase + 3u * dfstride];

        omegaptr[idx] = dvdx - dudy;

        idx += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Vector curl of a two-dimensional scalar field,
 * out = ( d(omega)/dx1, -d(omega)/dx0 ), at every quadrature point of
 * every element of the block.
 *
 * The derivative factors are read as in Curl2DScalarStdMatKernel; both
 * output components are written by the same work item. The second grid
 * dimension picks the homogeneous plane, nqTot * nelmt values apart in
 * @p deriv and @p out.
 *
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   nelmt       Elements in the block, including padding.
 * @param   derivoffset Stride between the planes of @p deriv.
 * @param   outoffset   Stride between the components of @p out.
 * @param   dfptr       Derivative factors of the block.
 * @param   deriv       Reference-space derivatives of omega, ordered as
 *                      deriv[d * derivoffset + idx] for direction d.
 * @param   out         Output, two components of nqTot * nelmt values.
 * @param   threadBlock Thread-block handle supplied by the launcher
 *                      macro.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void Curl2DVectorStdMatKernel(
    const unsigned int nqTot, const size_t nelmt, const size_t derivoffset,
    const size_t outoffset, const TData *dfptr, const TData *deriv, TData *out,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 4u;
    const size_t nsize         = nqTot * nelmt;

    size_t idx            = getGlobalIdx<0>(threadBlock);
    const unsigned int p  = getBlockIdx<1>(threadBlock);
    const TData *derivptr = deriv + nsize * p;
    TData *outptr         = out + nsize * p;
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        const size_t dfbase =
            DEFORMED ? ndf * nqTot * e + (idx - e * nqTot) : ndf * e;
        const unsigned int dfstride = DEFORMED ? nqTot : 1u;

        const TData d0 = derivptr[0u * derivoffset + idx];
        const TData d1 = derivptr[1u * derivoffset + idx];

        outptr[idx] = d0 * dfptr[dfbase + 2u * dfstride] +
                      d1 * dfptr[dfbase + 3u * dfstride];
        outptr[outoffset + idx] = -(d0 * dfptr[dfbase + 0u * dfstride] +
                                    d1 * dfptr[dfbase + 1u * dfstride]);

        idx += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Curl of a three-dimensional vector field at every quadrature
 * point of every element of the block; used for both passes of the
 * curl-curl operator, omega = curl u and out = curl omega.
 *
 * The derivative-factor entry k * 3 + d maps the reference-direction-d
 * derivative into the Cartesian-k gradient component, the nine factors
 * being read as in Curl2DScalarStdMatKernel. Only the six off-diagonal
 * Cartesian derivatives are formed, the diagonal ones cancelling out of
 * the curl:
 *     out0 = d(u2)/dx1 - d(u1)/dx2,
 *     out1 = d(u0)/dx2 - d(u2)/dx0,
 *     out2 = d(u1)/dx0 - d(u0)/dx1.
 *
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   nelmt       Elements in the block, including padding.
 * @param   derivoffset Stride between the planes of @p deriv.
 * @param   outoffset   Stride between the components of @p out: the plane
 *                      stride of the workspace for the omega pass, the
 *                      component size of the output block for the final
 *                      pass.
 * @param   dfptr       Derivative factors of the block.
 * @param   deriv       Reference-space derivatives of the three
 *                      components, ordered as
 *                      deriv[(d * 3 + c) * derivoffset + idx] for
 *                      direction d and component c.
 * @param   out         Output, three components of nqTot * nelmt values.
 * @param   threadBlock Thread-block handle supplied by the launcher
 *                      macro.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void Curl3DStdMatKernel(
    const unsigned int nqTot, const size_t nelmt, const size_t derivoffset,
    const size_t outoffset, const TData *dfptr, const TData *deriv, TData *out,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const size_t nsize         = nqTot * nelmt;

    size_t idx = getGlobalIdx<0>(threadBlock);
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        const size_t dfbase =
            DEFORMED ? ndf * nqTot * e + (idx - e * nqTot) : ndf * e;
        const unsigned int dfstride = DEFORMED ? nqTot : 1u;

        TData d0[3], d1[3], d2[3];
        for (unsigned int c = 0u; c < 3u; ++c)
        {
            d0[c] = deriv[(0u * 3u + c) * derivoffset + idx];
            d1[c] = deriv[(1u * 3u + c) * derivoffset + idx];
            d2[c] = deriv[(2u * 3u + c) * derivoffset + idx];
        }

        TData df[9];
        for (unsigned int m = 0u; m < 9u; ++m)
        {
            df[m] = dfptr[dfbase + m * dfstride];
        }

        // dw/dy - dv/dz
        out[idx] = (d0[2] * df[3] + d1[2] * df[4] + d2[2] * df[5]) -
                   (d0[1] * df[6] + d1[1] * df[7] + d2[1] * df[8]);
        // du/dz - dw/dx
        out[outoffset + idx] = (d0[0] * df[6] + d1[0] * df[7] + d2[0] * df[8]) -
                               (d0[2] * df[0] + d1[2] * df[1] + d2[2] * df[2]);
        // dv/dx - du/dy
        out[2u * outoffset + idx] =
            (d0[1] * df[0] + d1[1] * df[1] + d2[1] * df[2]) -
            (d0[0] * df[3] + d1[0] * df[4] + d2[0] * df[5]);

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::MultiRegions::detail
