///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceStdMatKernels.hpp
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
// Description: Chain-rule kernels of the device StdMat physical
// derivative.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivDeviceStdMatKernels.hpp
 * @brief Chain-rule kernels of the device StdMat physical derivative:
 * contract standard-region derivatives with the element's geometric
 * derivative factors, one work item per point of the block.
 *
 * @details
 * These are the device counterparts of the kernels in
 * PhysDerivSerialAVXStdMatKernels.hpp and perform the same chain rule
 * \f[ \frac{\partial u}{\partial x_k}
 *     = \sum_{d} \frac{\partial \xi_d}{\partial x_k}
 *       \frac{\partial u}{\partial \xi_d}, \f]
 * but over the whole block at once: one work item per (element, point)
 * pair on the first grid dimension, the homogeneous planes (and, for
 * MultiplyByDerivFactorKernel, the components) on the further grid
 * dimensions of the same launch.
 *
 * The derivative factors are laid out transposed with respect to the
 * Serial/AVX ones (DerivFactorKey with `transpose = true`, see
 * GeometricDataWarehouse.hpp): per element the `ndf = coordDim *
 * dimension` factors form contiguous planes, of one value each for a
 * regular element and of nqTot values each for a deformed one. Hence
 * the two branches index differently, the deformed one reaching factor
 * `k * dimension + d` of element @em e at
 * `ndf * nqTot * e + nqTot * (k * dimension + d) + i` -- written as
 * `(ndf - 1) * nqTot * e + ...` because the within-plane flat index
 * added last, `idx`, already carries one `nqTot * e`.
 *
 * MultiplyByDerivFactorKernel produces all `coordDim` components at once
 * and is used by PhysDeriv itself (and by the AdvectionDealias StdMat
 * implementation); MultiplyByDerivDirFactorKernel produces one chosen
 * component and is used by the Divergence StdMat implementation.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Contract the standard-region derivatives of a whole block into
 * all @p ncoord Cartesian derivatives.
 *
 * Safe to run in place (@p out equal to @p in, as PhysDeriv does): every
 * work item forms all @p ncoord results in registers before storing any
 * of them.
 *
 * The second grid dimension picks the homogeneous plane and the third the
 * component. Each component holds @p outDim slots of one plane per
 * homogeneous plane, so the pair selects the slot
 * `outDim * nhomo * c + m`, `nqTot * nelmt` values apart, in both @p in
 * and @p out; the same geometry serves every plane.
 *
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam TthreadBlock Back-end thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Cartesian components to produce (coordDim); at
 *                      most 3, the size of the per-point register
 *                      buffer.
 * @param   dimension   Standard directions to contract over.
 * @param   outDim      Direction slots per component: coordDim, or 3 in
 *                      3DH1, the third left to the z-derivative.
 * @param   nelmt       Elements in the block, including padding.
 * @param   inoffset    Stride between the standard-direction planes of
 *                      @p in.
 * @param   outoffset   Stride between the Cartesian planes of @p out.
 * @param   dfptr       Derivative factors of the block, transposed
 *                      layout.
 * @param   in          Standard-direction derivatives, plane @em d at
 *                      `d * inoffset`.
 * @param   out         Cartesian derivatives, plane @em k written at
 *                      `k * outoffset`.
 * @param   threadBlock Thread-block handle supplied by the launcher macro.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void MultiplyByDerivFactorKernel(
    const unsigned nqTot, const unsigned ncoord, const unsigned dimension,
    const unsigned outDim, const size_t nelmt, const size_t inoffset,
    const size_t outoffset, const TData *dfptr, const TData *in, TData *out,
    const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx               = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const TData *inptr       = in + nsize * (outDim * nmode * c + m);
    TData *outptr            = out + nsize * (outDim * nmode * c + m);
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData tmp[3];
        if constexpr (DEFORMED)
        {
            for (unsigned int k = 0; k < ncoord; k++)
            {
                tmp[k] = dfptr[(ndf - 1) * nqTot * e + nqTot * (k * dimension) +
                               idx] *
                         inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[(ndf - 1) * nqTot * e +
                                    nqTot * (k * dimension + d) + idx] *
                              inptr[idx + d * inoffset];
                }
            }
        }
        else
        {
            for (unsigned int k = 0; k < ncoord; k++)
            {
                tmp[k] = dfptr[(ndf * e + k * dimension)] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[(ndf * e + k * dimension + d)] *
                              inptr[idx + d * inoffset];
                }
            }
        }

        for (unsigned int k = 0; k < ncoord; k++)
        {
            outptr[k * outoffset + idx] = tmp[k];
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Contract the standard-region derivatives of a whole block into
 * the single Cartesian derivative @p dir, overwriting or accumulating.
 *
 * The runtime @p dir argument plays the role that the DIR template
 * parameter plays in the sum-factorised kernels: it selects the
 * derivative-factor rows `dir * dimension + d`. The second grid dimension
 * picks the homogeneous plane, `nqTot * nelmt` values apart in both @p in
 * and @p out; the same geometry serves every plane.
 *
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam TthreadBlock Back-end thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   dir         Cartesian component to produce.
 * @param   nqTot       Quadrature points per element.
 * @param   ncoord      Cartesian components the factor array holds
 *                      (coordDim); sets the per-element factor stride.
 * @param   dimension   Standard directions to contract over.
 * @param   nelmt       Elements in the block, including padding.
 * @param   inoffset    Stride between the standard-direction planes of
 *                      @p in.
 * @param   dfptr       Derivative factors of the block, transposed
 *                      layout.
 * @param   in          Standard-direction derivatives, plane @em d at
 *                      `d * inoffset`.
 * @param   out         Single Cartesian derivative plane.
 * @param   threadBlock Thread-block handle supplied by the launcher macro.
 */
template <bool APPEND, bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void MultiplyByDerivDirFactorKernel(
    const unsigned dir, const unsigned nqTot, const unsigned ncoord,
    const unsigned dimension, const size_t nelmt, const size_t inoffset,
    const TData *dfptr, const TData *in, TData *out,
    const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx           = getGlobalIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    const TData *inptr   = in + nsize * p;
    TData *outptr        = out + nsize * p;
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData tmp;
        if constexpr (DEFORMED)
        {
            tmp =
                dfptr[(ndf - 1) * nqTot * e + nqTot * (dir * dimension) + idx] *
                inptr[idx];
            for (unsigned int d = 1; d < dimension; d++)
            {
                tmp += dfptr[(ndf - 1) * nqTot * e +
                             nqTot * (dir * dimension + d) + idx] *
                       inptr[idx + d * inoffset];
            }
        }
        else
        {
            tmp = dfptr[(ndf * e + dir * dimension)] * inptr[idx];
            for (unsigned int d = 1; d < dimension; d++)
            {
                tmp += dfptr[(ndf * e + dir * dimension + d)] *
                       inptr[idx + d * inoffset];
            }
        }

        if constexpr (APPEND)
        {
            outptr[idx] += tmp;
        }
        else
        {
            outptr[idx] = tmp;
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::MultiRegions::detail
