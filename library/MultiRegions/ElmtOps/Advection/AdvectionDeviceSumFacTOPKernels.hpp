///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDeviceSumFacTOPKernels.hpp
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
// Description: SumFacTOP device kernels of the advection operator, one
// element per thread block.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionDeviceSumFacTOPKernels.hpp
 * @brief SumFacTOP device kernels of the advection operator: one element
 * per thread block, threads indexed over the element's quadrature points.
 *
 * @details
 * These kernels compute the same thing as the SumFac ones in
 * AdvectionDeviceSumFacKernels.hpp -- the tensor-direction derivatives,
 * the collapsed-to-standard coordinate correction, the geometric
 * derivative factors and the dot product with the advection velocity --
 * but distribute the work differently. A whole thread block takes one
 * element at a time and its threads stride over the element's quadrature
 * points, each thread computing the output at the points it owns. In two
 * and three dimensions the element's input values are staged in shared
 * memory first, so that the tensor contractions, which read across the
 * element, hit shared rather than global memory; the derivative factors
 * and the velocity stay in global memory and are read at the thread's own
 * point index, the collapsed-coordinate tables at the direction indices
 * that point resolves into. Accordingly the data is not interleaved (one
 * element's values are contiguous) and the derivative factors are taken
 * in the transposed layout, with each factor component contiguous over
 * the element's points.
 *
 * The file provides, for each dimensionality: the per-element kernel
 * (Advection*DSumFacTOPKernel) and the device entry point the block
 * operator launches (the AdvectionKernelLauncher overload for that
 * dimension), which unpacks the size parameter and walks the block's
 * elements one thread block at a time, one grid row per component or
 * homogeneous mode, staging each element's input in shared memory in two
 * and three dimensions before calling the per-element kernel.
 * AdvectionSharedMemorySize reports how much dynamic shared memory such a
 * launch must request. The launchers are constrained to the SumFacTOP
 * implementation tag and are otherwise identical in name and signature to
 * the SumFac ones, which is how AdvectionBlockOpImpl::OperatorNDImpl can
 * launch either strategy from one call site.
 *
 * Everything in the namespace is guarded by NEKTAR_ENABLE_DEVICE and
 * DEVICE_COMPILE_ONLY, so it exists only in translation units compiled for
 * a device back-end.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a one-dimensional shape: none, since the 1D
 * launcher reads the element's input straight from global memory.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFacTOP.
 * @tparam TPhysSizeParameter1D  1D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Element sizes; unused.
 *
 * @return Zero.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int AdvectionSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter1D sizeParam1D)
{
    return 0;
}

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a two-dimensional shape.
 *
 * One thread block holds one element's input values, so the requirement
 * is the element's total number of quadrature points, whatever the shape.
 * The collapsed-coordinate factor tables are not staged.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFacTOP.
 * @tparam TPhysSizeParameter2D  2D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Quadrature-point counts per direction.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int AdvectionSharedMemorySize(
    const TPhysSizeParameter2D sizeParam2D)
{
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    return nq0 * nq1;
}

/**
 * @brief Dynamic shared memory a SumFacTOP launch needs per thread block,
 * in elements of TData, for a three-dimensional shape.
 *
 * As in two dimensions: one element's input values, i.e. the element's
 * total number of quadrature points.
 *
 * @tparam SHAPE_TYPE            Element shape; unused.
 * @tparam Implementation        Implementation tag; this overload is
 *                               constrained to SumFacTOP.
 * @tparam TPhysSizeParameter3D  3D physical-space size parameter (see
 *                               ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Quadrature-point counts per direction.
 *
 * @return Shared-memory requirement in elements of TData.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int AdvectionSharedMemorySize(
    const TPhysSizeParameter3D sizeParam3D)
{
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    return nq0 * nq1 * nq2;
}

/**
 * @brief One element of a 1D block, shared by a thread block:
 * out = scale * du/dxi * sum_d df(d) V_d at the quadrature points the
 * calling thread owns.
 *
 * The threads stride over the element's points; for each of them the
 * derivative along the single reference direction is formed by
 * contracting @p D0 over the element's values, then multiplied by the
 * derivative factor and velocity component of each coordinate direction.
 * Everything is addressed by the plain point index, since the element's
 * data is contiguous; in one dimension nothing is staged in shared memory
 * and @p in points straight at the element's values. The kernel ends with
 * a block barrier, as the 2D and 3D kernels do, where it protects the
 * staging of the launcher's next element.
 *
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   ncoord       Coordinate dimension, i.e. number of derivative
 *                       factors and velocity components.
 * @param   nq0          Quadrature points per element.
 * @param   D0           1D derivative matrix of the direction.
 * @param   df           Derivative factors of this element.
 * @param   advVel       Velocity data of this element; component d starts
 *                       at d * advVelOffset.
 * @param   advVelOffset Stride between velocity components.
 * @param   in           Input values of this element.
 * @param   out          Output values of this element.
 * @param   scale        Scalar factor applied to the result.
 * @param   threadBlock  Thread-block handle supplying the thread index,
 *                       the thread count and the barrier.
 */
template <bool APPEND, bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Advection1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[q];
        }

        // Multiply by derivative factors.
        TData tmp = 0.0;
        for (unsigned int d = 0u; d < ncoord; d++)
        {
            tmp += d0 * df[d * dfsize + dfindex] * advVel[d * advVelOffset + i];
        }

        if constexpr (APPEND)
        {
            out[i] += scale * tmp;
        }
        else
        {
            out[i] = scale * tmp;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element of a 2D block, shared by a thread block:
 * out = scale * sum_k V_k sum_d df(k,d) du/dxi_d at the quadrature points
 * the calling thread owns.
 *
 * The threads stride over the element's nq0 * nq1 points; the flat index
 * splits into the direction-0 and direction-1 indices, and each thread
 * contracts D0 and D1 over the element's staged input to form the two
 * tensor-direction derivatives at its own point. For triangles these are
 * collapsed-coordinate derivatives, so d0 is first multiplied by
 * f1[j] = 2/(1 - eta_1) and d1 augmented by d0 * f0[i] with
 * f0[i] = (1 + eta_0)/2. The 2 * ncoord derivative factors are then
 * applied and contracted with the velocity, a third factor/velocity pair
 * contributing when @p ncoord is 3. Derivative factors are indexed with
 * the component stride dfsize -- the transposed layout -- and the barrier
 * at the end protects the shared input against the next element's
 * staging.
 *
 * @tparam SHAPE_TYPE   Element shape; selects the collapsed-coordinate
 *                      correction.
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   ncoord       Coordinate dimension, i.e. number of
 *                       gradient/velocity components (2 or 3).
 * @param   nq0          Quadrature points in direction 0.
 * @param   nq1          Quadrature points in direction 1.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   f0           Collapsed-coordinate factor (1 + eta_0)/2 at the
 *                       direction-0 points (triangles only).
 * @param   f1           Collapsed-coordinate factor 2/(1 - eta_1) at the
 *                       direction-1 points (triangles only).
 * @param   df           Derivative factors of this element.
 * @param   advVel       Velocity data of this element; component k starts
 *                       at k * advVelOffset.
 * @param   advVelOffset Stride between velocity components.
 * @param   in           Input values of this element, staged in shared
 *                       memory by the caller.
 * @param   out          Output values of this element.
 * @param   scale        Scalar factor applied to the result.
 * @param   threadBlock  Thread-block handle supplying the thread index,
 *                       the thread count and the barrier.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Advection2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT advVel,
    const size_t advVelOffset, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = idx / nq0;
        const unsigned int dfindex = DEFORMED ? idx : 0;

        // Get advection velocity.
        TData vx, vy;
        vx = advVel[idx];
        vy = advVel[idx + 1u * advVelOffset];

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[q * nq1 + j] * in[nq0 * q + i];
        }

        // Moving from collapsed to standard coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            d0 *= f1[j];
            d1 += d0 * f0[i];
        }

        // Multiply by derivative factors.
        TData tmp;
        tmp = vx *
              (d0 * df[0u * dfsize + dfindex] + d1 * df[1u * dfsize + dfindex]);
        tmp += vy * (d0 * df[2u * dfsize + dfindex] +
                     d1 * df[3u * dfsize + dfindex]);
        if (ncoord == 3u)
        {
            TData vz = advVel[idx + 2u * advVelOffset];
            tmp += vz * (d0 * df[4u * dfsize + dfindex] +
                         d1 * df[5u * dfsize + dfindex]);
        }

        if constexpr (APPEND)
        {
            out[idx] += scale * tmp;
        }
        else
        {
            out[idx] = scale * tmp;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief One element of a 3D block, shared by a thread block:
 * out = scale * sum_k V_k sum_d df(k,d) du/dxi_d at the quadrature points
 * the calling thread owns.
 *
 * The threads stride over the element's nq0 * nq1 * nq2 points, each
 * splitting its flat index into the three direction indices and
 * contracting D0, D1 and D2 over the staged input. Shapes with collapsed
 * coordinates are then chain-ruled back to standard coordinates with the
 * tabulated factors -- hexahedra need no correction, prisms and pyramids
 * fold the corrected direction-0 (and, for pyramids, direction-1)
 * derivative into direction 2, and tetrahedra combine all four tables --
 * before the nine derivative factors are applied and contracted with the
 * velocity. The barrier at the end protects the shared
 * input against the next element's staging.
 *
 * @tparam SHAPE_TYPE   Element shape; selects the collapsed-coordinate
 *                      correction.
 * @tparam APPEND       Accumulate onto @p out instead of overwriting.
 * @tparam DEFORMED     Read per-point rather than per-element derivative
 *                      factors.
 * @tparam TthreadBlock Back-end thread-block handle type.
 *
 * @param   nq0          Quadrature points in direction 0.
 * @param   nq1          Quadrature points in direction 1.
 * @param   nq2          Quadrature points in direction 2.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   D2           1D derivative matrix of direction 2.
 * @param   f0           (1 + eta_0)/2 at the direction-0 points.
 * @param   f1           (1 + eta_1)/2 at the direction-1 points.
 * @param   f1m          2/(1 - eta_1) at the direction-1 points.
 * @param   f2           2/(1 - eta_2) at the direction-2 points.
 * @param   df           Derivative factors of this element, nine per
 *                       element or per point.
 * @param   advVel       Velocity data of this element; component k starts
 *                       at k * advVelOffset.
 * @param   advVelOffset Stride between velocity components.
 * @param   in           Input values of this element, staged in shared
 *                       memory by the caller.
 * @param   out          Output values of this element.
 * @param   scale        Scalar factor applied to the result.
 * @param   threadBlock  Thread-block handle supplying the thread index,
 *                       the thread count and the barrier.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Advection3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
    const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1 * nq2;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = (idx / nq0) % nq1;
        const unsigned int k       = idx / (nq0 * nq1);
        const unsigned int dfindex = DEFORMED ? idx : 0;

        // Get advection velocity.
        TData vx, vy, vz;
        vx = advVel[idx];
        vy = advVel[idx + 1u * advVelOffset];
        vz = advVel[idx + 2u * advVelOffset];

        // Compute tensorial derivative.
        // Direction 0
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[nq0 * nq1 * k + nq0 * j + q];
        }

        // Direction 1
        TData d1 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq1; ++q)
        {
            d1 += D1[q * nq1 + j] * in[nq0 * nq1 * k + nq0 * q + i];
        }

        // Direction 2
        TData d2 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq2; ++q)
        {
            d2 += D2[q * nq2 + k] * in[nq0 * nq1 * q + nq0 * j + i];
        }

        // Moving from collapsed to standard coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                      SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData tmp0 = f1m[j] * f2[k] * d0;
            TData tmp1 = f0[i] * tmp0;
            TData tmp2 = f2[k] * d1;
            d0         = tmp0;
            d1         = tmp1 + tmp2;
            d2 += tmp1 + f1[j] * tmp2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            d0 *= f2[k];
            d2 += f0[i] * d0;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            d0 *= f2[k];
            d1 *= f2[k];
            d2 += f0[i] * d0 + f1[j] * d1;
        }

        // Multiply by derivative factors.
        TData tmp;
        tmp = vx *
              (d0 * df[0u * dfsize + dfindex] + d1 * df[1u * dfsize + dfindex] +
               d2 * df[2u * dfsize + dfindex]);
        tmp += vy * (d0 * df[3u * dfsize + dfindex] +
                     d1 * df[4u * dfsize + dfindex] +
                     d2 * df[5u * dfsize + dfindex]);
        tmp += vz * (d0 * df[6u * dfsize + dfindex] +
                     d1 * df[7u * dfsize + dfindex] +
                     d2 * df[8u * dfsize + dfindex]);

        if constexpr (APPEND)
        {
            out[idx] += scale * tmp;
        }
        else
        {
            out[idx] = scale * tmp;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Device entry point of the 1D SumFacTOP advection kernel.
 *
 * Launched by AdvectionBlockOpImpl::OperatorNDImpl through
 * DEVICE_2DGRID_KERNEL_LAUNCHER, which appends the shared-memory pointer
 * (unused here) and the thread-block handle. Unpacks the size parameter
 * -- runtime or compile-time, both satisfying IsPhysSizeParameter1D_v as
 * the static_assert enforces. Each thread block then starts at its own
 * block index in the grid's first dimension and strides by that
 * dimension's range, so the block's @p nelmt elements (padding included)
 * are covered whatever grid size was launched -- the grid may hold fewer
 * thread blocks than there are elements (see GetDeviceGridSize in
 * ElmtBlockOp.hpp). The grid's second dimension gives the index c of the
 * component -- or homogeneous mode -- being advected, read once and fixed
 * for every element this thread block takes, since that dimension holds
 * exactly one block row for each. Element e's slice of the input and
 * output arrays is selected by pointer arithmetic that also offsets by c,
 * since @p in and @p out address every component, the velocity slice by
 * e and the plane c % @p nhomo, and the derivative-factor slice by e
 * alone -- the same velocity and factors serve every component -- before
 * Advection1DSumFacTOPKernel computes the element's output. The enable_if
 * pins this overload to the SumFacTOP tag; the SumFac tag resolves to the
 * identically named launcher in AdvectionDeviceSumFacKernels.hpp.
 *
 * @tparam SHAPE_TYPE           Element shape (segments only; unused).
 * @tparam Implementation       Implementation tag; SumFacTOP here.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter1D 1D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam1D  Element sizes: coordinate dimension and
 *                       quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   D0           1D derivative matrix of the direction.
 * @param   df           Derivative factors of the block.
 * @param   advVel       Velocity data of the block.
 * @param   advVelOffset Stride between velocity components.
 * @param   nhomo        Homogeneous planes per variable.
 * @param   in           Input values of the block.
 * @param   out          Output values of the block.
 * @param   scale        Scalar factor applied to the result.
 * @param   shmemptr     Dynamic shared memory of the thread block; unused
 *                       in one dimension.
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    const unsigned int nhomo, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale,
    [[maybe_unused]] unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter1D or TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr     = df + ndf * dfsize * e;
        const TData *inptr     = in + nq0 * nelmt * c + nq0 * e;
        TData *outptr          = out + nq0 * nelmt * c + nq0 * e;
        const TData *advVelPtr = advVel + nq0 * nelmt * (c % nhomo) + nq0 * e;
        Advection1DSumFacTOPKernel<APPEND, DEFORMED>(
            ncoord, nq0, D0, dfptr, advVelPtr, advVelOffset, inptr, outptr,
            scale, threadBlock);
        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D SumFacTOP advection kernel.
 *
 * As in 1D, but the shared memory is used: FETCH_SHARED_MEMORY() binds
 * @p shmemptr to the block's dynamic shared memory (a no-op on the
 * back-ends that pass it in). Each element's input values are then copied
 * from global into that shared memory -- the threads share the copy work
 * with the same stride they later compute with -- and a barrier publishes
 * the copy before Advection2DSumFacTOPKernel reads across it. That kernel
 * ends with a barrier of its own, so the copy for the next element cannot
 * start while a thread is still reading this one.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFacTOP here.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter2D 2D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam2D  Element sizes: coordinate dimension and
 *                       per-direction quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   f0           Collapsed-coordinate factor of direction 0.
 * @param   f1           Collapsed-coordinate factor of direction 1.
 * @param   df           Derivative factors of the block.
 * @param   advVel       Velocity data of the block.
 * @param   advVelOffset Stride between velocity components.
 * @param   nhomo        Homogeneous planes per variable.
 * @param   in           Input values of the block.
 * @param   out          Output values of the block.
 * @param   scale        Scalar factor applied to the result.
 * @param   shmemptr     Dynamic shared memory of the thread block, at
 *                       least one element's worth of values (see
 *                       AdvectionSharedMemorySize).
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT advVel,
    const size_t advVelOffset, const unsigned int nhomo,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData scale,
    unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter2D or TemplatedPhysSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int ncoord = sizeParam2D.ncoord();
    const unsigned int nq0    = sizeParam2D.nq0();
    const unsigned int nq1    = sizeParam2D.nq1();

    const unsigned int ndf    = 2 * ncoord;
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    TData *s_wsp0             = (TData *)shmemptr;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr = df + ndf * dfsize * e;
        const TData *inptr = in + nqTot * nelmt * c + nqTot * e;
        TData *outptr      = out + nqTot * nelmt * c + nqTot * e;
        const TData *advVelPtr =
            advVel + nqTot * nelmt * (c % nhomo) + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        Advection2DSumFacTOPKernel<SHAPE_TYPE, APPEND, DEFORMED>(
            ncoord, nq0, nq1, D0, D1, f0, f1, dfptr, advVelPtr, advVelOffset,
            s_wsp0, outptr, scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D SumFacTOP advection kernel.
 *
 * Counterpart of the 2D overload for three-dimensional shapes; the size
 * parameter carries no coordinate dimension, since a 3D element always
 * has three. One element is taken per thread block at a time, its input
 * staged in shared memory behind a barrier, then Advection3DSumFacTOPKernel
 * computes its output.
 *
 * @tparam SHAPE_TYPE           Element shape.
 * @tparam Implementation       Implementation tag; SumFacTOP here.
 * @tparam APPEND               Accumulate onto @p out.
 * @tparam DEFORMED             Per-point derivative factors.
 * @tparam TPhysSizeParameter3D 3D physical-space size parameter.
 * @tparam TthreadBlock         Back-end thread-block handle type.
 *
 * @param   sizeParam3D  Element sizes: per-direction quadrature points.
 * @param   nelmt        Elements in the block, including padding.
 * @param   D0           1D derivative matrix of direction 0.
 * @param   D1           1D derivative matrix of direction 1.
 * @param   D2           1D derivative matrix of direction 2.
 * @param   f0           (1 + eta_0)/2 at the direction-0 points.
 * @param   f1           (1 + eta_1)/2 at the direction-1 points.
 * @param   f1m          2/(1 - eta_1) at the direction-1 points.
 * @param   f2           2/(1 - eta_2) at the direction-2 points.
 * @param   df           Derivative factors of the block.
 * @param   advVel       Velocity data of the block.
 * @param   advVelOffset Stride between velocity components.
 * @param   nhomo        Homogeneous planes per variable; unused in 3D.
 * @param   in           Input values of the block.
 * @param   out          Output values of the block.
 * @param   scale        Scalar factor applied to the result.
 * @param   shmemptr     Dynamic shared memory of the thread block, at
 *                       least one element's worth of values (see
 *                       AdvectionSharedMemorySize).
 * @param   threadBlock  Thread-block handle supplied by the launcher
 *                       macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TPhysSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
    const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    [[maybe_unused]] const unsigned int nhomo, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter3D or TemplatedPhysSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;

    TData *s_wsp0             = (TData *)shmemptr;
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr     = df + ndf * dfsize * e;
        const TData *inptr     = in + nqTot * nelmt * c + nqTot * e;
        TData *outptr          = out + nqTot * nelmt * c + nqTot * e;
        const TData *advVelPtr = advVel + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        Advection3DSumFacTOPKernel<SHAPE_TYPE, APPEND, DEFORMED>(
            nq0, nq1, nq2, D0, D1, D2, f0, f1, f1m, f2, dfptr, advVelPtr,
            advVelOffset, s_wsp0, outptr, scale, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
