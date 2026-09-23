///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseDeviceSumFacTOPKernels.hpp
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
// Description: Device SumFacTOP kernels of the inner product with the
// basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTBaseDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernels of the inner product with the basis: one
 * element per thread block, with the basis tables and all intermediates
 * staged in shared memory.
 *
 * @details
 * These are the kernels the Device/SumFacTOP block implementation launches
 * (see IProductWRTBaseDeviceSumFac.hpp); the SumFac counterparts, where
 * one thread owns a whole element, are in
 * IProductWRTBaseDeviceSumFacKernels.hpp. Everything below is compiled only
 * into device translation units of a device-enabled build
 * (NEKTAR_ENABLE_DEVICE and DEVICE_COMPILE_ONLY).
 *
 * One thread block integrates one element at a time and strides over the
 * block's elements (getBlockIdx .. getBlockRange), the grid's y dimension
 * selecting the component (times homogeneous mode); the field storage is
 * reshaped to interleave width 1 before the launch, so an element's values
 * are contiguous. Within a stage the block's threads stride over that
 * stage's output entries (getLocalIdx .. getLocalRange), and consecutive
 * stages are separated by localBarrier() calls, since one thread's stage-n
 * output is read by others in stage n+1.
 *
 * The IProductWRTBaseKernelLauncher overloads -- one per dimension,
 * selected by the type of the size parameter, and each in a form with and
 * a form without the quadrature metric -- own the shared-memory layout: first
 * the per-element scratch (the element's physical values and one region
 * per intermediate stage), then the one-dimensional basis tables, copied
 * in once before the element loop, and finally -- for the nodal shapes --
 * the modal coefficients that precede the nodal mapping.
 * IProductWRTBaseSharedMemorySize computes exactly that total; the
 * global-memory workspace of the SumFac strategy is not needed here, so
 * IProductWRTBaseWorkSpaceSize returns zero for every dimension.
 *
 * Unlike the SumFac kernels, the shape kernels below take no weights and
 * no Jacobians: the launchers apply the whole metric once, as they copy
 * the element's physical values into shared memory, multiplying each point
 * by the tensor product of the per-direction weights and by the element
 * Jacobian (one value per element, or one per point when DEFORMED). The
 * overloads without a metric copy the values unchanged. Consequently a
 * single set of shape kernels serves both cases.
 *
 * A basis table basisN holds one row of nqN values per mode, the row index
 * being the shape's combined mode index along a collapsed direction.
 * Because a thread is responsible for an arbitrary flat output index
 * rather than for a loop nest, the collapsed shapes take precomputed
 * mode-index tables (pindex, qindex, rindex; see
 * ModeIndexDataWarehouse.hpp) that recover the mode indices a stage needs
 * from that flat index. Which of p, q and r a given table holds, and
 * whether it is keyed by a flat mode or by a flat (p,q) pair, differs per
 * shape and is stated in the parameter documentation of each kernel below.
 * @p isModified marks a first-direction eModified_A basis, whose extra
 * collapsed vertex and edge contributions are summed per thread and then
 * reduced across the block with blockReduceSum, which accumulates onto the
 * mode it corrects. For the nodal shapes the modal kernel of the matching
 * non-nodal shape writes to a shared-memory slice and MatVecSumFacTOPKernel
 * then maps the result to the nodal coefficients with the transpose of the
 * nodal-to-modal matrix.
 *
 * @see IProductWRTBaseDeviceSumFacKernels.hpp for the
 * one-element-per-thread strategy.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsDeviceKernels.hpp>

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Global workspace a 1D SumFacTOP launch needs, in TData values:
 * none, this strategy keeping the element's values and every intermediate
 * stage in shared memory.
 *
 * No overload of this helper requests global workspace for the SumFacTOP
 * strategy; the overloads mirror the SumFac ones in
 * IProductWRTBaseDeviceSumFacKernels.hpp, which do, so that the block
 * implementation can query the size the same way for either strategy.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam Implementation    Operators::SumFacTOP.
 * @tparam TSizeParameter1D  1D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block; unused.
 * @param   sizeParam1D  Sizes of the expansion; unused.
 *
 * @return Zero.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Global workspace a 2D SumFacTOP launch needs: none, for the
/// reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    return 0;
}

/// @brief Global workspace a 3D SumFacTOP launch needs: none, for the
/// reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter3D sizeParam3D)
{
    return 0;
}

/**
 * @brief Shared memory a 1D SumFacTOP inner product needs, in TData
 * values: nq0, the element's physical values.
 *
 * The segment kernel reads its basis table straight from global memory and
 * contracts into the output, so nothing else is staged.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam Implementation    Operators::SumFacTOP.
 * @tparam TSizeParameter1D  1D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 *
 * @return Shared-memory size in TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTBaseSharedMemorySize(
    const TSizeParameter1D sizeParam1D)
{
    const unsigned int nq0 = sizeParam1D.nq0();

    return nq0;
}

/**
 * @brief Shared memory a 2D SumFacTOP inner product needs, in TData
 * values.
 *
 * The sum of the regions the 2D IProductWRTBaseKernelLauncher lays out:
 * the element's nq0 * nq1 physical values, the nm0 * nq1 direction-0 sums,
 * and copies of both basis tables -- nm0 * nq0 for direction 0 and, for
 * direction 1, one row per mode of the shape's ordering, so nm1 * nq1 for
 * a quadrilateral but nmTot * nq1 for a triangle. NodalTri adds nmTot
 * values for the modal coefficients that precede the nodal mapping.
 *
 * @tparam SHAPE_TYPE        Quad, Tri or NodalTri.
 * @tparam Implementation    Operators::SumFacTOP.
 * @tparam TSizeParameter2D  2D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam2D  Modal and quadrature sizes per direction.
 *
 * @return Shared-memory size in TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTBaseSharedMemorySize(
    const TSizeParameter2D sizeParam2D)
{
    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return nm0 * nq0 + nm1 * nq1 + nq0 * nq1 + nm0 * nq1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        return nm0 * nq0 + nmTot * nq1 + nq0 * nq1 + nm0 * nq1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return nm0 * nq0 + nmTot * nq1 + nq0 * nq1 + nm0 * nq1 + nmTot;
    }
}

/**
 * @brief Shared memory a 3D SumFacTOP inner product needs, in TData
 * values.
 *
 * As in 2D: the element's nq0 * nq1 * nq2 physical values, the
 * nm0 * nq1 * nq2 direction-0 sums and the direction-1 sums of the second
 * stage, plus copies of the three basis tables, each sized by the number
 * of rows the shape's mode ordering gives that direction (nm01, nmode2 and
 * the like below), and, for the nodal shapes, nmTot values for the modal
 * coefficients that precede the nodal mapping.
 *
 * @tparam SHAPE_TYPE        Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam Implementation    Operators::SumFacTOP.
 * @tparam TSizeParameter3D  3D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam3D  Modal and quadrature sizes per direction.
 *
 * @return Shared-memory size in TData values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTBaseSharedMemorySize(
    const TSizeParameter3D sizeParam3D)
{
    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm01 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm01 * nq2 + nmTot;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + nmTot;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
    }
}

/**
 * @brief Segment SumFacTOP kernel: the block's threads stride over the nm0
 * output modes, each contracting the element's staged values against one
 * basis row.
 *
 * Returns after a localBarrier, so the caller may reuse the shared scratch
 * for the next element.
 *
 * @tparam SCALE   Multiply the result by @p scale.
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   nm0         Modes of the expansion.
 * @param   nq0         Quadrature points of the expansion.
 * @param   basis0      Basis table (nm0 rows of nq0).
 * @param   in          Element's physical values in shared memory, already
 *                      carrying the quadrature metric if it is in use.
 * @param   out         Element's coefficients.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseSegSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int p = idx0; p < nm0; p += stride)
    {
        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            sum += in[i] * basis0[p * nq0 + i];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[p] += sum;
        }
        else
        {
            out[p] = sum;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Quadrilateral SumFacTOP kernel: two thread-strided stages
 * separated by a barrier.
 *
 * In the first stage the threads stride over the nm0 * nq1 entries of
 * @p wsp, entry p * nq1 + j being the direction-0 contraction of line j
 * against basis row p; in the second they stride over the nmTot output
 * coefficients, each contracting its column of @p wsp against a
 * direction-1 basis row. Both output indices are decomposed from the flat
 * thread index by division, no mode-index table being needed for a
 * tensor-product shape.
 *
 * @tparam SCALE   Multiply the result by @p scale.
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   nm0,nm1     Modes per direction.
 * @param   nmTot       Coefficients per element.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   nqTot       Quadrature points per element; unused here.
 * @param   basis0      Direction-0 basis table in shared memory.
 * @param   basis1      Direction-1 basis table in shared memory.
 * @param   in          Element's physical values in shared memory, already
 *                      carrying the quadrature metric if it is in use.
 * @param   out         Element's coefficients.
 * @param   wsp         Shared scratch for the direction-0 sums, nm0 * nq1
 *                      values.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseQuadSumFacTOPKernel(
    const unsigned int nm0, [[maybe_unused]] const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    [[maybe_unused]] const unsigned int nqTot, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int p = idx / nq1;
        unsigned int cnt_ji  = nq0 * j;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            sum += in[cnt_ji] * basis0[p * nq0 + i];
        }
        wsp[idx] = sum;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int q = idx / nm0;
        unsigned int cnt_pj  = nq1 * p;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
        {
            sum += wsp[cnt_pj] * basis1[q * nq1 + j];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum;
        }
        else
        {
            out[idx] = sum;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Triangular SumFacTOP kernel: the quadrilateral structure with the
 * collapsed mode ordering.
 *
 * The second stage's thread index is the flat triangular mode index, from
 * which @p pindex recovers the p the thread's column of @p wsp belongs to;
 * @p basis1 is indexed by that same flat mode.
 *
 * With @p isModified set, each thread sums the collapsed-vertex correction
 * over its share of the quadrature points and blockReduceSum accumulates
 * the block's total onto mode 1.
 *
 * @tparam SCALE   Multiply the result by @p scale.
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   nm0,nm1     Modes per direction.
 * @param   nmTot       Coefficients per element (the triangular count, not
 *                      the product of the two).
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   nqTot       Quadrature points per element.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   pindex      Table giving p for each flat (p,q) mode.
 * @param   basis0      Direction-0 basis table in shared memory.
 * @param   basis1      Direction-1 basis table in shared memory, one row
 *                      per combined mode.
 * @param   in          Element's physical values, already carrying the
 *                      quadrature metric if it is in use; staged in shared
 *                      memory by the metric launcher, passed straight from
 *                      global memory by the metric-free one.
 * @param   out         Element's coefficients.
 * @param   wsp         Shared scratch for the direction-0 sums, nm0 * nq1
 *                      values.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTriSumFacTOPKernel(
    const unsigned int nm0, [[maybe_unused]] const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nqTot, const bool isModified,
    const unsigned int *NEK_RESTRICT pindex, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int p = idx / nq1;
        unsigned int cnt_ji  = nq0 * j;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            sum += in[cnt_ji] * basis0[p * nq0 + i];
        }
        wsp[idx] = sum;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = pindex[idx];
        unsigned int cnt_pj  = nq1 * p;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
        {
            sum += wsp[cnt_pj] * basis1[idx * nq1 + j];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum;
        }
        else
        {
            out[idx] = sum;
        }
    }

    // Correction for singular vertex in collapsed coordinates.
    // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
    // With contributions from every quadrature point
    if (isModified)
    {
        localBarrier(threadBlock);

        TData prod = 0.0;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            prod += basis0[nq0 + i] * basis1[nq1 + j] * in[idx];
        }

        if constexpr (SCALE)
        {
            prod *= scale;
        }

        blockReduceSum(prod, threadBlock, out + 1);
    }

    localBarrier(threadBlock);
}

/**
 * @brief Hexahedral SumFacTOP kernel: three thread-strided stages
 * separated by barriers.
 *
 * The threads stride first over the nm0 * nq1 * nq2 direction-0 sums in
 * @p wsp0, then over the nm0 * nm1 * nq2 direction-1 sums in @p wsp1, then
 * over the nmTot output coefficients, each stage decomposing its flat
 * index into the mode and point indices it needs.
 *
 * @tparam SCALE   Multiply the result by @p scale.
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nmTot       Coefficients per element.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   nqTot       Quadrature points per element; unused here.
 * @param   basis0,basis1,basis2    Basis tables in shared memory.
 * @param   in          Element's physical values in shared memory, already
 *                      carrying the quadrature metric if it is in use.
 * @param   out         Element's coefficients.
 * @param   wsp0,wsp1   Shared scratch of the two intermediate stages.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseHexSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1,
    [[maybe_unused]] const unsigned int nm2, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const unsigned int nqTot, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[i + nq0 * p];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int k = idx % nq2;
        const unsigned int q = (idx / nq2) % nm1;
        const unsigned int p = idx / (nq2 * nm1);
        unsigned int cnt_pkj = nq2 * nq1 * p + nq1 * k;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += wsp0[cnt_pkj] * basis1[q * nq1 + j];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int q = (idx / nm0) % nm1;
        const unsigned int r = idx / (nm0 * nm1);
        unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q;

        TData sum = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
        {
            sum += wsp1[cnt_pqk] * basis2[r * nq2 + k];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum;
        }
        else
        {
            out[idx] = sum;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Tetrahedral SumFacTOP kernel: the hexahedral structure with the
 * tetrahedral mode ordering, resolved through the mode-index tables.
 *
 * The second stage strides over the nm01 * nq2 entries of @p wsp1,
 * nm01 = (2 * nm1 - nm0 + 1) * nm0 / 2 being the number of (p,q) mode
 * pairs, with @p pindex1 giving each pair's p. The third strides over the
 * nmTot coefficients, @p pindex2 and @p qindex2 giving the mode's p and q,
 * from which the flat (p,q) pair and the direction-2 basis row are
 * recomputed -- the latter skipping the rows an anisotropic order leaves
 * out.
 *
 * With @p isModified set, the top-vertex, bottom-vertex and singular-edge
 * corrections are summed per thread and reduced onto modes 1, nm2 and
 * nm2 + r respectively. Two variants of that reduction exist: for
 * nm2 <= NM2_MAX all corrections are accumulated in one register array
 * during a single sweep of the quadrature points; beyond that the sweep is
 * repeated per singular-edge mode to bound the register use.
 *
 * @tparam SCALE   Multiply the result by @p scale.
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nmTot       Coefficients per element.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   nqTot       Quadrature points per element.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   pindex1     Table giving p for each flat (p,q) pair.
 * @param   pindex2     Table giving p for each flat (p,q,r) mode.
 * @param   qindex2     Table giving q for each flat (p,q,r) mode.
 * @param   basis0,basis1,basis2    Basis tables in shared memory, the
 *                      latter two indexed by combined modes.
 * @param   in          Element's physical values, already carrying the
 *                      quadrature metric if it is in use; staged in shared
 *                      memory by the metric launcher, passed straight from
 *                      global memory by the metric-free one.
 * @param   out         Element's coefficients.
 * @param   wsp0,wsp1   Shared scratch of the two intermediate stages.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTetSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nqTot, const bool isModified,
    const unsigned int *NEK_RESTRICT pindex1,
    const unsigned int *NEK_RESTRICT pindex2,
    const unsigned int *NEK_RESTRICT qindex2, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[i + nq0 * p];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm01 * nq2; idx += stride)
    {
        const unsigned int mode_pq = idx / nq2;
        const unsigned int p       = pindex1[mode_pq];
        const unsigned int k       = idx % nq2;
        unsigned int cnt_pkj       = nq1 * nq2 * p + nq1 * k;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += basis1[mode_pq * nq1 + j] * wsp0[cnt_pkj];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p       = pindex2[idx];
        const unsigned int q       = qindex2[idx];
        const unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u + q;
        const unsigned int mode2 =
            idx + ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k)
        {
            tmp += wsp1[mode_pq * nq2 + k] * basis2[mode2 * nq2 + k];
        }

        if constexpr (SCALE)
        {
            tmp *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += tmp;
        }
        else
        {
            out[idx] = tmp;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        localBarrier(threadBlock);

        constexpr unsigned int NM2_MAX = 8;
        if (nm2 <= NM2_MAX)
        {
            TData prod[NM2_MAX] = {0.0};
            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                // top vertex
                prod[nm2 - 1u] +=
                    (basis0[i] * basis1[nq1 + j] + basis0[nq0 + i] * basis1[j] +
                     basis0[nq0 + i] * basis1[nq1 + j]) *
                    basis2[nq2 + k] * in[idx];

                // singular edge
                TData tmp = basis1[nq1 + j] * basis0[nq0 + i] * in[idx];
#pragma unroll
                for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                {
                    prod[r] += basis2[(r + 1u) * nq2 + k] * tmp;
                }

                // bottom vertex
                prod[0] += basis2[k] * tmp;
            }
#if !defined(NEKTAR_ENABLE_SYCL)
#pragma unroll
#endif
            for (unsigned int r = 0u; r < nm2; ++r)
            {
                if constexpr (SCALE)
                {
                    prod[r] *= scale;
                }

                if (r == nm2 - 1u)
                {
                    blockReduceSum(prod[r], threadBlock, out + 1);
                }
                else
                {
                    blockReduceSum(prod[r], threadBlock, out + nm2 + r);
                }
            }
        }
        else
        {
            TData prod0 = 0.0;
            TData prod1 = 0.0;

            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                // top vertex
                prod0 +=
                    (basis0[i] * basis1[nq1 + j] + basis0[nq0 + i] * basis1[j] +
                     basis0[nq0 + i] * basis1[nq1 + j]) *
                    basis2[nq2 + k] * in[idx];

                // bottom vertex
                prod1 +=
                    basis0[nq0 + i] * basis1[nq1 + j] * basis2[k] * in[idx];
            }

            if constexpr (SCALE)
            {
                prod0 *= scale;
                prod1 *= scale;
            }

            blockReduceSum(prod0, threadBlock, out + 1);
            blockReduceSum(prod1, threadBlock, out + nm2);

            // singular edge
            for (unsigned int r = 1u; r < nm2 - 1u; ++r)
            {
                TData prod = 0.0;

                for (unsigned int idx = idx0; idx < nqTot; idx += stride)
                {
                    const unsigned int i = idx % nq0;
                    const unsigned int j = (idx / nq0) % nq1;
                    const unsigned int k = idx / (nq0 * nq1);

                    prod += basis2[(r + 1u) * nq2 + k] * basis1[nq1 + j] *
                            basis0[nq0 + i] * in[idx];
                }

                if constexpr (SCALE)
                {
                    prod *= scale;
                }

                blockReduceSum(prod, threadBlock, out + nm2 + r);
            }
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Prismatic SumFacTOP kernel: three thread-strided stages, with
 * direction 1 a full tensor-product direction and direction 2 collapsed
 * against direction 0.
 *
 * The second stage is the hexahedral one (nm0 * nm1 * nq2 entries); in the
 * third, @p pindex, @p qindex and @p rindex give the thread's mode
 * indices, from which the combined (p,r) row of @p basis2 is recomputed.
 *
 * With @p isModified set, the collapsed-edge correction is summed per
 * thread and reduced onto the modes nm2 * q + 1; as in the tetrahedron, a
 * register array serves while nm1 <= NM1_MAX and the quadrature sweep is
 * repeated per q beyond that.
 *
 * @tparam SCALE   Multiply the result by @p scale.
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nmTot       Coefficients per element.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   nqTot       Quadrature points per element.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   pindex,qindex,rindex    Tables giving p, q and r for each flat
 *                      mode.
 * @param   basis0,basis1,basis2    Basis tables in shared memory, the last
 *                      indexed by the combined (p,r) mode.
 * @param   in          Element's physical values, already carrying the
 *                      quadrature metric if it is in use; staged in shared
 *                      memory by the metric launcher, passed straight from
 *                      global memory by the metric-free one.
 * @param   out         Element's coefficients.
 * @param   wsp0,wsp1   Shared scratch of the two intermediate stages.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePrismSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nqTot, const bool isModified,
    const unsigned int *NEK_RESTRICT pindex,
    const unsigned int *NEK_RESTRICT qindex,
    const unsigned int *NEK_RESTRICT rindex, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = nq1 * nq0 * k + nq0 * j;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[nq0 * p + i];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int k = idx % nq2;
        const unsigned int q = (idx / nq2) % nm1;
        const unsigned int p = idx / (nq2 * nm1);
        unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += basis1[q * nq1 + j] * wsp0[cnt_pkj];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p       = pindex[idx];
        const unsigned int q       = qindex[idx];
        const unsigned int r       = rindex[idx];
        const unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u + r;
        unsigned int cnt_pqk       = nm1 * nq2 * p + nq2 * q;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
        {
            sum_k += basis2[mode_pr * nq2 + k] * wsp1[cnt_pqk];
        }

        if constexpr (SCALE)
        {
            sum_k *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum_k;
        }
        else
        {
            out[idx] = sum_k;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        localBarrier(threadBlock);

        constexpr unsigned int NM1_MAX = 8;
        if (nm1 <= NM1_MAX)
        {
            TData prod[NM1_MAX] = {0.0};

            for (unsigned int idx = idx0; idx < nqTot; idx += stride)
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                TData tmp = in[idx] * basis2[nq2 + k] * basis0[nq0 + i];
#pragma unroll
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    prod[q] += tmp * basis1[q * nq1 + j];
                }
            }

#if !defined(NEKTAR_ENABLE_SYCL)
#pragma unroll
#endif
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                if constexpr (SCALE)
                {
                    prod[q] *= scale;
                }

                blockReduceSum(prod[q], threadBlock, out + nm2 * q + 1);
            }
        }
        else
        {
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                TData prod = 0.0;

                for (unsigned int idx = idx0; idx < nqTot; idx += stride)
                {
                    const unsigned int i = idx % nq0;
                    const unsigned int j = (idx / nq0) % nq1;
                    const unsigned int k = idx / (nq0 * nq1);

                    prod += in[idx] * basis2[nq2 + k] * basis1[q * nq1 + j] *
                            basis0[nq0 + i];
                }

                if constexpr (SCALE)
                {
                    prod *= scale;
                }

                blockReduceSum(prod, threadBlock, out + nm2 * q + 1);
            }
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Pyramidal SumFacTOP kernel: three thread-strided stages, with
 * direction 2 collapsed against both other directions.
 *
 * The third stage takes the thread's p and q from @p pindex and @p qindex
 * and derives its direction-2 basis row from the flat mode index, offset
 * past the rows an anisotropic order leaves out.
 *
 * With @p isModified set, the collapsed top-vertex correction is summed
 * per thread and reduced onto mode 1.
 *
 * @tparam SCALE   Multiply the result by @p scale.
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nmTot       Coefficients per element.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   nqTot       Quadrature points per element.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   pindex,qindex   Tables giving p and q for each flat mode.
 * @param   basis0,basis1,basis2    Basis tables in shared memory, the last
 *                      indexed by the combined mode.
 * @param   in          Element's physical values, already carrying the
 *                      quadrature metric if it is in use; staged in shared
 *                      memory by the metric launcher, passed straight from
 *                      global memory by the metric-free one.
 * @param   out         Element's coefficients.
 * @param   wsp0,wsp1   Shared scratch of the two intermediate stages.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool SCALE, bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePyrSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nqTot, const bool isModified,
    const unsigned int *NEK_RESTRICT pindex,
    const unsigned int *NEK_RESTRICT qindex, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int j = idx % nq1;
        const unsigned int k = (idx / nq1) % nq2;
        const unsigned int p = idx / (nq1 * nq2);
        unsigned int cnt_kji = k * nq1 * nq0 + j * nq0;

        TData sum_kj = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
        {
            sum_kj += in[cnt_kji] * basis0[nq0 * p + i];
        }
        wsp0[idx] = sum_kj;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int k = idx % nq2;
        const unsigned int q = (idx / nq2) % nm1;
        const unsigned int p = idx / (nq2 * nm1);
        unsigned int cnt_pkj = nq1 * nq2 * p + k * nq1;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
        {
            sum_k += basis1[q * nq1 + j] * wsp0[cnt_pkj];
        }
        wsp1[idx] = sum_k;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nmTot; idx += stride)
    {
        const unsigned int p = pindex[idx];
        const unsigned int q = qindex[idx];
        const unsigned int mode2 =
            idx + ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);
        unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q;

        TData sum_k = 0.0;
#pragma unroll
        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
        {
            sum_k += basis2[mode2 * nq2 + k] * wsp1[cnt_pqk];
        }
        if constexpr (SCALE)
        {
            sum_k *= scale;
        }

        if constexpr (APPEND)
        {
            out[idx] += sum_k;
        }
        else
        {
            out[idx] = sum_k;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        localBarrier(threadBlock);

        TData prod = 0.0;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);

            // top vertex
            prod += (basis0[i] * basis1[nq1 + j] + basis0[nq0 + i] * basis1[j] +
                     basis0[nq0 + i] * basis1[nq1 + j]) *
                    basis2[nq2 + k] * in[idx];
        }

        if constexpr (SCALE)
        {
            prod *= scale;
        }

        blockReduceSum(prod, threadBlock, out + 1);
    }

    localBarrier(threadBlock);
}

/**
 * @brief Device entry point of the 1D SumFacTOP inner product, quadrature
 * metric included.
 *
 * Unpacks the element sizes from @p sizeParam1D -- runtime values or
 * compile-time constants, see ElmtHelper.hpp -- fetches the shared-memory
 * base, then block-strides over the block's segments within the component
 * the grid's y index selects, staging each element's weighted values in
 * shared memory before calling the segment kernel. The staging loop is
 * where the quadrature metric enters: each thread writes in * jac * w0 for
 * its share of the points, the Jacobian being read per point when DEFORMED
 * and once per element otherwise. A localBarrier then makes the staged
 * values visible to the whole block. The enable_if on the Implementation
 * tag is what makes the SumFac and SumFacTOP launchers of the same name
 * distinguishable; the size-parameter type is checked by a static_assert
 * instead. The launch bounds come from GetMaxThreadPerBlock (for SumFacTOP
 * the templated size parameter's mode count rounded to warps and capped at
 * the device's default block size, which is also the bound when the sizes
 * are runtime values).
 *
 * @tparam SHAPE_TYPE       Shape of the block's elements (Seg).
 * @tparam Implementation   Operators::SumFacTOP.
 * @tparam SCALE,APPEND     Passed through to the segment kernel.
 * @tparam DEFORMED         @p jac holds one value per quadrature point
 *                          rather than a single per-element value; read by
 *                          the staging loop, not passed to the segment
 *                          kernel.
 * @tparam TSizeParameter1D Size-parameter type carrying nm0 and nq0.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam1D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   basis0      Basis table.
 * @param   w0          Quadrature weights.
 * @param   nodToMod    Nodal-to-modal matrix; unused in 1D.
 * @param   jac         Block's Jacobians.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Global-memory workspace; unused by SumFacTOP.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base, sized by
 *                      IProductWRTBaseSharedMemorySize.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter1D sizeParam1D, const size_t nelmt,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *s_wsp0 = (TData *)shmemptr;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nq0 * nelmt * c + nq0 * e;
        TData *outptr       = out + nm0 * nelmt * c + nm0 * e;

        for (unsigned int i = idx0; i < nq0; i += stride)
        {
            if constexpr (DEFORMED)
            {
                s_wsp0[i] = inptr[i] * jacptr[i] * w0[i];
            }
            else
            {
                s_wsp0[i] = inptr[i] * jacptr[0] * w0[i];
            }
        }

        localBarrier(threadBlock);

        IProductWRTBaseSegSumFacTOPKernel<SCALE, APPEND>(
            nm0, nq0, basis0, s_wsp0, outptr, scale, threadBlock);
        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 1D SumFacTOP inner product without the
 * quadrature metric; selected by the absence of the @p w0 and @p jac
 * arguments.
 *
 * The same block-strided loop as the quadrature overload, staging the
 * element's values unchanged.
 *
 * @param   sizeParam1D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   basis0      Basis table.
 * @param   nodToMod    Nodal-to-modal matrix; unused in 1D.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Global-memory workspace; unused by SumFacTOP.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter1D sizeParam1D, const size_t nelmt,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    TData *s_wsp0 = (TData *)shmemptr;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nq0 * nelmt * c + nq0 * e;
        TData *outptr      = out + nm0 * nelmt * c + nm0 * e;

        for (unsigned int i = idx0; i < nq0; i += stride)
        {
            s_wsp0[i] = inptr[i];
        }

        localBarrier(threadBlock);

        IProductWRTBaseSegSumFacTOPKernel<SCALE, APPEND>(
            nm0, nq0, basis0, s_wsp0, outptr, scale, threadBlock);
        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D SumFacTOP inner product, quadrature
 * metric included.
 *
 * Unpacks the element sizes from @p sizeParam2D, lays out shared memory
 * and copies the basis tables in, then loops over the block's elements,
 * staging each element's weighted values and calling the shape kernel
 * selected at compile time. The shared-memory regions follow the order
 * s_wsp0 (the element's nq0 * nq1 values), s_wsp1 (the nm0 * nq1
 * direction-0 sums), s_basis0, s_basis1 and, for NodalTri, s_out1ptr for
 * the modal coefficients MatVecSumFacTOPKernel maps onto @p out. The
 * direction-1 table has one row per mode of the shape's ordering, hence
 * nmTot rows for a triangle. Staging the element's values is where the
 * quadrature metric enters, in * jac * w0 * w1 per point.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam Implementation   Operators::SumFacTOP.
 * @tparam SCALE,APPEND     Passed through to the shape kernel, except that
 *                          for NodalTri the shape kernel is instantiated
 *                          with APPEND false and @p APPEND governs the
 *                          nodal mapping instead.
 * @tparam DEFORMED         @p jac holds one value per quadrature point
 *                          rather than a single per-element value; read by
 *                          the staging loop, not passed to the shape
 *                          kernel.
 * @tparam TSizeParameter2D Size-parameter type carrying the mode and point
 *                          counts.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam2D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0      Mode-index table of the collapsed shapes.
 * @param   basis0,basis1   Basis tables per direction.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   nodToMod    Nodal-to-modal matrix; nodal shapes only.
 * @param   jac         Block's Jacobians.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Global-memory workspace; unused by SumFacTOP, which
 *                      works entirely in shared memory.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter2D sizeParam2D, const size_t nelmt,
        const bool isModified, const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();

    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset = 0, nmode0 = 0, nmode1 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nm1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nmTot;
    }

    TData *s_wsp0    = (TData *)shmemptr;
    TData *s_wsp1    = s_wsp0 + nqTot;
    TData *s_basis0  = s_wsp1 + offset;
    TData *s_basis1  = s_basis0 + nmode0 * nq0;
    TData *s_out1ptr = s_basis1 + nmode1 * nq1;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nqTot * nelmt * c + nqTot * e;
        TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            if constexpr (DEFORMED)
            {
                s_wsp0[idx] = inptr[idx] * jacptr[idx] * w0[i] * w1[j];
            }
            else
            {
                s_wsp0[idx] = inptr[idx] * jacptr[0] * w0[i] * w1[j];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, s_wsp0,
                outptr, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, s_wsp0, outptr, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<SCALE, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, s_wsp0, s_out1ptr, s_wsp1, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D SumFacTOP inner product without the
 * quadrature metric; selected by the absence of the weight and Jacobian
 * arguments.
 *
 * The same shared-memory layout and element loop as the quadrature
 * overload, copying the element's values unchanged.
 *
 * @note The three shape branches do not read those staged values alike:
 * Quad passes the shared copy s_wsp0 to its shape kernel, whereas Tri and
 * NodalTri pass inptr, the element's values in global memory. The copy
 * into s_wsp0 still runs in every case.
 *
 * @param   sizeParam2D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0      Mode-index table of the collapsed shapes.
 * @param   basis0,basis1   Basis tables per direction.
 * @param   nodToMod    Nodal-to-modal matrix; nodal shapes only.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Global-memory workspace; unused by SumFacTOP.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, typename TSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter2D sizeParam2D, const size_t nelmt,
        const bool isModified, const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData scale, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();

    const unsigned int nqTot = nq0 * nq1;

    unsigned int offset = 0, nmode0 = 0, nmode1 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nm1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nmTot;
    }

    TData *s_wsp0    = (TData *)shmemptr;
    TData *s_wsp1    = s_wsp0 + nqTot;
    TData *s_basis0  = s_wsp1 + offset;
    TData *s_basis1  = s_basis0 + nmode0 * nq0;
    TData *s_out1ptr = s_basis1 + nmode1 * nq1;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nqTot * nelmt * c + nqTot * e;
        TData *outptr      = out + nmTot * nelmt * c + nmTot * e;

        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, s_wsp0,
                outptr, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, inptr, outptr, s_wsp1, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<SCALE, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, inptr, s_out1ptr, s_wsp1, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D SumFacTOP inner product, quadrature
 * metric included.
 *
 * Unpacks the element sizes from @p sizeParam3D, lays out shared memory
 * and copies the three basis tables in, then loops over the block's
 * elements, staging each element's weighted values and calling the shape
 * kernel selected at compile time. The shared-memory regions follow the
 * order s_wsp0 (the element's nq0 * nq1 * nq2 values), s_wsp1 and s_wsp2
 * (the two intermediate stages), the three basis tables and, for the nodal
 * shapes, s_out1ptr for the modal coefficients MatVecSumFacTOPKernel maps
 * onto @p out. Each table's row count follows the shape's mode ordering --
 * the local nmode0, nmode1 and nmode2 -- which is why the collapsed shapes
 * need more rows than nm1 or nm2. Staging the element's values is where
 * the quadrature metric enters, in * jac * w0 * w1 * w2 per point.
 *
 * @note For NodalTet and NodalPrism the shape kernel is instantiated with
 * this launcher's @p APPEND, whereas the 2D launcher instantiates the
 * NodalTri kernel with APPEND false and lets the flag govern the nodal
 * mapping alone. The modal slice s_out1ptr is carved out of @p shmemptr
 * outside the element loop and is neither zeroed nor reset between
 * elements, so under APPEND the first element of a thread block
 * accumulates onto uninitialised shared memory and every later one onto
 * the previous element's values, before MatVecSumFacTOPKernel accumulates
 * the mapped result onto @p out. IProductWRTBase itself launches with
 * APPEND false only.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam Implementation   Operators::SumFacTOP.
 * @tparam SCALE,APPEND     Passed through to the shape kernel; for the
 *                          nodal shapes @p APPEND also governs the nodal
 *                          mapping.
 * @tparam DEFORMED         @p jac holds one value per quadrature point
 *                          rather than a single per-element value; read by
 *                          the staging loop, not passed to the shape
 *                          kernel.
 * @tparam TSizeParameter3D Size-parameter type carrying the mode and point
 *                          counts.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam3D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0,index1,index2    Mode-index tables of the collapsed
 *                      shapes.
 * @param   basis0,basis1,basis2    Basis tables per direction.
 * @param   w0,w1,w2    Quadrature weights per direction.
 * @param   nodToMod    Nodal-to-modal matrix; nodal shapes only.
 * @param   jac         Block's Jacobians.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Global-memory workspace; unused by SumFacTOP, which
 *                      works entirely in shared memory.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified, const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();

    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset0 = 0, offset1 = 0, nmode0 = 0, nmode1 = 0, nmode2 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        nmode0  = nm0;
        nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *s_wsp0    = (TData *)shmemptr;
    TData *s_wsp1    = s_wsp0 + nqTot;
    TData *s_wsp2    = s_wsp1 + offset0;
    TData *s_basis0  = s_wsp2 + offset1;
    TData *s_basis1  = s_basis0 + nmode0 * nq0;
    TData *s_basis2  = s_basis1 + nmode1 * nq1;
    TData *s_out1ptr = s_basis2 + nmode2 * nq2;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    for (unsigned int idx = idx0; idx < nmode2 * nq2; idx += stride)
    {
        s_basis2[idx] = basis2[idx];
    }

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nqTot * nelmt * c + nqTot * e;
        TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);
            if constexpr (DEFORMED)
            {
                s_wsp0[idx] = inptr[idx] * jacptr[idx] * w0[i] * w1[j] * w2[k];
            }
            else
            {
                s_wsp0[idx] = inptr[idx] * jacptr[0] * w0[i] * w1[j] * w2[k];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, s_wsp0, outptr,
                s_wsp1, s_wsp2, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, s_wsp0, s_out1ptr,
                s_wsp1, s_wsp2, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, s_wsp0, outptr,
                s_wsp1, s_wsp2, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, s_wsp0, s_out1ptr,
                s_wsp1, s_wsp2, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, s_wsp0, outptr, s_wsp1,
                s_wsp2, scale, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D SumFacTOP inner product without the
 * quadrature metric; selected by the absence of the weight and Jacobian
 * arguments.
 *
 * The same shared-memory layout and element loop as the quadrature
 * overload, copying the element's values unchanged; the note on @p APPEND
 * for the nodal shapes applies here too.
 *
 * @note As in two dimensions, the shape branches do not read those staged
 * values alike: Hex passes the shared copy s_wsp0 to its shape kernel,
 * whereas Tet, NodalTet, Prism, NodalPrism and Pyr pass inptr, the
 * element's values in global memory. The copy into s_wsp0 still runs in
 * every case.
 *
 * @param   sizeParam3D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0,index1,index2    Mode-index tables of the collapsed
 *                      shapes.
 * @param   basis0,basis1,basis2    Basis tables per direction.
 * @param   nodToMod    Nodal-to-modal matrix; nodal shapes only.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Global-memory workspace; unused by SumFacTOP.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, typename TSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified, const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData scale,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();

    const unsigned int nqTot = nq0 * nq1 * nq2;

    unsigned int offset0 = 0, offset1 = 0, nmode0 = 0, nmode1 = 0, nmode2 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        nmode0  = nm0;
        nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nq1 * nq2;
        offset1 = nm0 * nm1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *s_wsp0    = (TData *)shmemptr;
    TData *s_wsp1    = s_wsp0 + nqTot;
    TData *s_wsp2    = s_wsp1 + offset0;
    TData *s_basis0  = s_wsp2 + offset1;
    TData *s_basis1  = s_basis0 + nmode0 * nq0;
    TData *s_basis2  = s_basis1 + nmode1 * nq1;
    TData *s_out1ptr = s_basis2 + nmode2 * nq2;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    for (unsigned int idx = idx0; idx < nmode2 * nq2; idx += stride)
    {
        s_basis2[idx] = basis2[idx];
    }

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nqTot * nelmt * c + nqTot * e;
        TData *outptr      = out + nmTot * nelmt * c + nmTot * e;

        // Copy to shared memory.
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, inptr, outptr,
                s_wsp1, s_wsp2, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, inptr, s_out1ptr,
                s_wsp1, s_wsp2, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, inptr, outptr,
                s_wsp1, s_wsp2, scale, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, inptr, s_out1ptr,
                s_wsp1, s_wsp2, scale, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<APPEND, true>(nmTot, nodToMod, s_out1ptr,
                                                outptr, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<SCALE, APPEND>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, inptr, outptr, s_wsp1,
                s_wsp2, scale, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
