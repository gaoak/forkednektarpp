///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransDeviceSumFacTOPKernels.hpp
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
// Description: Device SumFacTOP kernels of the backward transform, one
// element per thread block
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernels of the backward transform: one element
 * per thread block, with the coefficients, the basis tables and the
 * sum-factorisation intermediates staged in shared memory.
 *
 * These are the kernels the Device/SumFacTOP block implementation
 * launches (see BwdTransDeviceSumFac.hpp); the SumFac counterparts,
 * where a single thread owns a whole element, are in
 * BwdTransDeviceSumFacKernels.hpp. The other sum-factorised operator
 * families include this header for the same kernels, so their
 * signatures are shared. Everything below is compiled only into
 * device translation units of a device-enabled build
 * (NEKTAR_ENABLE_DEVICE and DEVICE_COMPILE_ONLY).
 *
 * ### Work decomposition
 * The threads of one block cooperate on one element, indexed over the
 * entries a sum-factorisation stage produces rather than over elements.
 * The block storage is left non-interleaved (width 1), so an element's
 * values are contiguous.
 *
 * The BwdTransKernelLauncher overloads take the element of their block's
 * first-dimension index and stride by the grid's first-dimension size;
 * within a stage each thread starts at its local thread index and
 * strides by the block size. The grid's second dimension instead holds
 * one block-column per component, `c = getBlockIdx<1>(threadBlock)`,
 * fixed for a block's whole element loop, so a single launch covers
 * every component (see BwdTransDeviceSumFac.hpp). Neither loop
 * constrains the launch configuration, which is why the block size may
 * be capped -- GetDeviceBlockSize rounds the per-element work count it
 * is given, the element's total mode count, up to whole warps and caps
 * it at the device's default block size -- and the grid sized by an
 * occupancy heuristic (GetDeviceGridSize).
 *
 * ### Shared memory and synchronisation
 * In two and three dimensions a block copies the basis tables into
 * shared memory once before its element loop; for each element it then
 * stages the coefficients and the intermediates of all but the last
 * sum-factorisation stage (one intermediate in two dimensions, two in
 * three) there as well, the last stage writing its physical values
 * straight to global memory. Every stage is closed by a localBarrier,
 * including the last one of each shape kernel, so that the staging area
 * is safe to overwrite for the next element the block takes; the barrier
 * that ends the coefficient staging is also what publishes the basis
 * tables. The one-dimensional path stages nothing at all: it reads
 * coefficients and basis straight from global memory.
 *
 * BwdTransSharedMemorySize gives the amount of dynamic shared memory the
 * block implementation requests. In one dimension it still returns a
 * non-zero figure, for storage the kernels never touch (see its 1D
 * overload).
 *
 * ### Conventions
 * The basis tables, mode orderings, isModified corrections and nodal
 * conversion are those of the SumFac kernels -- see the file notes of
 * BwdTransDeviceSumFacKernels.hpp -- with an element's data addressed
 * directly rather than through a warp lane offset.
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
 * @brief Global-memory workspace a 1D SumFacTOP backward transform
 * needs, in TData values: none.
 *
 * SumFacTOP holds every intermediate in shared memory, so this and the
 * 2D and 3D overloads all return 0. The block implementation queries
 * them along the same path as for SumFac and simply ends up with an
 * empty workspace (see BwdTransDeviceSumFac.hpp).
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    MultiRegions::SumFacTOP; the SumFac overload
 *                           lives in BwdTransDeviceSumFacKernels.hpp.
 * @tparam TSizeParameter1D  1D size parameter, runtime or templated.
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 *
 * @return 0.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t BwdTransWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Global-memory workspace a 2D SumFacTOP backward transform
/// needs: none, for the reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t BwdTransWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    return 0;
}

/// @brief Global-memory workspace a 3D SumFacTOP backward transform
/// needs: none, for the reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t BwdTransWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter3D sizeParam3D)
{
    return 0;
}

/**
 * @brief Dynamic shared memory a 1D SumFacTOP backward transform
 * requests, in TData values.
 *
 * Counts the element's nm0 coefficients plus the nm0 x nq0 basis table.
 * The 1D path stages neither of them: the launcher passes the
 * shared-memory pointer on unused and BwdTransSegSumFacTOPKernel reads
 * coefficients and basis straight from global memory, so the memory
 * this reserves is never written. It does, however, enter the occupancy
 * heuristic of GetDeviceGridSize.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TSizeParameter1D  1D size parameter, runtime or templated.
 *
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 *
 * @return Number of TData values of shared memory per thread block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int BwdTransSharedMemorySize(
    const TSizeParameter1D sizeParam1D)
{
    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    return nm0 + nm0 * nq0;
}

/**
 * @brief Dynamic shared memory a 2D SumFacTOP backward transform
 * requests, in TData values.
 *
 * The sum of the pieces the 2D launcher stages: the element's nmTot
 * coefficients, the intermediate of the first stage (nq0 * nm1 values
 * for the quadrilateral, nm0 * nq1 for the triangles) and the two basis
 * tables. The direction-1 table has nm1 rows for the quadrilateral and,
 * that direction being collapsed, nmTot rows for the triangles.
 *
 * @tparam SHAPE_TYPE        Quad, Tri or NodalTri.
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TSizeParameter2D  2D size parameter, runtime or templated.
 *
 * @param   sizeParam2D  Modal and quadrature sizes per direction.
 *
 * @return Number of TData values of shared memory per thread block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int BwdTransSharedMemorySize(
    const TSizeParameter2D sizeParam2D)
{
    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return nq0 * nm0 + nq1 * nm1 + nmTot + nq0 * nm1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return nm0 * nq0 + nmTot * nq1 + nmTot + nm0 * nq1;
    }
}

/**
 * @brief Dynamic shared memory a 3D SumFacTOP backward transform
 * requests, in TData values.
 *
 * The sum of the pieces the 3D launcher stages: the element's nmTot
 * coefficients, the intermediates of the first two stages and the three
 * basis tables. Writing nm01, nm02 and nmode2 for the combined-mode
 * counts computed at the top of the function, the table and stage sizes
 * are
 * - Hex: tables of nm0, nm1 and nm2 rows; stages nq0 * nm1 * nm2 and
 *   nq0 * nq1 * nm2;
 * - Tet and NodalTet: tables of nm0, nm01 and nmode2 rows; stages
 *   nm01 * nq2 and nm0 * nq1 * nq2;
 * - Prism and NodalPrism: tables of nm0, nm1 and nm02 rows; stages
 *   nm0 * nm1 * nq2 and nm0 * nq1 * nq2;
 * - Pyr: tables of nm0, nm1 and nmode2 rows, with the prism's stage
 *   sizes.
 *
 * The same expressions appear in the 3D launcher, which derives the
 * shared-memory pointers from them; the two must agree.
 *
 * @tparam SHAPE_TYPE        Hex, Tet, NodalTet, Prism, NodalPrism or
 *                           Pyr.
 * @tparam Implementation    MultiRegions::SumFacTOP.
 * @tparam TSizeParameter3D  3D size parameter, runtime or templated.
 *
 * @param   sizeParam3D  Modal and quadrature sizes per direction.
 *
 * @return Number of TData values of shared memory per thread block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int BwdTransSharedMemorySize(
    const TSizeParameter3D sizeParam3D)
{
    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return nq0 * nm0 + nq1 * nm1 + nq2 * nm2 + nmTot + (nq0 * nm1 * nm2) +
               (nq0 * nq1 * nm2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nmTot + (nm01 * nq2) +
               (nm0 * nq1 * nq2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + nmTot + (nm0 * nm1 * nq2) +
               (nm0 * nq1 * nq2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nmTot +
               (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
    }
}

/**
 * @brief Segment SumFacTOP kernel: evaluate one element's expansion,
 * the block's threads sharing out the quadrature points.
 *
 * Each thread takes the points at its local index plus multiples of the
 * block size and contracts the element's nm0 coefficients against the
 * basis table for each of them. Nothing is staged: @p in and @p basis0
 * are read from global memory. The closing barrier brings the block back
 * into step before it takes the next element.
 *
 * @tparam APPEND        Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock  Back-end thread-block handle type.
 *
 * @param   nm0         Modes of the expansion.
 * @param   nq0         Quadrature points of the expansion.
 * @param   basis0      Basis table: `basis0[p * nq0 + i]` is mode p at
 *                      point i.
 * @param   in          This element's nm0 coefficients.
 * @param   out         This element's nq0 physical values.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransSegSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; p++)
        {
            tmp += in[p] * basis0[p * nq0 + i];
        }

        if constexpr (APPEND)
        {
            out[i] += tmp;
        }
        else
        {
            out[i] = tmp;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Quadrilateral SumFacTOP kernel: tensor-product evaluation of
 * one element in two barrier-separated stages.
 *
 * Stage one shares out the nq0 * nm1 direction-0 sums: thread index
 * `idx` takes direction-1 mode `idx % nm1` at direction-0 point
 * `idx / nm1`, contracts the nm0 coefficients `in[q * nm0 + p]` against
 * @p basis0 and writes `wsp[idx]`. Stage two shares out the nqTot
 * output points: `idx` takes the point (`idx % nq0`, `idx / nq0`) and
 * contracts the nm1 values `wsp[i * nm1 + q]` against @p basis1. Both
 * loops stride by the block size, so any block size covers any element
 * size.
 *
 * @tparam APPEND        Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock  Back-end thread-block handle type.
 *
 * @param   nm0,nm1     Modes per direction.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   nqTot       Total quadrature points, nq0 * nq1.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table (nm1 rows of nq1).
 * @param   in          This element's staged coefficients.
 * @param   out         This element's nqTot physical values,
 *                      direction-0 point running fastest.
 * @param   wsp         Staging area for the first stage, nq0 * nm1
 *                      values.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransQuadSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nqTot,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nq0 * nm1; idx += stride)
    {
        const unsigned int q = idx % nm1;
        const unsigned int i = idx / nm1;
        unsigned int cnt_qp  = nm0 * q;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
        {
            tmp += in[cnt_qp] * basis0[p * nq0 + i];
        }
        wsp[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;
        unsigned int cnt_iq  = nm1 * i;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
        {
            tmp += wsp[cnt_iq] * basis1[q * nq1 + j];
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

    localBarrier(threadBlock);
}

/**
 * @brief Triangular SumFacTOP kernel: evaluation of one element in the
 * collapsed coordinate system, in two barrier-separated stages.
 *
 * The collapsed direction goes first. Stage one shares out the
 * nm0 * nq1 pairs (p, direction-1 point j): thread `idx` takes
 * `p = idx % nm0`, `j = idx / nm0`, starts at the first mode of p in
 * the triangular ordering, `(2 * nm1 - p + 1) * p / 2`, and contracts
 * its nm1 - p modes against @p basis1, whose rows carry the same
 * combined index. Stage two shares out the nqTot output points and
 * contracts the nm0 values `wsp[j * nm0 + p]` against @p basis0, adding
 * the @p isModified term `in[1] * basis0[nq0 + i] * basis1[nq1 + j]`
 * when asked.
 *
 * @tparam APPEND        Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock  Back-end thread-block handle type.
 *
 * @param   nm0,nm1     Modes per direction.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   nqTot       Total quadrature points, nq0 * nq1.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table, one row of nq1 per
 *                      combined (p,q) mode.
 * @param   in          This element's staged coefficients, in the
 *                      triangular mode ordering.
 * @param   out         This element's nqTot physical values.
 * @param   wsp         Staging area for the first stage, nm0 * nq1
 *                      values.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransTriSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nqTot, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nm0 * nq1; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int j = idx / nm0;
        unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            tmp += in[mode_pq] * basis1[mode_pq * nq1 + j];
        }
        wsp[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;
        unsigned int cnt_jp  = nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            tmp += in[1] * basis0[nq0 + i] * basis1[nq1 + j];
        }

#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_jp)
        {
            tmp += wsp[cnt_jp] * basis0[p * nq0 + i];
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

    localBarrier(threadBlock);
}

/**
 * @brief Hexahedral SumFacTOP kernel: three-direction tensor-product
 * evaluation of one element in three barrier-separated stages.
 *
 * Stage one shares out the nq0 * nm1 * nm2 direction-0 sums, thread
 * `idx` taking mode pair (`idx % nm1`, `(idx / nm1) % nm2`) at point
 * `idx / (nm1 * nm2)` and writing `wsp0[idx]`, so that @p wsp0 is
 * ordered (i, r, q) with q fastest. Stage two contracts direction 1 into
 * @p wsp1 over the nq0 * nq1 * nm2 triples (r, i, j), r fastest, and
 * stage three contracts direction 2 into the nqTot output points. Each
 * stage strides its index by the block size.
 *
 * @tparam APPEND        Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock  Back-end thread-block handle type.
 *
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   nqTot       Total quadrature points, nq0 * nq1 * nq2.
 * @param   basis0,basis1,basis2    Per-direction basis tables.
 * @param   in          This element's staged coefficients,
 *                      `in[(r * nm1 + q) * nm0 + p]`.
 * @param   out         This element's nqTot physical values,
 *                      direction-0 point running fastest.
 * @param   wsp0        Staging area for stage one, nq0 * nm1 * nm2
 *                      values.
 * @param   wsp1        Staging area for stage two, nq0 * nq1 * nm2
 *                      values.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransHexSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nq0 * nm1 * nm2; idx += stride)
    {
        const unsigned int q = idx % nm1;
        const unsigned int r = (idx / nm1) % nm2;
        const unsigned int i = idx / (nm1 * nm2);
        unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
        {
            tmp += in[cnt_rqp] * basis0[p * nq0 + i];
        }
        wsp0[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nq0 * nq1 * nm2; idx += stride)
    {
        const unsigned int r = idx % nm2;
        const unsigned int i = (idx / nm2) % nq0;
        const unsigned int j = idx / (nm2 * nq0);
        unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
        {
            tmp += wsp0[cnt_irq] * basis1[q * nq1 + j];
        }
        wsp1[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 2
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = (idx / nq0) % nq1;
        const unsigned int k = idx / (nq1 * nq0);
        unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
        {
            tmp += wsp1[cnt_jir] * basis2[r * nq2 + k];
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

    localBarrier(threadBlock);
}

/**
 * @brief Tetrahedral SumFacTOP kernel: evaluation of one element in the
 * collapsed coordinate system, in three barrier-separated stages.
 *
 * The directions are contracted in the order 2, 1, 0. Stage one shares
 * out the nm01 * nq2 pairs of a (p,q) mode pair and a direction-2 point,
 * where nm01 = (2 * nm1 - nm0 + 1) * nm0 / 2 counts the pairs with
 * q < nm1 - p; @p pindex and @p qindex resolve the flat pair index into
 * p and q without scanning, and the arithmetic that follows gives the
 * pair's first row in @p basis2 (`mode2`) and its first coefficient
 * (`mode_pqr`). The two differ when nm2 > nm1, the basis table also
 * enumerating pairs the coefficients do not carry -- as in the SumFac
 * kernel, where the same offset is accumulated by walking the modes.
 * @p wsp0 comes out ordered (k, pair) with the pair index fastest.
 *
 * Stage two shares out the nm0 * nq1 * nq2 triples (p, j, k), reading
 * the nm1 - p pairs of p from @p wsp0 against @p basis1 into @p wsp1,
 * ordered (k, j, p) with p fastest. Stage three shares out the nqTot
 * output points, contracts @p wsp1 against @p basis0 and, with
 * @p isModified set, adds the top-vertex, bottom-vertex and
 * singular-edge terms marked in the code.
 *
 * @tparam APPEND        Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock  Back-end thread-block handle type.
 *
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   nqTot       Total quadrature points, nq0 * nq1 * nq2.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   pindex      p of each (p,q) mode pair, nm01 entries.
 * @param   qindex      q of each (p,q) mode pair, nm01 entries.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table, one row of nq1 per
 *                      combined (p,q) mode.
 * @param   basis2      Direction-2 basis table, one row of nq2 per
 *                      combined (p,q,r) mode.
 * @param   in          This element's staged coefficients, in the
 *                      tetrahedral mode ordering.
 * @param   out         This element's nqTot physical values.
 * @param   wsp0        Staging area for stage one, nm01 * nq2 values.
 * @param   wsp1        Staging area for stage two, nm0 * nq1 * nq2
 *                      values.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransTetSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const bool isModified,
    const unsigned int *NEK_RESTRICT pindex,
    const unsigned int *NEK_RESTRICT qindex, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TthreadBlock &threadBlock)
{
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    // direction 2
    for (unsigned int idx = idx0; idx < nm01 * nq2; idx += stride)
    {
        const unsigned int k = idx / nm01;
        const unsigned int p = pindex[idx % nm01];
        const unsigned int q = qindex[idx % nm01];
        unsigned int mode2   = (2u * (nm2 - p) - q + 1u) * q;
        mode2 += nm2 * (nm2 + 1u) * p;
        mode2 -= (2u * nm2 + 1u) * (p - 1u) * p / 2u;
        mode2 += (p - 1u) * p * (2u * p - 1u) / 6u;
        mode2 /= 2u;
        unsigned int mode_pqr =
            mode2 -
            ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode2, ++mode_pqr)
        {
            tmp += in[mode_pqr] * basis2[k + nq2 * mode2];
        }
        wsp0[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int j = (idx / nm0) % nq1;
        const unsigned int k = idx / (nm0 * nq1);
        unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;
        unsigned int cnt_kpq = nm01 * k + mode_pq;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1 - p; ++q, ++cnt_kpq, ++mode_pq)
        {
            tmp += wsp0[cnt_kpq] * basis1[mode_pq * nq1 + j];
        }
        wsp1[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int j  = (idx / nq0) % nq1;
        const unsigned int k  = idx / (nq0 * nq1);
        unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            // top vertex
            tmp += basis0[i] * basis1[nq1 + j];
            tmp += basis0[nq0 + i] * basis1[j];
            tmp += basis0[nq0 + i] * basis1[nq1 + j];
            tmp *= basis2[nq2 + k] * in[1];

            // bottom vertex
            TData tmp1 = basis2[k] * in[nm2];

            // singular edge
#pragma unroll
            for (unsigned int r = 1u; r < nm2 - 1u; ++r)
            {
                tmp1 += basis2[(r + 1u) * nq2 + k] * in[nm2 + r];
            }
            tmp += basis1[nq1 + j] * basis0[nq0 + i] * tmp1;
        }

#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
        {
            tmp += wsp1[mode_kjp] * basis0[p * nq0 + i];
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

    localBarrier(threadBlock);
}

/**
 * @brief Prismatic SumFacTOP kernel: evaluation of one element in the
 * collapsed coordinate system, in three barrier-separated stages.
 *
 * Only direction 2 is collapsed: for each p the coefficients carry
 * nm2 - p modes r for every one of the nm1 modes q, and @p basis2 holds
 * one row per combined (p,r) mode. Stage one shares out the
 * nm0 * nm1 * nq2 triples (q, p, k), each thread computing the start of
 * its p block in the basis table, `(2 * nm2 - p + 1) * p / 2`, and of
 * its (p,q) run of coefficients, and writing @p wsp0 ordered (k, p, q)
 * with q fastest. Stage two contracts direction 1 into @p wsp1 over the
 * nm0 * nq1 * nq2 triples (p, j, k), p fastest, and stage three
 * contracts direction 0 into the nqTot output points, adding with
 * @p isModified set the collapsed-vertex term summed over q in the
 * code.
 *
 * @tparam APPEND        Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock  Back-end thread-block handle type.
 *
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   nqTot       Total quadrature points, nq0 * nq1 * nq2.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table (nm1 rows of nq1).
 * @param   basis2      Direction-2 basis table, one row of nq2 per
 *                      combined (p,r) mode.
 * @param   in          This element's staged coefficients, in the
 *                      prismatic mode ordering.
 * @param   out         This element's nqTot physical values.
 * @param   wsp0        Staging area for stage one, nm0 * nm1 * nq2
 *                      values.
 * @param   wsp1        Staging area for stage two, nm0 * nq1 * nq2
 *                      values.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransPrismSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    // direction 2
    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int q  = idx % nm1;
        const unsigned int p  = (idx / nm1) % nm0;
        const unsigned int k  = idx / (nm1 * nm0);
        unsigned int mode_pr  = (2u * nm2 - p + 1u) * p / 2u;
        unsigned int mode_pqr = mode_pr * nm1 + (nm2 - p) * q;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr, ++mode_pr)
        {
            tmp += in[mode_pqr] * basis2[mode_pr * nq2 + k];
        }
        wsp0[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int p  = idx % nm0;
        const unsigned int j  = (idx / nm0) % nq1;
        const unsigned int k  = idx / (nm0 * nq1);
        unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
        {
            tmp += wsp0[mode_kpq] * basis1[q * nq1 + j];
        }
        wsp1[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int j  = (idx / nq0) % nq1;
        const unsigned int k  = idx / (nq0 * nq1);
        unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
#pragma unroll
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                tmp += in[q * nm2 + 1u] * basis1[q * nq1 + j];
            }
            tmp *= basis2[nq2 + k] * basis0[nq0 + i];
        }

#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
        {
            tmp += wsp1[mode_kjp] * basis0[p * nq0 + i];
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

    localBarrier(threadBlock);
}

/**
 * @brief Pyramidal SumFacTOP kernel: evaluation of one element in the
 * collapsed coordinate system, in three barrier-separated stages.
 *
 * Structured like the prismatic kernel, except that the direction-2 mode
 * count of a (p,q) pair is nm2 - max(p,q) and @p basis2 holds one row
 * per combined (p,q,r) mode. Stage one's closed-form arithmetic
 * therefore yields both the pair's first coefficient (`mode_pqr`) and
 * its first basis row (`mode2`), which sits further on when nm2 > nm1
 * because the table also enumerates pairs the coefficients do not carry.
 * Stages two and three are those of the prism, the third adding the
 * top-vertex term with @p isModified set.
 *
 * @tparam APPEND        Accumulate onto @p out instead of overwriting.
 * @tparam TthreadBlock  Back-end thread-block handle type.
 *
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   nqTot       Total quadrature points, nq0 * nq1 * nq2.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table (nm1 rows of nq1).
 * @param   basis2      Direction-2 basis table, one row of nq2 per
 *                      combined (p,q,r) mode.
 * @param   in          This element's staged coefficients, in the
 *                      pyramidal mode ordering.
 * @param   out         This element's nqTot physical values.
 * @param   wsp0        Staging area for stage one, nm0 * nm1 * nq2
 *                      values.
 * @param   wsp1        Staging area for stage two, nm0 * nq1 * nq2
 *                      values.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransPyrSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    // direction 2
    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int q = idx % nm1;
        const unsigned int p = (idx / nm1) % nm0;
        const unsigned int k = idx / (nm1 * nm0);
        unsigned int mode2 =
            (nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u;
        unsigned int mode_pqr = nm1 * (2u * nm2 + 1u - nm1) * p;
        mode_pqr -= (p - 1u) * p / 2u;
        mode_pqr -= (p - 1u) * p * (2u * p - 1u) / 6u;
        mode_pqr /= 2u;
        mode_pqr += (q < p)
                        ? q * (nm2 - p)
                        : p * (nm2 - p) +
                              ((2u * (nm2 - p) - (q - p) + 1u) * (q - p)) / 2u;
        mode2 += mode_pqr;

        TData tmp             = 0.0;
        const unsigned ulimit = (q < p) ? nm2 - p : nm2 - q;
#pragma unroll
        for (unsigned int r = 0u; r < ulimit; ++r, ++mode2, ++mode_pqr)
        {
            tmp += in[mode_pqr] * basis2[mode2 * nq2 + k];
        }
        wsp0[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int p  = idx % nm0;
        const unsigned int j  = (idx / nm0) % nq1;
        const unsigned int k  = idx / (nm0 * nq1);
        unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
        {
            tmp += wsp0[mode_kpq] * basis1[q * nq1 + j];
        }
        wsp1[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int j  = (idx / nq0) % nq1;
        const unsigned int k  = idx / (nq0 * nq1);
        unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            // top vertex
            tmp += basis0[i] * basis1[nq1 + j];
            tmp += basis0[nq0 + i] * basis1[j];
            tmp += basis0[nq0 + i] * basis1[nq1 + j];
            tmp *= basis2[nq2 + k] * in[1];
        }

#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
        {
            tmp += wsp1[mode_kjp] * basis0[p * nq0 + i];
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

    localBarrier(threadBlock);
}

/**
 * @brief Device entry point of the 1D backward transform under
 * SumFacTOP: unpack the size parameter, then walk the block's elements,
 * one thread block per element, applying the segment kernel to each.
 *
 * Launched through DEVICE_2DGRID_KERNEL_LAUNCHER, which appends the last
 * two arguments and supplies grid, block, shared-memory size and stream
 * (see BwdTransDeviceSumFac.hpp). The defaulted Enable template
 * parameter restricts this definition to the SumFacTOP tag, so that the
 * SumFac file can define an overload of the same name for its own tag;
 * anything but a 1D size parameter is rejected by the static_assert.
 * FETCH_SHARED_MEMORY binds @p shmemptr to the block's dynamic shared
 * memory on the CUDA/HIP back-ends, and is a no-op where the launcher
 * macro passes the pointer in; nothing is staged in shared memory in one
 * dimension, so it goes unused once unpacked.
 *
 * The block takes the element at its index in the grid's first dimension
 * and advances by that dimension's block count, so the block's
 * elements -- padding included, their results being discarded -- are
 * covered however many blocks were launched.
 *
 * @tparam SHAPE_TYPE       Seg.
 * @tparam Implementation   MultiRegions::SumFacTOP.
 * @tparam APPEND           Accumulate onto @p out instead of
 *                          overwriting.
 * @tparam TSizeParameter1D 1D size parameter, in the runtime or the
 *                          compile-time form; with the latter the sizes
 *                          below are compile-time constants.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 *
 * @param   sizeParam1D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A; the segment
 *                      kernel needs no correction, so unused.
 * @param   basis0      Basis table in device memory.
 * @param   nodToMod    Nodal-to-modal matrix; no 1D nodal shape, so
 *                      unused.
 * @param   in          Block's coefficients for every component.
 * @param   out         Block's physical values for every component.
 * @param   wsp         Global-memory workspace; SumFacTOP keeps no
 *                      intermediates there, so unused.
 * @param   shmemptr    Shared-memory base, appended by the launcher;
 *                      unused in one dimension.
 * @param   threadBlock Thread-block handle, appended by the launcher.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    BwdTransKernelLauncher(const TSizeParameter1D sizeParam1D,
                           const size_t nelmt,
                           [[maybe_unused]] const bool isModified,
                           const TData *NEK_RESTRICT basis0,
                           [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
                           const TData *NEK_RESTRICT in,
                           TData *NEK_RESTRICT out,
                           [[maybe_unused]] TData *NEK_RESTRICT wsp,
                           [[maybe_unused]] unsigned char *shmemptr,
                           const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nm0 * nelmt * c + nm0 * e;
        TData *outptr      = out + nq0 * nelmt * c + nq0 * e;
        BwdTransSegSumFacTOPKernel<APPEND>(nm0, nq0, basis0, inptr, outptr,
                                           threadBlock);
        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D backward transform under
 * SumFacTOP: stage the basis tables, then walk the block's elements, one
 * thread block per element, applying the shape kernel selected by
 * @p SHAPE_TYPE to each.
 *
 * Constrained, launched and bounded as the 1D entry point, but here
 * @p shmemptr is carved up: the two basis tables are copied in once
 * before the element loop, and each element then stages its
 * coefficients and the first stage's intermediate. The pieces and their
 * sizes are those BwdTransSharedMemorySize adds up, and the two must
 * agree. For NodalTri the staged coefficients are converted to the
 * modified modal basis before the triangular kernel runs on them.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam Implementation   MultiRegions::SumFacTOP.
 * @tparam APPEND           Accumulate onto @p out instead of
 *                          overwriting.
 * @tparam TSizeParameter2D 2D size parameter, runtime or compile-time
 *                          form.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 *
 * @param   sizeParam2D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0,basis1   Per-direction basis tables.
 * @param   nodToMod    Nodal-to-modal matrix; nodal shapes only.
 * @param   in          Block's coefficients for every component.
 * @param   out         Block's physical values for every component.
 * @param   wsp         Global-memory workspace; unused by SumFacTOP.
 * @param   shmemptr    Shared-memory base, appended by the launcher.
 * @param   threadBlock Thread-block handle, appended by the launcher.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    BwdTransKernelLauncher(
        const TSizeParameter2D sizeParam2D, const size_t nelmt,
        const bool isModified, const TData *NEK_RESTRICT basis0,
        const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
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

    unsigned int offset{0}, nmode0{0}, nmode1{0};
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = nm1 * nq0;
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

    TData *s_wsp0   = (TData *)shmemptr;
    TData *s_wsp1   = s_wsp0 + nmTot;
    TData *s_basis0 = s_wsp1 + offset;
    TData *s_basis1 = s_basis0 + nm0 * nq0;

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
        const TData *inptr = in + nmTot * nelmt * c + nmTot * e;
        TData *outptr      = out + nqTot * nelmt * c + nqTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            // Nodal to Modal in shared memory
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, s_wsp0, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                s_wsp0[idx] = inptr[idx];
            }
            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            BwdTransQuadSumFacTOPKernel<APPEND>(nm0, nm1, nq0, nq1, nqTot,
                                                s_basis0, s_basis1, s_wsp0,
                                                outptr, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            BwdTransTriSumFacTOPKernel<APPEND>(
                nm0, nm1, nq0, nq1, nqTot, isModified, s_basis0, s_basis1,
                s_wsp0, outptr, s_wsp1, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D backward transform under
 * SumFacTOP: stage the basis tables, then walk the block's elements, one
 * thread block per element, applying the shape kernel selected by
 * @p SHAPE_TYPE to each.
 *
 * As the 2D entry point, with three basis tables and two stage
 * intermediates staged instead of two and one; the shared-memory pieces
 * and their sizes are those BwdTransSharedMemorySize adds up. The
 * mode-index tables are used by the tetrahedral kernel, which resolves a
 * flat (p,q) pair index through them; they are null for every other
 * shape. For NodalTet and NodalPrism the staged coefficients are
 * converted to the modified modal basis before the shape kernel runs on
 * them.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism or
 *                          Pyr.
 * @tparam Implementation   MultiRegions::SumFacTOP.
 * @tparam APPEND           Accumulate onto @p out instead of
 *                          overwriting.
 * @tparam TSizeParameter3D 3D size parameter, runtime or compile-time
 *                          form.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 *
 * @param   sizeParam3D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0,index1   Mode-index tables; tetrahedra only.
 * @param   basis0,basis1,basis2    Per-direction basis tables.
 * @param   nodToMod    Nodal-to-modal matrix; nodal shapes only.
 * @param   in          Block's coefficients for every component.
 * @param   out         Block's physical values for every component.
 * @param   wsp         Global-memory workspace; unused by SumFacTOP.
 * @param   shmemptr    Shared-memory base, appended by the launcher.
 * @param   threadBlock Thread-block handle, appended by the launcher.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    BwdTransKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified, const unsigned int *index0,
        const unsigned int *index1, const TData *NEK_RESTRICT basis0,
        const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
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

    unsigned int offset0{0}, offset1{0}, nmode0{0}, nmode1{0}, nmode2{0};
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = nq0 * nm1 * nm2;
        offset1 = nq0 * nq1 * nm2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        offset0 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *s_wsp0   = (TData *)shmemptr;
    TData *s_wsp1   = s_wsp0 + nmTot;
    TData *s_wsp2   = s_wsp1 + offset0;
    TData *s_basis0 = s_wsp2 + offset1;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_basis2 = s_basis1 + nmode1 * nq1;

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
        const TData *inptr = in + nmTot * nelmt * c + nmTot * e;
        TData *outptr      = out + nqTot * nelmt * c + nqTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism ||
                      SHAPE_TYPE == LibUtilities::NodalTet)
        {
            // Nodal to Modal in shared memory
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, s_wsp0, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                s_wsp0[idx] = inptr[idx];
            }
            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            BwdTransHexSumFacTOPKernel<APPEND>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            BwdTransTetSumFacTOPKernel<APPEND>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, index0, index1,
                s_basis0, s_basis1, s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2,
                threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            BwdTransPrismSumFacTOPKernel<APPEND>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2,
                threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            BwdTransPyrSumFacTOPKernel<APPEND>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, s_wsp0, outptr, s_wsp1, s_wsp2,
                threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
