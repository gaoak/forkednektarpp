///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransDeviceSumFacKernels.hpp
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
// Description: Device SumFac kernels of the backward transform, one
// element per thread
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransDeviceSumFacKernels.hpp
 * @brief Device SumFac kernels of the backward transform: one element
 * per thread, with the sum-factorisation intermediates held in a
 * global-memory workspace.
 *
 * These are the kernels the Device/SumFac block implementation launches
 * (see BwdTransDeviceSumFac.hpp); the SumFacTOP counterparts, where a
 * whole thread block cooperates on one element, are in
 * BwdTransDeviceSumFacTOPKernels.hpp. The other sum-factorised operator
 * families include this header for the same kernels, so their
 * signatures are shared. Everything below is compiled only into
 * device translation units of a device-enabled build
 * (NEKTAR_ENABLE_DEVICE and DEVICE_COMPILE_ONLY).
 *
 * ### Work decomposition and data layout
 * One thread evaluates one element. The block storage is reshaped to
 * interleave width warpSize before the launch, so that the elements of a
 * warp are interleaved: element `e` is handled by lane
 * `ilane = e % warpSize` of warp `iwarp = e / warpSize`, and value `n`
 * of that element sits at offset `warpSize * n + ilane` within its
 * warp's group of `num_data * warpSize` values. Consecutive lanes
 * therefore address consecutive words.
 *
 * The BwdTransKernelLauncher overloads stride over the elements of the
 * block: the stride, `getGlobalRange<0>(threadBlock)`, is a whole number
 * of warps -- the launch uses one warp per block, see
 * GetDeviceBlockSize -- so a thread keeps its lane index for every
 * element it visits. Each thread's component index,
 * `c = getBlockIdx<1>(threadBlock)`, is fixed for that whole stride
 * loop: the grid's second dimension holds `ncomp = GetNumComponents() *
 * GetNumHomoModes()` blocks, one per component, so a single launch
 * covers every component of the block (see BwdTransDeviceSumFac.hpp).
 *
 * ### Sum factorisation
 * Each shape kernel contracts the coefficients against one direction's
 * 1D basis table at a time, holding the intermediates in the workspace
 * slices (`wsp` in two dimensions, a pair of slices in three) that
 * BwdTransWorkSpaceSize accounts for. A basis table `basisN` holds one
 * row of nqN values per mode, so that `basisN[m * nqN + i]` is mode `m`
 * evaluated at point `i`; along the collapsed directions of the
 * triangle, tetrahedron, prism and pyramid the row index `m` is the
 * shape's combined mode index rather than a single-direction mode
 * number. Coefficients are read in that same mode ordering: for the
 * tensor-product shapes (quadrilateral, hexahedron) the direction-0 mode
 * index runs fastest, whereas along a collapsed direction it is the
 * innermost index of the combined ordering that does -- q for the
 * triangle, r for the tetrahedron, prism and pyramid. Physical values
 * are written in point order, with the direction-0 point running
 * fastest.
 *
 * ### Modified bases and nodal shapes
 * @p isModified marks a first-direction eModified_A basis, for which the
 * shape kernels add the extra terms the modified expansion carries at
 * its collapsed vertex modes and, for the tetrahedron, at its singular
 * edge; each term is identical to the corresponding one in
 * StdRegions/Operators/BwdTransSumFacStdKernels.hpp. For the nodal
 * shapes the coefficients are first converted to the modified modal
 * basis by MatVecKernel and the modal kernel of the matching non-nodal
 * shape is then run on the converted values.
 *
 * @see BwdTransSerialAVXSumFacKernels.hpp for the Serial/AVX form of the
 * same decomposition, expressed over SIMD vectors instead of warp lanes.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Workspace a 1D SumFac backward transform needs, in TData
 * values: none.
 *
 * The segment kernel contracts its only direction straight into the
 * output and so keeps no intermediates. The overload exists so that the
 * block implementation can query a workspace size uniformly for every
 * dimension (see BwdTransDeviceSumFac.hpp).
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    Operators::SumFac; the SumFacTOP overload
 *                           lives in BwdTransDeviceSumFacTOPKernels.hpp.
 * @tparam TSizeParameter1D  1D size parameter, runtime or templated.
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 *
 * @return 0.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t BwdTransWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    size_t wspsize = 0;

    if constexpr (SHAPE_TYPE == LibUtilities::Seg)
    {
        wspsize = 0;
    }

    return wspsize;
}

/**
 * @brief Workspace a 2D SumFac backward transform needs, in TData
 * values.
 *
 * One flat global-memory region serves the whole launch: a stage needing
 * `s` values per element occupies `s * nelmt` consecutive values,
 * warp-interleaved exactly like the field data, and the stages follow
 * one another (BwdTransKernelLauncher does the slicing). Per element
 * that is
 * - Quad: nm1 values -- the direction-0 sums of one direction-0 point;
 * - Tri: nm0 values -- the direction-1 sums of one direction-1 point;
 * - NodalTri: the nmTot values of the modal-converted coefficients,
 *   followed by the nm0 values of the triangle.
 *
 * Shapes with no 2D SumFac kernel get 0.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Operators::SumFac.
 * @tparam TSizeParameter2D  2D size parameter, runtime or templated.
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam2D  Modal and quadrature sizes per direction.
 *
 * @return Number of TData values the launch's workspace must hold.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t BwdTransWorkSpaceSize(
    const size_t nelmt, const TSizeParameter2D sizeParam2D)
{
    size_t wspsize = 0;

    const unsigned int nm0 = sizeParam2D.nm0();
    const unsigned int nm1 = sizeParam2D.nm1();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        wspsize = nm1 * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        wspsize = nm0 * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nmTot = sizeParam2D.nmTot();

        wspsize = (nm0 + nmTot) * nelmt;
    }

    return wspsize;
}

/**
 * @brief Workspace a 3D SumFac backward transform needs, in TData
 * values.
 *
 * Laid out as in the 2D overload, with the two stage intermediates of
 * the shape kernels following one another. Per element that is
 * - Hex: nm1 * nm2, then nm2;
 * - Tet: (2 * nm1 - nm0 + 1) * nm0 / 2 -- the number of (p,q) mode pairs
 *   the collapsed ordering carries -- then nm0;
 * - Prism and Pyr: nm0 * nm1, then nm0;
 * - the nodal shapes prepend their total mode count nmTot for the
 *   modal-converted coefficients.
 *
 * Shapes with no 3D SumFac kernel get 0.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Operators::SumFac.
 * @tparam TSizeParameter3D  3D size parameter, runtime or templated.
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam3D  Modal and quadrature sizes per direction.
 *
 * @return Number of TData values the launch's workspace must hold.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t BwdTransWorkSpaceSize(
    const size_t nelmt, const TSizeParameter3D sizeParam3D)
{
    size_t wspsize = 0;

    const unsigned int nm0 = sizeParam3D.nm0();
    const unsigned int nm1 = sizeParam3D.nm1();
    const unsigned int nm2 = sizeParam3D.nm2();

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        wspsize = (nm1 * nm2 + nm2) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        wspsize = ((2 * nm1 - nm0 + 1) * nm0 / 2 + nm0) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();

        wspsize = (((2 * nm1 - nm0 + 1) * nm0 / 2 + nm0) + nmTot) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        wspsize = (nm0 * nm1 + nm0) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();

        wspsize = ((nm0 * nm1 + nm0) + nmTot) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        wspsize = (nm0 * nm1 + nm0) * nelmt;
    }

    return wspsize;
}

/**
 * @brief Dynamic shared memory a 1D SumFac backward transform needs, in
 * TData values: none.
 *
 * The SumFac kernels hold their intermediates in the global-memory
 * workspace and read the basis tables straight from global memory, so
 * nothing is staged per thread block. The SumFacTOP overloads of the
 * same name (BwdTransDeviceSumFacTOPKernels.hpp) return the size of the
 * staging area that strategy does use.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    Operators::SumFac.
 * @tparam TSizeParameter1D  1D size parameter, runtime or templated.
 *
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 *
 * @return 0.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int BwdTransSharedMemorySize(
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Dynamic shared memory a 2D SumFac backward transform needs:
/// none, for the reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int BwdTransSharedMemorySize(
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    return 0;
}

/// @brief Dynamic shared memory a 3D SumFac backward transform needs:
/// none, for the reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int BwdTransSharedMemorySize(
    [[maybe_unused]] const TSizeParameter3D sizeParam3D)
{
    return 0;
}

/**
 * @brief Segment SumFac kernel: evaluate one element's expansion at its
 * quadrature points.
 *
 * There is nothing to factorise in one dimension, so the single basis
 * table is contracted straight into the output and no workspace is
 * used. @p in and @p out address the whole warp's interleaved data,
 * from which @p ilane selects this thread's element.
 *
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   ilane   Lane index of this thread's element within its warp.
 * @param   nm0     Modes of the expansion.
 * @param   nq0     Quadrature points of the expansion.
 * @param   basis0  Basis table: `basis0[p * nq0 + i]` is mode p at
 *                  point i.
 * @param   in      Warp's interleaved coefficients, nm0 * warpSize
 *                  values.
 * @param   out     Warp's interleaved physical values, nq0 * warpSize
 *                  values.
 */
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransSegSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p)
        {
            tmp += in[warpsize * p + ilane] * basis0[p * nq0 + i];
        }

        if constexpr (APPEND)
        {
            out[warpsize * i + ilane] += tmp;
        }
        else
        {
            out[warpsize * i + ilane] = tmp;
        }
    }
}

/**
 * @brief Quadrilateral SumFac kernel: tensor-product evaluation of one
 * element, one direction-0 point at a time.
 *
 * For each direction-0 point i the direction-0 contraction reduces the
 * coefficients to the nm1 values of @p wsp, one per direction-1 mode;
 * the direction-1 contraction then turns those into the physical values
 * of the whole line i. Coefficients are read as `in[q * nm0 + p]`, so
 * the direction-0 mode index runs fastest.
 *
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   ilane   Lane index of this thread's element within its warp.
 * @param   nm0,nm1 Modes per direction.
 * @param   nq0,nq1 Quadrature points per direction.
 * @param   basis0  Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1  Direction-1 basis table (nm1 rows of nq1).
 * @param   in      Warp's interleaved coefficients, nm0 * nm1 per
 *                  element.
 * @param   out     Warp's interleaved physical values, nq0 * nq1 per
 *                  element, direction-0 point running fastest.
 * @param   wsp     Warp's workspace slice, nm1 values per element.
 */
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransQuadSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        // direction 0
        for (unsigned int q = 0u, cnt_qp = 0u; q < nm1; ++q)
        {
            TData tmp = 0.0;
#pragma unroll
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
            {
                tmp += in[warpsize * cnt_qp + ilane] * basis0[p * nq0 + i];
            }
            wsp[warpsize * q + ilane] = tmp;
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            TData tmp = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                tmp += wsp[warpsize * q + ilane] * basis1[q * nq1 + j];
            }

            if constexpr (APPEND)
            {
                out[warpsize * (nq0 * j + i) + ilane] += tmp;
            }
            else
            {
                out[warpsize * (nq0 * j + i) + ilane] = tmp;
            }
        }
    }
}

/**
 * @brief Triangular SumFac kernel: evaluation of one element in the
 * collapsed coordinate system, one direction-1 point at a time.
 *
 * The collapsed direction goes first: for each direction-1 point j and
 * each p, the nm1 - p modes (p,q) -- consecutive in the triangular mode
 * ordering -- are contracted against @p basis1, whose rows are indexed
 * by that same combined mode. The resulting nm0 values in @p wsp are
 * then contracted against @p basis0 for every point of the line j.
 *
 * With @p isModified set, the term
 * `in[warpsize + ilane] * basis0[nq0 + i] * basis1[nq1 + j]` is added,
 * exactly as in the Serial/AVX BwdTransTriKernel.
 *
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1     Modes per direction.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table, one row of nq1 per
 *                      combined (p,q) mode.
 * @param   in          Warp's interleaved coefficients in the
 *                      triangular mode ordering.
 * @param   out         Warp's interleaved physical values, nq0 * nq1
 *                      per element.
 * @param   wsp         Warp's workspace slice, nm0 values per element.
 */
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransTriSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        // direction 1
        for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
        {
            TData tmp = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < (nm1 - p); ++q, ++mode_pq)
            {
                tmp +=
                    in[warpsize * mode_pq + ilane] * basis1[mode_pq * nq1 + j];
            }
            wsp[warpsize * p + ilane] = tmp;
        }

        // direction 0
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            TData tmp = 0.0;

            if (isModified)
            {
                tmp += in[warpsize + ilane] * basis0[nq0 + i] * basis1[nq1 + j];
            }

#pragma unroll
            for (unsigned int p = 0u; p < nm0; ++p)
            {
                tmp += wsp[warpsize * p + ilane] * basis0[p * nq0 + i];
            }

            if constexpr (APPEND)
            {
                out[warpsize * cnt_ji + ilane] += tmp;
            }
            else
            {
                out[warpsize * cnt_ji + ilane] = tmp;
            }
        }
    }
}

/**
 * @brief Hexahedral SumFac kernel: three-direction tensor-product
 * evaluation of one element, one direction-0 point at a time.
 *
 * For each direction-0 point i the direction-0 contraction fills the
 * nm1 * nm2 values of @p wsp0; then, for each direction-1 point j, the
 * direction-1 contraction fills the nm2 values of @p wsp1 and the
 * direction-2 contraction writes the nq2 physical values of the line
 * (i,j). Coefficients are read as `in[(r * nm1 + q) * nm0 + p]`, the
 * direction-0 mode index running fastest.
 *
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   basis0,basis1,basis2    Per-direction basis tables.
 * @param   in          Warp's interleaved coefficients,
 *                      nm0 * nm1 * nm2 per element.
 * @param   out         Warp's interleaved physical values,
 *                      nq0 * nq1 * nq2 per element, direction-0 point
 *                      running fastest.
 * @param   wsp0        Warp's first workspace slice, nm1 * nm2 values
 *                      per element.
 * @param   wsp1        Warp's second workspace slice, nm2 values per
 *                      element.
 */
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransHexSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        // direction 0
        for (unsigned int r = 0u, cnt_rqp = 0u, cnt_rq = 0u; r < nm2; ++r)
        {
            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                {
                    tmp += in[warpsize * cnt_rqp + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_rq + ilane] = tmp;
            }
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int r = 0u, cnt_rq = 0u; r < nm2; ++r)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                {
                    tmp +=
                        wsp0[warpsize * cnt_rq + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * r + ilane] = tmp;
            }

            // direction 2
            for (unsigned int k = 0u; k < nq2; ++k)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int r = 0u; r < nm2; ++r)
                {
                    tmp += wsp1[warpsize * r + ilane] * basis2[r * nq2 + k];
                }

                if constexpr (APPEND)
                {
                    out[warpsize * (k * nq1 * nq0 + j * nq0 + i) + ilane] +=
                        tmp;
                }
                else
                {
                    out[warpsize * (k * nq1 * nq0 + j * nq0 + i) + ilane] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Tetrahedral SumFac kernel: evaluation of one element in the
 * collapsed coordinate system, one direction-2 point at a time.
 *
 * The directions are contracted in the order 2, 1, 0. For each
 * direction-2 point k, the modes (p,q,r) -- r consecutive within a
 * (p,q) pair -- are contracted against @p basis2 into the @p fpq value
 * of their (p,q) pair; then, per direction-1 point j, the (p,q) pairs
 * are contracted against @p basis1 into the nm0 values of @p fp, which
 * the direction-0 contraction turns into the physical values of the
 * line (j,k).
 *
 * Two mode counters run through the first stage: `mode_pqr` walks the
 * coefficients, while `mode2` walks the rows of @p basis2, which
 * enumerate every (p,q,r) with q < nm2 - p. When nm2 > nm1 the table
 * therefore holds rows the coefficient array has no entries for, and
 * `mode2` is advanced past them at the end of each p.
 *
 * With @p isModified set, the top-vertex, bottom-vertex and
 * singular-edge terms marked in the code are added, exactly as in the
 * Serial/AVX BwdTransTetKernel.
 *
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table, one row of nq1 per
 *                      combined (p,q) mode.
 * @param   basis2      Direction-2 basis table, one row of nq2 per
 *                      combined (p,q,r) mode.
 * @param   in          Warp's interleaved coefficients in the
 *                      tetrahedral mode ordering.
 * @param   out         Warp's interleaved physical values,
 *                      nq0 * nq1 * nq2 per element.
 * @param   fpq         Warp's first workspace slice, one value per
 *                      (p,q) mode pair and element.
 * @param   fp          Warp's second workspace slice, nm0 values per
 *                      element.
 */
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransTetSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT fpq, TData *NEK_RESTRICT fp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        // direction 2
        for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u;
             p < nm0; ++p)
        {
            for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int r = 0u; r < nm2 - p - q;
                     ++r, ++mode2, ++mode_pqr)
                {
                    tmp += in[warpsize * mode_pqr + ilane] *
                           basis2[nq2 * mode2 + k];
                }
                fpq[warpsize * mode_pq + ilane] = tmp;
            }

            // increment mode in case nm2>nm1
#pragma unroll
            for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
            {
                mode2 += nm2 - p - q;
            }
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                {
                    tmp += fpq[warpsize * mode_pq + ilane] *
                           basis1[mode_pq * nq1 + j];
                }
                fp[warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                TData tmp = 0.0;

                if (isModified)
                {
                    // top vertex
                    tmp += basis0[i] * basis1[nq1 + j];
                    tmp += basis0[nq0 + i] * basis1[j];
                    tmp += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp *= basis2[nq2 + k] * in[warpsize + ilane];

                    // bottom vertex
                    TData tmp1 = basis2[k] * in[warpsize * nm2 + ilane];

                    // singular edge
#pragma unroll
                    for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                    {
                        tmp1 += basis2[(r + 1u) * nq2 + k] *
                                in[warpsize * (nm2 + r) + ilane];
                    }
                    tmp += basis1[nq1 + j] * basis0[nq0 + i] * tmp1;
                }

#pragma unroll
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += fp[warpsize * p + ilane] * basis0[p * nq0 + i];
                }

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_kji + ilane] += tmp;
                }
                else
                {
                    out[warpsize * cnt_kji + ilane] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Prismatic SumFac kernel: evaluation of one element in the
 * collapsed coordinate system, one direction-2 point at a time.
 *
 * As for the tetrahedron the directions are contracted in the order
 * 2, 1, 0, but only direction 2 is collapsed: for each p the
 * coefficients carry nm2 - p modes r for every one of the nm1 modes q,
 * and @p basis2 holds one row per combined (p,r) mode, whose block for
 * p starts at the running offset `mode_pr`. The direction-2 contraction
 * fills the nm0 * nm1 values of @p fpq, the direction-1 contraction the
 * nm0 values of @p fp, and the direction-0 contraction the physical
 * values of the line (j,k).
 *
 * With @p isModified set, the collapsed-vertex term summed over q in
 * the code is added, exactly as in the Serial/AVX BwdTransPrismKernel.
 *
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table (nm1 rows of nq1).
 * @param   basis2      Direction-2 basis table, one row of nq2 per
 *                      combined (p,r) mode.
 * @param   in          Warp's interleaved coefficients in the
 *                      prismatic mode ordering.
 * @param   out         Warp's interleaved physical values,
 *                      nq0 * nq1 * nq2 per element.
 * @param   fpq         Warp's first workspace slice, nm0 * nm1 values
 *                      per element.
 * @param   fp          Warp's second workspace slice, nm0 values per
 *                      element.
 */
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransPrismSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT fpq, TData *NEK_RESTRICT fp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        // direction 2
        for (unsigned int p = 0u, mode_pr = 0u, mode_pq = 0u, mode_pqr = 0u;
             p < nm0; ++p)
        {
            for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                {
                    tmp += in[warpsize * mode_pqr + ilane] *
                           basis2[(mode_pr + r) * nq2 + k];
                }
                fpq[warpsize * mode_pq + ilane] = tmp;
            }
            mode_pr += nm2 - p;
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                {
                    tmp +=
                        fpq[warpsize * mode_pq + ilane] * basis1[q * nq1 + j];
                }
                fp[warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                TData tmp = 0.0;

                if (isModified)
                {
#pragma unroll
                    for (unsigned int q = 0u; q < nm1; ++q)
                    {
                        tmp += basis1[q * nq1 + j] *
                               in[warpsize * (nm2 * q + 1u) + ilane];
                    }
                    tmp *= basis2[nq2 + k] * basis0[nq0 + i];
                }

#pragma unroll
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += fp[warpsize * p + ilane] * basis0[p * nq0 + i];
                }

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_kji + ilane] += tmp;
                }
                else
                {
                    out[warpsize * cnt_kji + ilane] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Pyramidal SumFac kernel: evaluation of one element in the
 * collapsed coordinate system, one direction-2 point at a time.
 *
 * Structured like the prismatic kernel, except that the direction-2
 * mode count of a (p,q) pair is nm2 - max(p,q) and @p basis2 holds one
 * row per combined (p,q,r) mode. As in the tetrahedral kernel the basis
 * table enumerates pairs the coefficient array has no entries for when
 * nm2 > nm1, and its counter `mode2` is advanced past them at the end
 * of each p.
 *
 * With @p isModified set, the top-vertex term marked in the code is
 * added, exactly as in the Serial/AVX BwdTransPyrKernel.
 *
 * @tparam APPEND  Accumulate onto @p out instead of overwriting it.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table (nm1 rows of nq1).
 * @param   basis2      Direction-2 basis table, one row of nq2 per
 *                      combined (p,q,r) mode.
 * @param   in          Warp's interleaved coefficients in the
 *                      pyramidal mode ordering.
 * @param   out         Warp's interleaved physical values,
 *                      nq0 * nq1 * nq2 per element.
 * @param   fpq         Warp's first workspace slice, nm0 * nm1 values
 *                      per element.
 * @param   fp          Warp's second workspace slice, nm0 values per
 *                      element.
 */
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void BwdTransPyrSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT fpq, TData *NEK_RESTRICT fp)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        // direction 2
        for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u;
             p < nm0; ++p)
        {
            for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int r = 0u; r < nm2 - std::max(p, q);
                     ++r, ++mode2, ++mode_pqr)
                {
                    tmp += in[warpsize * mode_pqr + ilane] *
                           basis2[mode2 * nq2 + k];
                }
                fpq[warpsize * mode_pq + ilane] = tmp;
            }

            // increment mode in case nm2>nm1
#pragma unroll
            for (unsigned int q = nm1; q < nm2; ++q)
            {
                mode2 += nm2 - q;
            }
        }

        // direction 1
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                {
                    tmp +=
                        fpq[warpsize * mode_pq + ilane] * basis1[q * nq1 + j];
                }
                fp[warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                TData tmp = 0.0;

                if (isModified)
                {
                    // top vertex
                    tmp += basis0[i] * basis1[nq1 + j];
                    tmp += basis0[nq0 + i] * basis1[j];
                    tmp += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp *= basis2[nq2 + k] * in[warpsize + ilane];
                }

#pragma unroll
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += fp[warpsize * p + ilane] * basis0[p * nq0 + i];
                }

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_kji + ilane] += tmp;
                }
                else
                {
                    out[warpsize * cnt_kji + ilane] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Device entry point of the 1D backward transform under SumFac:
 * unpack the size parameter, then walk the block's elements, one per
 * thread, applying the segment kernel to each.
 *
 * Launched through DEVICE_2DGRID_KERNEL_LAUNCHER, which appends the last
 * two arguments and supplies grid, block, shared-memory size and stream
 * (see BwdTransDeviceSumFac.hpp). The defaulted Enable template
 * parameter restricts this definition to the SumFac tag, so that the
 * SumFacTOP file can define an overload of the same name for its own
 * tag; the requirement of a 1D size parameter is enforced by the
 * static_assert in the body instead. The launch bound comes from
 * GetMaxThreadPerBlock, which for SumFac is the warp size the launcher
 * uses as block size. FETCH_SHARED_MEMORY binds @p shmemptr to the
 * block's dynamic shared memory on the CUDA/HIP back-ends, and is a
 * no-op where the launcher macro passes the pointer in; SumFac requests
 * none either way, so @p shmemptr goes unused once unpacked.
 *
 * The thread then starts at its index in the grid's first dimension and
 * advances by that dimension's thread count, so the block's elements --
 * padding included, their results being discarded -- are covered however
 * many threads were launched. Each visit derives the element's warp and
 * lane and offsets @p in and @p out to that warp's interleaved group.
 *
 * @tparam SHAPE_TYPE       Seg.
 * @tparam Implementation   Operators::SumFac.
 * @tparam APPEND           Accumulate onto @p out instead of
 *                          overwriting.
 * @tparam TSizeParameter1D 1D size parameter, in the runtime or the
 *                          compile-time form; with the latter the sizes
 *                          below are compile-time constants.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 *
 * @param   sizeParam1D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A; the
 *                      segment kernel needs no correction, so unused.
 * @param   basis0      Basis table in device memory.
 * @param   nodToMod    Nodal-to-modal matrix; no 1D nodal shape, so
 *                      unused.
 * @param   in          Block's coefficients for every component.
 * @param   out         Block's physical values for every component.
 * @param   wsp         Block's workspace; the segment kernel keeps no
 *                      intermediates, so unused.
 * @param   shmemptr    Shared-memory base, appended by the launcher;
 *                      unused by SumFac.
 * @param   threadBlock Thread-block handle, appended by the launcher.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *inptr = in + nm0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nq0 * (nelmt * c + warpsize * iwarp);
        BwdTransSegSumFacKernel<APPEND>(ilane, nm0, nq0, basis0, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D backward transform under SumFac:
 * unpack the size parameter, then walk the block's elements, one per
 * thread, applying the shape kernel selected by @p SHAPE_TYPE to each.
 *
 * Constrained, launched and bounded as the 1D entry point.
 *
 * Strides over the elements as the 1D entry point does, and
 * additionally slices @p wsp: within a stage's region, component `c`'s
 * values for warp `iwarp` start at
 * `stage size * (nelmt * c + warpSize * iwarp)`, and the regions of
 * successive stages are `stage size * nelmt * ncomp` apart, matching
 * BwdTransWorkSpaceSize times ncomp. For NodalTri the element's
 * coefficients are converted to the modified modal basis by MatVecKernel
 * into the first region, and the triangular kernel then runs on the
 * converted values with the second region as its workspace.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam Implementation   Operators::SumFac.
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
 * @param   wsp         Block's workspace, sized by
 *                      BwdTransWorkSpaceSize times ncomp.
 * @param   shmemptr    Shared-memory base, appended by the launcher;
 *                      unused by SumFac.
 * @param   threadBlock Thread-block handle, appended by the launcher.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    BwdTransKernelLauncher(const TSizeParameter2D sizeParam2D,
                           const size_t nelmt, const bool isModified,
                           const TData *NEK_RESTRICT basis0,
                           const TData *NEK_RESTRICT basis1,
                           const TData *NEK_RESTRICT nodToMod,
                           const TData *NEK_RESTRICT in,
                           TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp,
                           [[maybe_unused]] unsigned char *shmemptr,
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wspptr = wsp + nm1 * (nelmt * c + warpsize * iwarp);
            BwdTransQuadSumFacKernel<APPEND>(ilane, nm0, nm1, nq0, nq1, basis0,
                                             basis1, inptr, outptr, wspptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wspptr = wsp + nm0 * (nelmt * c + warpsize * iwarp);
            BwdTransTriSumFacKernel<APPEND>(ilane, nm0, nm1, nq0, nq1,
                                            isModified, basis0, basis1, inptr,
                                            outptr, wspptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *in1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wspptr = wsp + nmTot * nelmt * ncomp +
                            nm0 * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransTriSumFacKernel<APPEND>(ilane, nm0, nm1, nq0, nq1,
                                            isModified, basis0, basis1, in1ptr,
                                            outptr, wspptr);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D backward transform under SumFac:
 * unpack the size parameter, then walk the block's elements, one per
 * thread, applying the shape kernel selected by @p SHAPE_TYPE to each.
 *
 * Constrained, launched and bounded as the 1D entry point. The two
 * mode-index tables are part of the shared signature but are used only
 * by the SumFacTOP tetrahedral kernel; the block implementation passes
 * null pointers here.
 *
 * As the 2D entry point, with two stage regions in @p wsp instead of one
 * and, for NodalTet and NodalPrism, a leading region of nmTot values per
 * element holding the coefficients MatVecKernel has converted to the
 * modified modal basis.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism or
 *                          Pyr.
 * @tparam Implementation   Operators::SumFac.
 * @tparam APPEND           Accumulate onto @p out instead of
 *                          overwriting.
 * @tparam TSizeParameter3D 3D size parameter, runtime or compile-time
 *                          form.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 *
 * @param   sizeParam3D Mode and quadrature-point counts of an element.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0,index1   Mode-index tables; unused here.
 * @param   basis0,basis1,basis2    Per-direction basis tables.
 * @param   nodToMod    Nodal-to-modal matrix; nodal shapes only.
 * @param   in          Block's coefficients for every component.
 * @param   out         Block's physical values for every component.
 * @param   wsp         Block's workspace, sized by
 *                      BwdTransWorkSpaceSize times ncomp.
 * @param   shmemptr    Shared-memory base, appended by the launcher;
 *                      unused by SumFac.
 * @param   threadBlock Thread-block handle, appended by the launcher.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    BwdTransKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified, [[maybe_unused]] const unsigned int *index0,
        [[maybe_unused]] const unsigned int *index1,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + nm1 * nm2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nm1 * nm2) * nelmt * ncomp +
                          nm2 * (nelmt * c + warpsize * iwarp);
            BwdTransHexSumFacKernel<APPEND>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                            basis0, basis1, basis2, inptr,
                                            outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

            TData *wsp0 = wsp + nm01 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nm01 * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            BwdTransTetSumFacKernel<APPEND>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                            isModified, basis0, basis1, basis2,
                                            inptr, outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

            TData *in1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0   = wsp + nmTot * nelmt * ncomp +
                          nm01 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nm01 + nmTot) * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransTetSumFacKernel<APPEND>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                            isModified, basis0, basis1, basis2,
                                            in1ptr, outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 = wsp + nm0 * nm1 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nm0 * nm1 * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            BwdTransPrismSumFacKernel<APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, inptr, outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *in1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0   = wsp + nmTot * nelmt * ncomp +
                          nm0 * nm1 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nm0 * nm1 + nmTot) * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransPrismSumFacKernel<APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, in1ptr, outptr, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 = wsp + nm0 * nm1 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nm0 * nm1 * nelmt * ncomp +
                          nm0 * (nelmt * c + warpsize * iwarp);
            BwdTransPyrSumFacKernel<APPEND>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                            isModified, basis0, basis1, basis2,
                                            inptr, outptr, wsp0, wsp1);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
