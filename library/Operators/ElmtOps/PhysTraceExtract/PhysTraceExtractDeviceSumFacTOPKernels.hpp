///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractDeviceSumFacTOPKernels.hpp
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
// Description: PhysTraceExtract SumFacTOP device kernels: one thread
// block per element on width-1 element-contiguous storage, threads
// striding the trace points, following the volume-op SumFacTOP pattern.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysTraceExtractDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernels of the trace extraction: one thread
 * block per element, threads striding the trace points.
 *
 * @details
 * The dedicated thread-over-points counterpart of
 * PhysTraceExtractDeviceSumFacKernels.hpp. Both compute the same
 * operator; they differ in how an element is mapped onto threads, and
 * so in how its data is addressed:
 *
 * - the SumFac kernels put one element on one warp *lane*, so entry
 *   @em n of element \f$e = (i_{warp}, i_{lane})\f$ lives at
 *   `buf[warpsize * n + ilane]` and a lane walks its element's points
 *   serially;
 * - these kernels put one element on one thread *block*, so an
 *   element's entries are contiguous -- entry @em n at `buf[n]` from an
 *   element base of `numData * e` -- and the block's threads share the
 *   point loops, striding them by the block size with a
 *   localBarrier() between dependent stages.
 *
 * That is why the host class pairs them with an implementation
 * interleave width of one (PhysTraceExtractDeviceSumFac.hpp) and
 * with the SumFacTOP launch geometry of GetDeviceBlockSize() and
 * GetDeviceGridSize(), and why the intermediate face-shaped blocks live
 * in dynamic shared memory rather than in the global workspace the
 * SumFac path partitions per warp.
 *
 * Layout of an element's packed trace output is the shared one: traces
 * ordered by normal direction, the N0 pair first, then N1, then N2,
 * face-major within a pair.
 *
 * The trace tables, the TraceExtractDispatch record and its two
 * producers GetTraceEdgeDispatch()/GetTraceFaceDispatch(), and the
 * per-element bound PhysTraceExtractBlockSize() are shared with the
 * SumFac kernels and included from there rather than duplicated: they
 * describe the operator's trace numbering and the size of an element's
 * intermediate block, which do not depend on the thread mapping.
 *
 * @see PhysTraceExtractDeviceSumFac.hpp for the host dispatch that
 * launches these under the SumFacTOP tag.
 * @see PhysTraceExtractDeviceSumFacKernels.hpp for the warp-per-lane
 * kernels and for the shared trace tables.
 * @see IProductWRTPhysTraceDeviceSumFacTOPKernels.hpp for the adjoint
 * operator's TOP kernels.
 */

#pragma once

#include "Operators/ElmtOps/PhysTraceExtract/PhysTraceExtractDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

// Helper functions

/// @brief Workspace of a whole block, per component: none, the
/// intermediates living in the dynamic shared memory
/// PhysTraceExtractSharedMemorySize() sizes.
template <typename Implementation, typename TTraceSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t PhysTraceExtractWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Workspace of a whole block, per component: none, the
/// intermediates living in the dynamic shared memory
/// PhysTraceExtractSharedMemorySize() sizes.
template <typename Implementation, typename TTraceSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t PhysTraceExtractWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter2D sizeParam2D)
{
    return 0;
}

/// @brief Workspace of a whole block, per component: none, the
/// intermediates living in the dynamic shared memory
/// PhysTraceExtractSharedMemorySize() sizes.
template <typename Implementation, typename TTraceSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t PhysTraceExtractWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter3D sizeParam3D)
{
    return 0;
}

/// @brief Dynamic shared memory one block needs, in elements: none, a
/// segment's traces being its end vertices with nothing to interpolate
/// tangentially.
template <typename Implementation, typename TTraceSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysTraceExtractSharedMemorySize(
    [[maybe_unused]] const TTraceSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Dynamic shared memory one block needs, in elements: the edge
/// block the normal stage writes, one PhysTraceExtractBlockSize(). One
/// element is resident per block at a time, so the per-element bound of
/// the global workspace is what a block needs.
template <typename Implementation, typename TTraceSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysTraceExtractSharedMemorySize(
    const TTraceSizeParameter2D sizeParam2D)
{
    return PhysTraceExtractBlockSize(sizeParam2D);
}

/// @brief Dynamic shared memory one block needs, in elements: three
/// PhysTraceExtractBlockSize(), two for the face block the normal stage
/// writes and one for the tangential scratch, the split the kernels
/// re-derive from the same helper.
template <typename Implementation, typename TTraceSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysTraceExtractSharedMemorySize(
    const TTraceSizeParameter3D sizeParam3D)
{
    return 3u * PhysTraceExtractBlockSize(sizeParam3D);
}

// --- 2D ---

/**
 * @brief One two-dimensional element, one normal direction: extract the
 * edges of that direction and resample them onto their trace points.
 *
 * The one-block-per-element twin of BwdTransQuadSumFacKernelTrace. Both
 * stages live in a single kernel, with the normal direction fixed by
 * @p NormalDir and a barrier between them. Stage one takes the volume
 * field to the [@p edg, @p nedg) window of edges of that direction and
 * writes `nedge = nedg - edg` values per in-edge point into @p wsp with
 * the edges interleaved, the threads striding the flattened
 * (in-edge point, edge) index. Stage two contracts the in-edge direction
 * against @p tbasis onto @p npts trace points and writes each edge at
 * its own offset in @p out.
 *
 * Under @p endPtsCollocated stage one is a plane selection, the lower
 * edge of the direction being the first volume plane and the upper one
 * the last. Under @p isCollocated stage two is a copy. The @c ntbasis
 * row stride is a compile-time constant taken from the shape, two in
 * direction 0 for both shapes and two in direction 1 except for a
 * triangle, so it is the direction's trace count on the bulk and the
 * per-trace route alike.
 *
 * The name is inherited: nothing from the shared BwdTrans kernels is
 * called here.
 *
 * @tparam SHAPE_TYPE    Quad, Tri or NodalTri.
 * @tparam NormalDir     Normal direction of the edges handled: 0 or 1.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   nm0,nm1     Volume quadrature points per direction.
 * @param   nq0,nq1     Trace quadrature counts per normal direction.
 *                      Unused: the count in play arrives as @p npts.
 * @param   edg         Position of the first edge within its direction.
 * @param   nedg        One past the position of the last edge.
 * @param   npts        Trace quadrature points of these edges.
 * @param   ntbasis     eInterp table \f$h_p(\pm 1)\f$ of the normal
 *                      direction.
 * @param   tbasis      eInterp table to the trace points of the in-edge
 *                      direction, indexed `tbasis[q * npts + i]`.
 * @param   in          Volume field of this block's element.
 * @param   out         Trace values, already offset to this direction's
 *                      block.
 * @param   wsp         Scratch between the two stages in shared memory,
 *                      `2 * max(nm0,nm1)` entries.
 * @param   isCollocated      Trace points of this direction coincide
 *                      with the volume points of the in-edge direction.
 * @param   endPtsCollocated  Volume rule of the normal direction
 *                      contains the domain endpoints.
 * @param   threadBlock Thread block cooperating on this element.
 * @param   offset_edge0,offset_edge1     Where each of the two edges
 *                      covered starts within @p out. The per-trace route
 *                      leaves both at zero, its window being one edge
 *                      wide.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, unsigned int NormalDir,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransQuadSumFacTOPKernelTrace(
    const unsigned int nm0, const unsigned int nm1,
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1, const unsigned int edg,
    const unsigned int nedg, const unsigned int npts,
    const TData *NEK_RESTRICT ntbasis, const TData *NEK_RESTRICT tbasis,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, bool isCollocated, bool endPtsCollocated,
    const TthreadBlock &threadBlock, unsigned int offset_edge0 = 0,
    unsigned int offset_edge1 = 0)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    const unsigned int nedge  = nedg - edg;

    if constexpr (NormalDir == 0)
    {
        constexpr unsigned int ntrace = 2u;

        // 1. Normal direction (dir 0) to wsp
        for (unsigned int idx = idx0; idx < nedge * nm1; idx += stride)
        {
            const unsigned int localE = idx % nedge;
            const unsigned int q      = idx / nedge;
            const unsigned int e      = edg + localE;

            if (endPtsCollocated)
            {
                unsigned int offset_in;
                if constexpr (SHAPE_TYPE == LibUtilities::Tri)
                {
                    offset_in = (e == 0) ? 0 : (nm0 - 1);
                }
                else
                {
                    offset_in = e * (nm0 - 1);
                }
                wsp[idx] = in[q * nm0 + offset_in];
            }
            else
            {
                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += in[q * nm0 + p] * ntbasis[p * ntrace + e];
                }
                wsp[idx] = tmp;
            }
        }

        localBarrier(threadBlock);

        // 2. Trace direction (dir 1) to out
        for (unsigned int idx = idx0; idx < nedge * npts; idx += stride)
        {
            const unsigned int localE = idx % nedge;
            const unsigned int j      = idx / nedge;
            const unsigned int offset_out =
                (localE == 0) ? offset_edge0 : offset_edge1;

            if (isCollocated)
            {
                out[offset_out + j] = wsp[idx];
            }
            else
            {
                TData tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    tmp += wsp[q * nedge + localE] * tbasis[q * npts + j];
                }
                out[offset_out + j] = tmp;
            }
        }
    }
    else // NormalDir == 1
    {
        // From the shape table, not from a Tri/Quad test: the bulk route
        // passes the nodal enumerators straight through, and a NodalTri has
        // one trace in this direction just as a Tri does.
        constexpr unsigned int ntrace =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

        // 1. Normal direction (dir 1) to wsp
        for (unsigned int idx = idx0; idx < nedge * nm0; idx += stride)
        {
            const unsigned int localE = idx % nedge;
            const unsigned int p      = idx / nedge;
            const unsigned int e      = edg + localE;

            if (endPtsCollocated)
            {
                const unsigned int offset_in = e * nm0 * (nm1 - 1);
                wsp[idx]                     = in[offset_in + p];
            }
            else
            {
                TData tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    tmp += in[q * nm0 + p] * ntbasis[q * ntrace + e];
                }
                wsp[idx] = tmp;
            }
        }

        localBarrier(threadBlock);

        // 2. Trace direction (dir 0) to out
        for (unsigned int idx = idx0; idx < nedge * npts; idx += stride)
        {
            const unsigned int localE = idx % nedge;
            const unsigned int i      = idx / nedge;
            const unsigned int offset_out =
                (localE == 0) ? offset_edge0 : offset_edge1;

            if (isCollocated)
            {
                out[offset_out + i] = wsp[idx];
            }
            else
            {
                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += wsp[p * nedge + localE] * tbasis[p * npts + i];
                }
                out[offset_out + i] = tmp;
            }
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief All edges of one two-dimensional element: both directions, in
 * packed trace order.
 *
 * The bulk worker, and the one-block-per-element twin of
 * PhysTraceExtract2DSumFacKernelCore. The direction-0 pair goes into the
 * first `2 * nq0` entries of @p out, lower edge then upper edge, and the
 * direction-1 traces follow: a pair for a quadrilateral, at offsets 0
 * and @p nq1 past that, and a single edge for a triangle, whose
 * direction-1 upper end is the collapsed vertex. Both shapes run the
 * same leaf kernel template, its @c Tri instantiation differing only in
 * the direction-0 offset branch, which agrees numerically, and in the
 * direction-1 @c ntrace constant; the hypotenuse is just the
 * \f$\eta_0 = +1\f$ line of the collapsed square.
 *
 * @tparam SHAPE_TYPE    Quad, Tri or NodalTri.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   nm0,nm1     Volume quadrature points per direction.
 * @param   nq0,nq1     Trace quadrature points of the direction-0 and
 *                      direction-1 edges.
 * @param   ntbasis0,ntbasis1   eInterp tables \f$h_p(\pm 1)\f$ per
 *                      direction.
 * @param   basis0,basis1       eInterp tables to the trace points, one
 *                      per normal direction; the @c tbasis of the rest
 *                      of the file.
 * @param   wsp         Scratch between the two stages in shared memory,
 *                      `2 * max(nm0,nm1)` entries.
 * @param   in          Volume field of this block's element.
 * @param   out         All of this element's trace values.
 * @param   isCollocated0,isCollocated1     Trace points of that normal
 *                      direction coincide with the volume points of the
 *                      in-edge direction.
 * @param   endPtsCollocated0,endPtsCollocated1   Volume rule of that
 *                      direction contains the domain endpoints.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysTraceExtract2DSumFacTOPKernelCore(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const TData *ntbasis0, const TData *ntbasis1,
    const TData *basis0, const TData *basis1, TData *wsp, const TData *in,
    TData *out, bool isCollocated0, bool isCollocated1, bool endPtsCollocated0,
    bool endPtsCollocated1, const TthreadBlock &threadBlock)
{
    const unsigned int offset_Trace0_Edge0 = 0;
    const unsigned int offset_Trace0_Edge1 = nq0;
    const unsigned int offset_Trace1       = 2 * nq0;

    // Dir 0 normal set
    BwdTransQuadSumFacTOPKernelTrace<SHAPE_TYPE, 0>(
        nm0, nm1, nq0, nq1, 0, 2, nq0, ntbasis0, basis0, in, out, wsp,
        isCollocated0, endPtsCollocated0, threadBlock, offset_Trace0_Edge0,
        offset_Trace0_Edge1);

    // Dir 1 normal set. The trace count comes from the shape table rather
    // than from a Tri/Quad test, because the bulk route passes the nodal
    // enumerators straight through: a NodalTri has one trace here, and
    // testing only for Tri gave it two.
    constexpr unsigned int ntrace1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

    BwdTransQuadSumFacTOPKernelTrace<SHAPE_TYPE, 1>(
        nm0, nm1, nq0, nq1, 0, ntrace1, nq1, ntbasis1, basis1, in,
        out + offset_Trace1, wsp, isCollocated1, endPtsCollocated1, threadBlock,
        0, (ntrace1 == 1u) ? 0u : nq1);
}

// --- 3D ---

/**
 * @brief Leaf kernel, tangential stage in three dimensions: resample a
 * face, or a pair of faces, from the volume grid onto the face's own
 * quadrature points.
 *
 * The one-block-per-element twin of PhysInterpFaceKernelTrace. Takes
 * @p nface faces laid out `(f, q1, q0)`, the first tangential index
 * fastest, and applies the two interpolation tables in turn,
 * \f$\text{out}_{fji} = \sum_q t^1_{qj} \sum_p t^0_{pi}\,
 * \text{in}_{fqp}\f$. Three arms: with @p isCollocated1 only direction 0
 * is interpolated, straight into @p out; with @p isCollocated0 only
 * direction 1 is, again straight into @p out; otherwise direction 0 goes
 * into @p wsp and direction 1 from there into @p out, with a barrier
 * between the two. The both-collocated case never arrives, the glue
 * level having skipped this kernel entirely for it.
 *
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   nface       Faces covered, already the width of the window.
 * @param   nqfrom0,nqfrom1     Volume quadrature counts of the two
 *                      tangential directions, the source grid.
 * @param   nqto0,nqto1 Trace quadrature counts of the face, the target
 *                      grid.
 * @param   tbasis0,tbasis1     eInterp tables to the trace points of the
 *                      two tangential directions, indexed
 *                      `tbasis0[p * nqto0 + i]`.
 * @param   wsp         Scratch between the two contractions in shared
 *                      memory, `nface * nqfrom1 * nqto0` entries.
 *                      Untouched by the two collocated arms.
 * @param   in          Face values on the volume face grid.
 * @param   out         Face values on the trace grid.
 * @param   isCollocated0,isCollocated1   That tangential direction's
 *                      trace points coincide with the volume points it
 *                      comes from, so its table is the identity.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysInterpFaceTOPKernelTrace(
    const unsigned int nface, const unsigned int nqfrom0,
    const unsigned int nqfrom1, const unsigned int nqto0,
    const unsigned int nqto1, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, TData *NEK_RESTRICT wsp,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, bool isCollocated0,
    bool isCollocated1, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if (isCollocated1)
    {
        // Interpolate dir 0 directly to output.
        for (unsigned int idx = idx0; idx < nface * nqfrom1 * nqto0;
             idx += stride)
        {
            const unsigned int i  = idx % nqto0;
            const unsigned int q1 = (idx / nqto0) % nqfrom1;
            const unsigned int f  = idx / (nqto0 * nqfrom1);

            TData tmp = in[(f * nqfrom1 + q1) * nqfrom0] * tbasis0[i];
            for (unsigned int p = 1u; p < nqfrom0; ++p)
            {
                tmp += in[(f * nqfrom1 + q1) * nqfrom0 + p] *
                       tbasis0[p * nqto0 + i];
            }
            out[idx] = tmp;
        }
    }
    else if (isCollocated0)
    {
        // Interpolate dir 1 only.
        for (unsigned int idx = idx0; idx < nface * nqto0 * nqto1;
             idx += stride)
        {
            const unsigned int j = idx % nqto1;
            const unsigned int i = (idx / nqto1) % nqto0;
            const unsigned int f = idx / (nqto1 * nqto0);

            TData tmp = in[(f * nqfrom1) * nqto0 + i] * tbasis1[j];
            for (unsigned int q = 1u; q < nqfrom1; ++q)
            {
                tmp +=
                    in[(f * nqfrom1 + q) * nqto0 + i] * tbasis1[q * nqto1 + j];
            }
            out[(f * nqto1 + j) * nqto0 + i] = tmp;
        }
    }
    else
    {
        // Interpolate dir 0 to workspace.
        for (unsigned int idx = idx0; idx < nface * nqfrom1 * nqto0;
             idx += stride)
        {
            const unsigned int i  = idx % nqto0;
            const unsigned int q1 = (idx / nqto0) % nqfrom1;
            const unsigned int f  = idx / (nqto0 * nqfrom1);

            TData tmp = in[(f * nqfrom1 + q1) * nqfrom0] * tbasis0[i];
            for (unsigned int p = 1u; p < nqfrom0; ++p)
            {
                tmp += in[(f * nqfrom1 + q1) * nqfrom0 + p] *
                       tbasis0[p * nqto0 + i];
            }
            wsp[idx] = tmp;
        }

        localBarrier(threadBlock);

        // Interpolate dir 1 to output.
        for (unsigned int idx = idx0; idx < nface * nqto0 * nqto1;
             idx += stride)
        {
            const unsigned int j = idx % nqto1;
            const unsigned int i = (idx / nqto1) % nqto0;
            const unsigned int f = idx / (nqto1 * nqto0);

            TData tmp = wsp[(f * nqfrom1) * nqto0 + i] * tbasis1[j];
            for (unsigned int q = 1u; q < nqfrom1; ++q)
            {
                tmp +=
                    wsp[(f * nqfrom1 + q) * nqto0 + i] * tbasis1[q * nqto1 + j];
            }
            out[(f * nqto1 + j) * nqto0 + i] = tmp;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Leaf kernel, normal stage in three dimensions: take the volume
 * field to the direction-0 faces of the [@p fac, @p nface) window.
 *
 * The one-block-per-element twin of PhysExtractEndFacesN0KernelTrace3D,
 * the threads striding the flattened output index. Produces one value
 * per (face, direction-2 point, direction-1 point), direction 1 fastest, either
 * by selecting the boundary plane \f$i = 0\f$ or \f$i = nq_0 - 1\f$ when @p
 * END_PTS_COLLOCATED0, or by contracting direction 0 against \f$h_p(\pm 1)\f$
 * in the general arm. The result is the face grid sampled at the volume
 * quadrature points, which the caller either keeps, the tangential directions
 * being collocated, or hands to PhysInterpFaceTOPKernelTrace.
 *
 * @tparam END_PTS_COLLOCATED0    Direction-0 volume rule contains the
 *                      domain endpoints.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   fac         Position of the first face within direction 0.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of the @c ntbasis table: the number of
 *                      traces the direction has, which is not @p nface
 *                      when the call covers part of the direction.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   in          Volume field of this block's element.
 * @param   out         Face values on the volume face grid,
 *                      `(nface - fac) * nq2 * nq1` entries.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED0 = false, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysExtractEndFacesN0TOPKernelTrace3D(
    const unsigned int fac, const unsigned int nface,
    const unsigned int tstride, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT ntbasis0,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    const unsigned int nq0nq1 = nq0 * nq1;
    const unsigned int nfac   = nface - fac;

    for (unsigned int idx = idx0; idx < nfac * nq2 * nq1; idx += stride)
    {
        const unsigned int j  = idx % nq1;
        const unsigned int k  = (idx / nq1) % nq2;
        const unsigned int lf = idx / (nq1 * nq2);
        const unsigned int f  = fac + lf;

        if constexpr (END_PTS_COLLOCATED0)
        {
            out[idx] = in[k * nq0nq1 + f * (nq0 - 1) + j * nq0];
        }
        else
        {
            const unsigned int offset_fkp = k * nq0nq1 + j * nq0;
            TData tmp                     = in[offset_fkp] * ntbasis0[f];
            for (unsigned int p = 1u; p < nq0; ++p)
            {
                tmp += in[offset_fkp + p] * ntbasis0[p * tstride + f];
            }
            out[idx] = tmp;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Leaf kernel, normal stage in three dimensions: take the volume
 * field to the direction-1 faces of the [@p fac, @p nface) window.
 *
 * The one-block-per-element twin of PhysExtractEndFacesN1KernelTrace3D,
 * the threads striding the flattened output index. Produces one value
 * per (face, direction-2 point, direction-0 point), direction 0 fastest, either
 * by selecting the boundary plane \f$j = 0\f$ or \f$j = nq_1 - 1\f$ when @p
 * END_PTS_COLLOCATED1, or by contracting direction 1 against \f$h_q(\pm 1)\f$
 * in the general arm. The result is the face grid sampled at the volume
 * quadrature points, which the caller either keeps, the tangential directions
 * being collocated, or hands to PhysInterpFaceTOPKernelTrace.
 *
 * @tparam END_PTS_COLLOCATED1    Direction-1 volume rule contains the
 *                      domain endpoints.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   fac         Position of the first face within direction 1.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of the @c ntbasis table: the number of
 *                      traces the direction has, which is not @p nface
 *                      when the call covers part of the direction.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis1    eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   in          Volume field of this block's element.
 * @param   out         Face values on the volume face grid,
 *                      `(nface - fac) * nq2 * nq0` entries.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED1 = false, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysExtractEndFacesN1TOPKernelTrace3D(
    const unsigned int fac, const unsigned int nface,
    const unsigned int tstride, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT ntbasis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    const unsigned int nq0nq1 = nq0 * nq1;
    const unsigned int nfac   = nface - fac;

    for (unsigned int idx = idx0; idx < nfac * nq2 * nq0; idx += stride)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int k  = (idx / nq0) % nq2;
        const unsigned int lf = idx / (nq0 * nq2);
        const unsigned int f  = fac + lf;

        if constexpr (END_PTS_COLLOCATED1)
        {
            out[idx] = in[k * nq0nq1 + f * nq0 * (nq1 - 1) + i];
        }
        else
        {
            const unsigned int offset_k = k * nq0nq1;
            TData tmp                   = in[offset_k + i] * ntbasis1[f];
            for (unsigned int q = 1u; q < nq1; ++q)
            {
                tmp += in[offset_k + q * nq0 + i] * ntbasis1[q * tstride + f];
            }
            out[idx] = tmp;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Leaf kernel, normal stage in three dimensions: take the volume
 * field to the direction-2 faces of the [@p fac, @p nface) window.
 *
 * The one-block-per-element twin of PhysExtractEndFacesN2KernelTrace3D,
 * the threads striding the flattened output index. Produces one value
 * per (face, direction-1 point, direction-0 point), direction 0 fastest, either
 * by selecting the boundary plane \f$k = 0\f$ or \f$k = nq_2 - 1\f$ when @p
 * END_PTS_COLLOCATED2, or by contracting direction 2 against \f$h_r(\pm 1)\f$
 * in the general arm. The result is the face grid sampled at the volume
 * quadrature points, which the caller either keeps, the tangential directions
 * being collocated, or hands to PhysInterpFaceTOPKernelTrace.
 *
 * @tparam END_PTS_COLLOCATED2    Direction-2 volume rule contains the
 *                      domain endpoints.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   fac         Position of the first face within direction 2.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of the @c ntbasis table: the number of
 *                      traces the direction has, which is not @p nface
 *                      when the call covers part of the direction.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis2    eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   in          Volume field of this block's element.
 * @param   out         Face values on the volume face grid,
 *                      `(nface - fac) * nq1 * nq0` entries.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED2 = false, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysExtractEndFacesN2TOPKernelTrace3D(
    const unsigned int fac, const unsigned int nface,
    const unsigned int tstride, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT ntbasis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    const unsigned int nq01   = nq0 * nq1;
    const unsigned int nfac   = nface - fac;

    for (unsigned int idx = idx0; idx < nfac * nq01; idx += stride)
    {
        const unsigned int cnt_ji = idx % nq01;
        const unsigned int lk     = idx / nq01;
        const unsigned int k      = fac + lk;

        if constexpr (END_PTS_COLLOCATED2)
        {
            out[idx] = in[cnt_ji + k * nq01 * (nq2 - 1)];
        }
        else
        {
            TData tmp = in[cnt_ji] * ntbasis2[k];
            for (unsigned int r = 1u; r < nq2; ++r)
            {
                tmp += in[cnt_ji + r * nq01] * ntbasis2[r * tstride + k];
            }
            out[idx] = tmp;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Glue for the direction-0 normal faces: normal stage, then the
 * tangential stage if it is needed.
 *
 * The one-block-per-element twin of PhysExtractFaceN0KernelTrace3D. When
 * both tangential directions are collocated the trace grid @em is the
 * volume face grid, and PhysExtractEndFacesN0TOPKernelTrace3D writes
 * straight into @p out. Otherwise it writes into @p wsp1 and
 * PhysInterpFaceTOPKernelTrace resamples (nq1,nq2) onto `(nq00,nq01)`, using
 * @p wsp2 when neither direction is collocated.
 *
 * @tparam END_PTS_COLLOCATED0    Direction-0 volume rule contains the
 *                      domain endpoints.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   fac         Position of the first face within direction 0.
 * @param   nfac        One past the position of the last face.
 * @param   tstride     Row stride of the @c ntbasis table: the number of
 *                      traces the direction has.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   nq00,nq01   Trace quadrature points of the two tangential
 *                      directions of a direction-0 face.
 * @param   tbasis00,tbasis01   eInterp tables to those trace points.
 * @param   wsp1        Face values between the two stages, in shared
 *                      memory.
 * @param   wsp2        Scratch of the fully general tangential stage.
 * @param   in          Volume field of this block's element.
 * @param   out         Face values on the trace grid.
 * @param   isCollocated00,isCollocated01   That tangential direction's
 *                      trace points coincide with the volume points it
 *                      comes from.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED0 = false, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysExtractFaceN0TOPKernelTrace3D(
    const unsigned int fac, const unsigned int nfac, const unsigned int tstride,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT ntbasis0, const unsigned int nq00,
    const unsigned int nq01, const TData *NEK_RESTRICT tbasis00,
    const TData *NEK_RESTRICT tbasis01, TData *NEK_RESTRICT wsp1,
    TData *NEK_RESTRICT wsp2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, bool isCollocated00, bool isCollocated01,
    const TthreadBlock &threadBlock)
{
    if (isCollocated00 && isCollocated01)
    {
        PhysExtractEndFacesN0TOPKernelTrace3D<END_PTS_COLLOCATED0>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis0, in, out, threadBlock);
    }
    else
    {
        PhysExtractEndFacesN0TOPKernelTrace3D<END_PTS_COLLOCATED0>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis0, in, wsp1, threadBlock);

        PhysInterpFaceTOPKernelTrace(nfac - fac, nq1, nq2, nq00, nq01, tbasis00,
                                     tbasis01, wsp2, wsp1, out, isCollocated00,
                                     isCollocated01, threadBlock);
    }
}

/**
 * @brief Glue for the direction-1 normal faces: normal stage, then the
 * tangential stage if it is needed.
 *
 * The one-block-per-element twin of PhysExtractFaceN1KernelTrace3D. When
 * both tangential directions are collocated the trace grid @em is the
 * volume face grid, and PhysExtractEndFacesN1TOPKernelTrace3D writes
 * straight into @p out. Otherwise it writes into @p wsp1 and
 * PhysInterpFaceTOPKernelTrace resamples (nq0,nq2) onto `(nq10,nq11)`, using
 * @p wsp2 when neither direction is collocated.
 *
 * @tparam END_PTS_COLLOCATED1    Direction-1 volume rule contains the
 *                      domain endpoints.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   fac         Position of the first face within direction 1.
 * @param   nfac        One past the position of the last face.
 * @param   tstride     Row stride of the @c ntbasis table: the number of
 *                      traces the direction has.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis1    eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   nq10,nq11   Trace quadrature points of the two tangential
 *                      directions of a direction-1 face.
 * @param   tbasis10,tbasis11   eInterp tables to those trace points.
 * @param   wsp1        Face values between the two stages, in shared
 *                      memory.
 * @param   wsp2        Scratch of the fully general tangential stage.
 * @param   in          Volume field of this block's element.
 * @param   out         Face values on the trace grid.
 * @param   isCollocated10,isCollocated11   That tangential direction's
 *                      trace points coincide with the volume points it
 *                      comes from.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED1 = false, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysExtractFaceN1TOPKernelTrace3D(
    const unsigned int fac, const unsigned int nfac, const unsigned int tstride,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT ntbasis1, const unsigned int nq10,
    const unsigned int nq11, const TData *NEK_RESTRICT tbasis10,
    const TData *NEK_RESTRICT tbasis11, TData *NEK_RESTRICT wsp1,
    TData *NEK_RESTRICT wsp2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, bool isCollocated10, bool isCollocated11,
    const TthreadBlock &threadBlock)
{
    if (isCollocated10 && isCollocated11)
    {
        PhysExtractEndFacesN1TOPKernelTrace3D<END_PTS_COLLOCATED1>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis1, in, out, threadBlock);
    }
    else
    {
        PhysExtractEndFacesN1TOPKernelTrace3D<END_PTS_COLLOCATED1>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis1, in, wsp1, threadBlock);

        PhysInterpFaceTOPKernelTrace(nfac - fac, nq0, nq2, nq10, nq11, tbasis10,
                                     tbasis11, wsp2, wsp1, out, isCollocated10,
                                     isCollocated11, threadBlock);
    }
}

/**
 * @brief Glue for the direction-2 normal faces: normal stage, then the
 * tangential stage if it is needed.
 *
 * The one-block-per-element twin of PhysExtractFaceN2KernelTrace3D. When
 * both tangential directions are collocated the trace grid @em is the
 * volume face grid, and PhysExtractEndFacesN2TOPKernelTrace3D writes
 * straight into @p out. Otherwise it writes into @p wsp1 and
 * PhysInterpFaceTOPKernelTrace resamples (nq0,nq1) onto `(nq20,nq21)`, using
 * @p wsp2 when neither direction is collocated.
 *
 * @tparam END_PTS_COLLOCATED2    Direction-2 volume rule contains the
 *                      domain endpoints.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   fac         Position of the first face within direction 2.
 * @param   nfac        One past the position of the last face.
 * @param   tstride     Row stride of the @c ntbasis table: the number of
 *                      traces the direction has.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis2    eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   nq20,nq21   Trace quadrature points of the two tangential
 *                      directions of a direction-2 face.
 * @param   tbasis20,tbasis21   eInterp tables to those trace points.
 * @param   wsp1        Face values between the two stages, in shared
 *                      memory.
 * @param   wsp2        Scratch of the fully general tangential stage.
 * @param   in          Volume field of this block's element.
 * @param   out         Face values on the trace grid.
 * @param   isCollocated20,isCollocated21   That tangential direction's
 *                      trace points coincide with the volume points it
 *                      comes from.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED2 = false, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysExtractFaceN2TOPKernelTrace3D(
    const unsigned int fac, const unsigned int nfac, const unsigned int tstride,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT ntbasis2, const unsigned int nq20,
    const unsigned int nq21, const TData *NEK_RESTRICT tbasis20,
    const TData *NEK_RESTRICT tbasis21, TData *NEK_RESTRICT wsp1,
    TData *NEK_RESTRICT wsp2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, bool isCollocated20, bool isCollocated21,
    const TthreadBlock &threadBlock)
{
    if (isCollocated20 && isCollocated21)
    {
        PhysExtractEndFacesN2TOPKernelTrace3D<END_PTS_COLLOCATED2>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis2, in, out, threadBlock);
    }
    else
    {
        PhysExtractEndFacesN2TOPKernelTrace3D<END_PTS_COLLOCATED2>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis2, in, wsp1, threadBlock);

        PhysInterpFaceTOPKernelTrace(nfac - fac, nq0, nq1, nq20, nq21, tbasis20,
                                     tbasis21, wsp2, wsp1, out, isCollocated20,
                                     isCollocated21, threadBlock);
    }
}

/**
 * @brief Turn the runtime endpoint-collocation flag of direction 0 into
 * the template argument PhysExtractFaceN0TOPKernelTrace3D branches on.
 *
 * Both arms are written out so that both instantiations of the glue and
 * of its normal-stage leaf exist. Nothing else differs between them.
 *
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   fac,nfac    Half-open window of faces within direction 0.
 * @param   tstride     Row stride of the @c ntbasis table.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   nq00,nq01   Trace quadrature points of the two tangential
 *                      directions.
 * @param   tbasis00,tbasis01   eInterp tables to those trace points.
 * @param   wsp0,wsp1   Face workspace and general-contraction scratch,
 *                      handed on as the glue's @c wsp1 and @c wsp2.
 * @param   in          Volume field of this block's element.
 * @param   out         Face values on the trace grid.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags.
 * @param   endPtsCollocated0   Direction-0 volume rule contains the
 *                      domain endpoints.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void DispatchPhysExtractFaceN0TOPKernelTrace3D(
    const unsigned int fac, const unsigned int nfac, const unsigned int tstride,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const TData *NEK_RESTRICT ntbasis0, const unsigned int nq00,
    const unsigned int nq01, const TData *NEK_RESTRICT tbasis00,
    const TData *NEK_RESTRICT tbasis01, TData *NEK_RESTRICT wsp0,
    TData *NEK_RESTRICT wsp1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, bool isCollocated00, bool isCollocated01,
    bool endPtsCollocated0, const TthreadBlock &threadBlock)
{
    if (endPtsCollocated0)
    {
        PhysExtractFaceN0TOPKernelTrace3D<true>(
            fac, nfac, tstride, nm0, nm1, nm2, ntbasis0, nq00, nq01, tbasis00,
            tbasis01, wsp0, wsp1, in, out, isCollocated00, isCollocated01,
            threadBlock);
    }
    else
    {
        PhysExtractFaceN0TOPKernelTrace3D<false>(
            fac, nfac, tstride, nm0, nm1, nm2, ntbasis0, nq00, nq01, tbasis00,
            tbasis01, wsp0, wsp1, in, out, isCollocated00, isCollocated01,
            threadBlock);
    }
}

/**
 * @brief Turn the runtime endpoint-collocation flag of direction 1 into
 * the template argument PhysExtractFaceN1TOPKernelTrace3D branches on.
 *
 * Both arms are written out so that both instantiations of the glue and
 * of its normal-stage leaf exist. Nothing else differs between them.
 *
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   fac,nfac    Half-open window of faces within direction 1.
 * @param   tstride     Row stride of the @c ntbasis table.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   ntbasis1    eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   nq10,nq11   Trace quadrature points of the two tangential
 *                      directions.
 * @param   tbasis10,tbasis11   eInterp tables to those trace points.
 * @param   wsp0,wsp1   Face workspace and general-contraction scratch,
 *                      handed on as the glue's @c wsp1 and @c wsp2.
 * @param   in          Volume field of this block's element.
 * @param   out         Face values on the trace grid.
 * @param   isCollocated10,isCollocated11   Tangential collocation flags.
 * @param   endPtsCollocated1   Direction-1 volume rule contains the
 *                      domain endpoints.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void DispatchPhysExtractFaceN1TOPKernelTrace3D(
    const unsigned int fac, const unsigned int nfac, const unsigned int tstride,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const TData *NEK_RESTRICT ntbasis1, const unsigned int nq10,
    const unsigned int nq11, const TData *NEK_RESTRICT tbasis10,
    const TData *NEK_RESTRICT tbasis11, TData *NEK_RESTRICT wsp0,
    TData *NEK_RESTRICT wsp1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, bool isCollocated10, bool isCollocated11,
    bool endPtsCollocated1, const TthreadBlock &threadBlock)
{
    if (endPtsCollocated1)
    {
        PhysExtractFaceN1TOPKernelTrace3D<true>(
            fac, nfac, tstride, nm0, nm1, nm2, ntbasis1, nq10, nq11, tbasis10,
            tbasis11, wsp0, wsp1, in, out, isCollocated10, isCollocated11,
            threadBlock);
    }
    else
    {
        PhysExtractFaceN1TOPKernelTrace3D<false>(
            fac, nfac, tstride, nm0, nm1, nm2, ntbasis1, nq10, nq11, tbasis10,
            tbasis11, wsp0, wsp1, in, out, isCollocated10, isCollocated11,
            threadBlock);
    }
}

/**
 * @brief Turn the runtime endpoint-collocation flag of direction 2 into
 * the template argument PhysExtractFaceN2TOPKernelTrace3D branches on.
 *
 * Both arms are written out so that both instantiations of the glue and
 * of its normal-stage leaf exist. Nothing else differs between them.
 *
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   fac,nfac    Half-open window of faces within direction 2.
 * @param   tstride     Row stride of the @c ntbasis table.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   ntbasis2    eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   nq20,nq21   Trace quadrature points of the two tangential
 *                      directions.
 * @param   tbasis20,tbasis21   eInterp tables to those trace points.
 * @param   wsp0,wsp1   Face workspace and general-contraction scratch,
 *                      handed on as the glue's @c wsp1 and @c wsp2.
 * @param   in          Volume field of this block's element.
 * @param   out         Face values on the trace grid.
 * @param   isCollocated20,isCollocated21   Tangential collocation flags.
 * @param   endPtsCollocated2   Direction-2 volume rule contains the
 *                      domain endpoints.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void DispatchPhysExtractFaceN2TOPKernelTrace3D(
    const unsigned int fac, const unsigned int nfac, const unsigned int tstride,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const TData *NEK_RESTRICT ntbasis2, const unsigned int nq20,
    const unsigned int nq21, const TData *NEK_RESTRICT tbasis20,
    const TData *NEK_RESTRICT tbasis21, TData *NEK_RESTRICT wsp0,
    TData *NEK_RESTRICT wsp1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, bool isCollocated20, bool isCollocated21,
    bool endPtsCollocated2, const TthreadBlock &threadBlock)
{
    if (endPtsCollocated2)
    {
        PhysExtractFaceN2TOPKernelTrace3D<true>(
            fac, nfac, tstride, nm0, nm1, nm2, ntbasis2, nq20, nq21, tbasis20,
            tbasis21, wsp0, wsp1, in, out, isCollocated20, isCollocated21,
            threadBlock);
    }
    else
    {
        PhysExtractFaceN2TOPKernelTrace3D<false>(
            fac, nfac, tstride, nm0, nm1, nm2, ntbasis2, nq20, nq21, tbasis20,
            tbasis21, wsp0, wsp1, in, out, isCollocated20, isCollocated21,
            threadBlock);
    }
}

/**
 * @brief All faces of one three-dimensional element: the three normal
 * directions in packed trace order.
 *
 * The bulk worker, and the one-block-per-element twin of
 * PhysTraceExtract3DSumFacKernelCore. The direction-0 pair goes into the
 * first `2 * nq00 * nq01` entries of @p out, then the direction-1 faces,
 * a pair for every shape but the tetrahedron, then the direction-2
 * faces, a pair for the hexahedron and a single face otherwise. Each
 * group is dispatched as one window, so within a pair both faces come
 * out of a single pass of the normal stage.
 *
 * @tparam SHAPE_TYPE    Hex, Prism, NodalPrism, Pyr, Tet or NodalTet.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nq00,nq01,nq10,nq11,nq20,nq21   Trace quadrature points, per
 *                      normal direction and tangential direction.
 * @param   ntbasis0,ntbasis1,ntbasis2      eInterp tables
 *                      \f$h_p(\pm 1)\f$ per direction.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points.
 * @param   wsp0        Face values between the two stages, in shared
 *                      memory.
 * @param   wsp1        Scratch of the fully general tangential stage.
 * @param   in          Volume field of this block's element.
 * @param   out         All of this element's trace values.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2
 *                      Volume rule of each direction contains the domain
 *                      endpoints.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysTraceExtract3DSumFacTOPKernelCore(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq00, const unsigned int nq01, const unsigned int nq10,
    const unsigned int nq11, const unsigned int nq20, const unsigned int nq21,
    const TData *ntbasis0, const TData *ntbasis1, const TData *ntbasis2,
    const TData *tbasis00, const TData *tbasis01, const TData *tbasis10,
    const TData *tbasis11, const TData *tbasis20, const TData *tbasis21,
    TData *wsp0, TData *wsp1, const TData *in, TData *out, bool isCollocated00,
    bool isCollocated01, bool isCollocated10, bool isCollocated11,
    bool isCollocated20, bool isCollocated21, bool endPtsCollocated0,
    bool endPtsCollocated1, bool endPtsCollocated2,
    const TthreadBlock &threadBlock)
{
    // One ntbasis row per trace of the direction. A call covering part
    // of a direction still indexes at that stride, so it cannot be taken
    // from the loop bound; the bulk route covers each direction whole, so
    // here the two coincide.
    constexpr unsigned int tstride0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned int tstride1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned int tstride2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];

    // Faces with dir-0 normals (trace group tr=2).
    DispatchPhysExtractFaceN0TOPKernelTrace3D(
        0, tstride0, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01, tbasis00,
        tbasis01, wsp0, wsp1, in, out, isCollocated00, isCollocated01,
        endPtsCollocated0, threadBlock);

    unsigned int offset = tstride0 * nq00 * nq01;

    // Faces with dir-1 normals (trace group tr=1).
    DispatchPhysExtractFaceN1TOPKernelTrace3D(
        0, tstride1, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11, tbasis10,
        tbasis11, wsp0, wsp1, in, out + offset, isCollocated10, isCollocated11,
        endPtsCollocated1, threadBlock);
    offset += tstride1 * nq10 * nq11;

    // Faces with dir-2 normals (trace group tr=0).
    DispatchPhysExtractFaceN2TOPKernelTrace3D(
        0, tstride2, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21, tbasis20,
        tbasis21, wsp0, wsp1, in, out + offset, isCollocated20, isCollocated21,
        endPtsCollocated2, threadBlock);
}

/**
 * @brief Bulk path, one dimension: extract both end vertices of every
 * segment of the block.
 *
 * @details
 * The TOP counterpart of the one-dimensional
 * PhysTraceExtractKernelLauncher. One block takes one element at a
 * time from the grid-stride loop and its threads share the two-vertex
 * loop, so with fewer than two threads busy this is the one arm where
 * the thread-over-points mapping buys nothing; it exists for
 * uniformity, and because the bulk path must serve every shape.
 *
 * Element @em e of component @em c starts at `nmTot * (nelmt * c + e)`
 * in the input and at `2 * (nelmt * c + e)` in the output: the
 * width-one specialisation of the SumFac kernels' addressing, with
 * `warpsize` one, `iwarp` the element and `ilane` zero.
 *
 * @param   sizeParam1D Volume and trace point counts of the segment;
 *                      only nm0() is read.
 * @param   nelmt       Elements in the block, including padding.
 * @param   nmTot       Volume points per element.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$, indexed
 *                      `ntbasis0[p * 2 + t]`.
 * @param   in          Volume field, element contiguous.
 * @param   out         Trace field, element contiguous.
 * @param   wsp         Global workspace; a segment needs none.
 * @param   endPtsCollocated0   Volume rule carries the endpoints, so
 *                      each vertex value is a volume value.
 * @param   shmemptr    Dynamic shared memory; unused in one dimension.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TTraceSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter1D>()))
    PhysTraceExtractKernelLauncher(const TTraceSizeParameter1D sizeParam1D,
                                   const size_t nelmt, const unsigned int nmTot,
                                   const TData *NEK_RESTRICT ntbasis0,
                                   const TData *NEK_RESTRICT in,
                                   TData *NEK_RESTRICT out,
                                   [[maybe_unused]] TData *NEK_RESTRICT wsp,
                                   const bool endPtsCollocated0,
                                   [[maybe_unused]] unsigned char *shmemptr,
                                   const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter1D or "
                  "TemplatedTraceSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * (nelmt * c + e);
        TData *outptr      = out + 2 * (nelmt * c + e);

        if (endPtsCollocated0)
        {
            for (unsigned int t = idx0; t < 2u; t += stride)
            {
                outptr[t] = inptr[(t == 0u) ? 0u : (nm0 - 1u)];
            }
        }
        else
        {
            for (unsigned int t = idx0; t < 2u; t += stride)
            {
                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += inptr[p] * ntbasis0[p * 2 + t];
                }
                outptr[t] = tmp;
            }
        }
        localBarrier(threadBlock);
        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief Per-trace path, one dimension: extract one named vertex of
 * every segment into a chosen slot of the packed output.
 *
 * The single-trace counterpart of the one-dimensional
 * PhysTraceExtractKernelLauncher. One value per element, so thread
 * zero of the block writes it and the rest wait at the barrier.
 *
 * @param   traceid     0 for the lower vertex, 1 for the upper one;
 *                      also the column of @p ntbasis0 to read.
 * @param   numDataOut  Packed trace entries per element.
 * @param   outOffset   Where this trace's value sits within an
 *                      element's output.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TTraceSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter1D>()))
    PhysTraceExtractTraceKernelLauncher(
        const unsigned int traceid, const TTraceSizeParameter1D sizeParam1D,
        const size_t nelmt, const unsigned int nmTot,
        const unsigned int numDataOut, const unsigned int outOffset,
        const TData *NEK_RESTRICT ntbasis0, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const bool endPtsCollocated0, [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter1D or "
                  "TemplatedTraceSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();

    const unsigned int idx0 = getLocalIdx(threadBlock);

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * (nelmt * c + e);
        TData *outptr      = out + numDataOut * (nelmt * c + e) + outOffset;

        if (idx0 == 0u)
        {
            if (endPtsCollocated0)
            {
                outptr[0] = inptr[(traceid == 0u) ? 0u : (nm0 - 1u)];
            }
            else
            {
                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += inptr[p] * ntbasis0[p * 2 + traceid];
                }
                outptr[0] = tmp;
            }
        }
        localBarrier(threadBlock);
        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief Bulk path, two dimensions: extract every edge of every element
 * of the block.
 *
 * @details
 * The TOP counterpart of the two-dimensional
 * PhysTraceExtractKernelLauncher, walking the normal directions in
 * packing order through PhysTraceExtract2DSumFacTOPKernelCore.
 *
 * The edge block the normal stage writes lives in dynamic shared
 * memory, one PhysTraceExtractBlockSize() per block rather than the
 * SumFac path's one per element in the global workspace: with one
 * element resident per block at a time there is only ever one such
 * block live, and the host sizes the launch against the same helper so
 * the two cannot drift. @p wsp is therefore unused here.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TTraceSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter2D>()))
    PhysTraceExtractKernelLauncher(
        const TTraceSizeParameter2D sizeParam2D, const size_t nelmt,
        const unsigned int nmTot, const TData *NEK_RESTRICT ntbasis0,
        const TData *NEK_RESTRICT ntbasis1, const TData *NEK_RESTRICT basis0,
        const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const bool isCollocated0, const bool isCollocated1,
        const bool endPtsCollocated0, const bool endPtsCollocated1,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);
    TData *s_wsp = (TData *)shmemptr;

    const unsigned int nm0 = sizeParam2D.nm0();
    const unsigned int nm1 = sizeParam2D.nm1();
    const unsigned int nq0 = sizeParam2D.nq00();
    const unsigned int nq1 = sizeParam2D.nq10();

    const unsigned int nqTotOut = sizeParam2D.template nqTotTrace<SHAPE_TYPE>();

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * (nelmt * c + e);
        TData *outptr      = out + nqTotOut * (nelmt * c + e);

        PhysTraceExtract2DSumFacTOPKernelCore<SHAPE_TYPE>(
            nm0, nm1, nq0, nq1, ntbasis0, ntbasis1, basis0, basis1, s_wsp,
            inptr, outptr, isCollocated0, isCollocated1, endPtsCollocated0,
            endPtsCollocated1, threadBlock);
        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief Per-trace path, two dimensions: extract one named edge of
 * every element into a chosen slot of the packed output.
 *
 * GetTraceEdgeDispatch turns the edge id into a normal direction and a
 * one-edge window, exactly as on the SumFac path -- the trace
 * numbering does not depend on the thread mapping -- and the matching
 * @c NormalDir instantiation of BwdTransQuadSumFacTOPKernelTrace does
 * the work with both output offsets left at zero, @p out already
 * pointing at this trace's slot.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TTraceSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter2D>()))
    PhysTraceExtractTraceKernelLauncher(
        const unsigned int traceid, const TTraceSizeParameter2D sizeParam2D,
        const size_t nelmt, const unsigned int nmTot,
        const unsigned int numDataOut, const unsigned int outOffset,
        const TData *NEK_RESTRICT ntbasis0, const TData *NEK_RESTRICT ntbasis1,
        const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis10,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const bool isCollocated00,
        const bool isCollocated10, const bool endPtsCollocated0,
        const bool endPtsCollocated1, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);
    TData *s_wsp = (TData *)shmemptr;

    const unsigned int nm0  = sizeParam2D.nm0();
    const unsigned int nm1  = sizeParam2D.nm1();
    const unsigned int nq00 = sizeParam2D.nq00();
    const unsigned int nq10 = sizeParam2D.nq10();

    TraceExtractDispatch dispatch{};
    if (!GetTraceEdgeDispatch<SHAPE_TYPE>(traceid, dispatch))
    {
        return;
    }

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * (nelmt * c + e);
        TData *outptr      = out + numDataOut * (nelmt * c + e) + outOffset;

        if (dispatch.normalDir == 0u)
        {
            BwdTransQuadSumFacTOPKernelTrace<SHAPE_TYPE, 0>(
                nm0, nm1, nq00, nq10, dispatch.fac, dispatch.nfac, nq00,
                ntbasis0, tbasis00, inptr, outptr, s_wsp, isCollocated00,
                endPtsCollocated0, threadBlock);
        }
        else
        {
            BwdTransQuadSumFacTOPKernelTrace<SHAPE_TYPE, 1>(
                nm0, nm1, nq00, nq10, dispatch.fac, dispatch.nfac, nq10,
                ntbasis1, tbasis10, inptr, outptr, s_wsp, isCollocated10,
                endPtsCollocated1, threadBlock);
        }
        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief Bulk path, three dimensions: extract every face of every
 * element of the block.
 *
 * @details
 * The TOP counterpart of the three-dimensional
 * PhysTraceExtractKernelLauncher, walking the normal directions in
 * packing order through PhysTraceExtract3DSumFacTOPKernelCore.
 *
 * Shared memory is partitioned exactly as the SumFac path partitions
 * its per-element global workspace: `2 * PhysTraceExtractBlockSize()`
 * for the face block the normal stage writes, then one more for the
 * tangential scratch of PhysInterpFaceTOPKernelTrace -- the three
 * blocks PhysTraceExtractSharedMemorySize() asks for. The host sizes the
 * dynamic allocation against the same helper.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TTraceSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter3D>()))
    PhysTraceExtractKernelLauncher(
        const TTraceSizeParameter3D sizeParam3D, const size_t nelmt,
        const unsigned int nmTot, const TData *NEK_RESTRICT ntbasis0,
        const TData *NEK_RESTRICT ntbasis1, const TData *NEK_RESTRICT ntbasis2,
        const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
        const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
        const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const bool isCollocated00,
        const bool isCollocated01, const bool isCollocated10,
        const bool isCollocated11, const bool isCollocated20,
        const bool isCollocated21, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const bool endPtsCollocated2,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int max_wsp_size = PhysTraceExtractBlockSize(sizeParam3D);
    TData *s_wsp0                   = (TData *)shmemptr;
    TData *s_wsp1                   = s_wsp0 + 2u * max_wsp_size;

    const unsigned int nm0  = sizeParam3D.nm0();
    const unsigned int nm1  = sizeParam3D.nm1();
    const unsigned int nm2  = sizeParam3D.nm2();
    const unsigned int nq00 = sizeParam3D.nq00();
    const unsigned int nq01 = sizeParam3D.nq01();
    const unsigned int nq10 = sizeParam3D.nq10();
    const unsigned int nq11 = sizeParam3D.nq11();
    const unsigned int nq20 = sizeParam3D.nq20();
    const unsigned int nq21 = sizeParam3D.nq21();

    const unsigned int nqTotOut = sizeParam3D.template nqTotTrace<SHAPE_TYPE>();

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * (nelmt * c + e);
        TData *outptr      = out + nqTotOut * (nelmt * c + e);

        PhysTraceExtract3DSumFacTOPKernelCore<SHAPE_TYPE>(
            nm0, nm1, nm2, nq00, nq01, nq10, nq11, nq20, nq21, ntbasis0,
            ntbasis1, ntbasis2, tbasis00, tbasis01, tbasis10, tbasis11,
            tbasis20, tbasis21, s_wsp0, s_wsp1, inptr, outptr, isCollocated00,
            isCollocated01, isCollocated10, isCollocated11, isCollocated20,
            isCollocated21, endPtsCollocated0, endPtsCollocated1,
            endPtsCollocated2, threadBlock);
        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief Per-trace path, three dimensions: extract one named face of
 * every element into a chosen slot of the packed output.
 *
 * GetTraceFaceDispatch supplies the normal direction and a one-face
 * window, and the Dispatch layer fixes the endpoint template argument
 * before the per-direction glue runs. As in two dimensions the trace
 * tables are the shared ones.
 *
 * @note Every window GetTraceFaceDispatch produces is one face wide, so
 * the loop bound cannot double as the @c ntbasis row stride here; the
 * leaves take that stride separately from ShapeTypeNumTraceInDir, which
 * is what lets the lower face of a two-trace direction address the
 * table correctly.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TTraceSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter3D>()))
    PhysTraceExtractTraceKernelLauncher(
        const unsigned int traceid, const TTraceSizeParameter3D sizeParam3D,
        const size_t nelmt, const unsigned int nmTot,
        const unsigned int numDataOut, const unsigned int outOffset,
        const TData *NEK_RESTRICT ntbasis0, const TData *NEK_RESTRICT ntbasis1,
        const TData *NEK_RESTRICT ntbasis2, const TData *NEK_RESTRICT tbasis00,
        const TData *NEK_RESTRICT tbasis01, const TData *NEK_RESTRICT tbasis10,
        const TData *NEK_RESTRICT tbasis11, const TData *NEK_RESTRICT tbasis20,
        const TData *NEK_RESTRICT tbasis21, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const bool isCollocated00, const bool isCollocated01,
        const bool isCollocated10, const bool isCollocated11,
        const bool isCollocated20, const bool isCollocated21,
        const bool endPtsCollocated0, const bool endPtsCollocated1,
        const bool endPtsCollocated2, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int max_wsp_size = PhysTraceExtractBlockSize(sizeParam3D);
    TData *s_wsp0                   = (TData *)shmemptr;
    TData *s_wsp1                   = s_wsp0 + 2u * max_wsp_size;

    const unsigned int nm0  = sizeParam3D.nm0();
    const unsigned int nm1  = sizeParam3D.nm1();
    const unsigned int nm2  = sizeParam3D.nm2();
    const unsigned int nq00 = sizeParam3D.nq00();
    const unsigned int nq01 = sizeParam3D.nq01();
    const unsigned int nq10 = sizeParam3D.nq10();
    const unsigned int nq11 = sizeParam3D.nq11();
    const unsigned int nq20 = sizeParam3D.nq20();
    const unsigned int nq21 = sizeParam3D.nq21();

    // One ntbasis row per trace of the direction. The window
    // GetTraceFaceDispatch produces is one face wide, so the loop bound
    // cannot double as the row stride here: without this the lower face
    // of a two-trace direction would read the table at stride one.
    constexpr unsigned int tstride0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned int tstride1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned int tstride2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];

    TraceExtractDispatch dispatch{};
    if (!GetTraceFaceDispatch<SHAPE_TYPE>(traceid, dispatch))
    {
        return;
    }

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * (nelmt * c + e);
        TData *outptr      = out + numDataOut * (nelmt * c + e) + outOffset;

        switch (dispatch.normalDir)
        {
            case 0:
                DispatchPhysExtractFaceN0TOPKernelTrace3D(
                    dispatch.fac, dispatch.nfac, tstride0, nm0, nm1, nm2,
                    ntbasis0, nq00, nq01, tbasis00, tbasis01, s_wsp0, s_wsp1,
                    inptr, outptr, isCollocated00, isCollocated01,
                    endPtsCollocated0, threadBlock);
                break;
            case 1:
                DispatchPhysExtractFaceN1TOPKernelTrace3D(
                    dispatch.fac, dispatch.nfac, tstride1, nm0, nm1, nm2,
                    ntbasis1, nq10, nq11, tbasis10, tbasis11, s_wsp0, s_wsp1,
                    inptr, outptr, isCollocated10, isCollocated11,
                    endPtsCollocated1, threadBlock);
                break;
            case 2:
                DispatchPhysExtractFaceN2TOPKernelTrace3D(
                    dispatch.fac, dispatch.nfac, tstride2, nm0, nm1, nm2,
                    ntbasis2, nq20, nq21, tbasis20, tbasis21, s_wsp0, s_wsp1,
                    inptr, outptr, isCollocated20, isCollocated21,
                    endPtsCollocated2, threadBlock);
                break;
            default:
                break;
        }
        e += getBlockRange(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
