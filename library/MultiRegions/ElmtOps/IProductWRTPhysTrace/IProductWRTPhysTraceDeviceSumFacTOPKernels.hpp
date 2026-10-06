///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysTraceDeviceSumFacTOPKernels.hpp
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
// Description: IProductWRTPhysTrace SumFacTOP device kernels: one
// thread block per element on width-1 element-contiguous storage,
// threads striding the trace points and modes, following the volume-op
// SumFacTOP pattern.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysTraceDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernels of the trace inner product: one
 * thread block per element, threads striding the points and modes.
 *
 * @details
 * The dedicated thread-over-points counterpart of
 * IProductWRTPhysTraceDeviceSumFacKernels.hpp, and the adjoint of
 * PhysTraceExtractDeviceSumFacTOPKernels.hpp. Both this and the SumFac
 * family compute the same operator; they differ in how an element is
 * mapped onto threads, and so in how its data is addressed:
 *
 * - the SumFac kernels put one element on one warp *lane*, so entry
 *   @em n of element \f$e = (i_{warp}, i_{lane})\f$ lives at
 *   `buf[warpsize * n + ilane]`;
 * - these kernels put one element on one thread *block*, so an
 *   element's entries are contiguous, and the block's threads share the
 *   point and mode loops, striding them by the block size with a
 *   localBarrier() between dependent stages.
 *
 * That is why the host class pairs them with an implementation
 * interleave width of one (IProductWRTPhysTraceDeviceSumFac.hpp) and
 * with the SumFacTOP launch geometry, and why the trace-mode buffer and
 * the contraction scratch live in dynamic shared memory rather than in
 * the global workspace the SumFac path partitions per warp.
 *
 * Both routes accumulate rather than overwrite when asked to: the bulk
 * kernels branch on the runtime flag to the APPEND instantiation of
 * their core, and the per-trace kernels zero the volume modes first
 * only when it is clear.
 *
 * The per-element size helpers, the trace-Jacobian reader and the shape
 * tables are shared with the SumFac kernels and included from there
 * rather than duplicated: they describe the operator's data layout and
 * trace numbering, which do not depend on the thread mapping.
 *
 * @see IProductWRTPhysTraceDeviceSumFac.hpp for the host dispatch
 * that launches these under the SumFacTOP tag.
 * @see IProductWRTPhysTraceDeviceSumFacKernels.hpp for the
 * warp-per-lane kernels and the shared helpers.
 * @see PhysTraceExtractDeviceSumFacTOPKernels.hpp for the adjoint
 * operator's TOP kernels.
 */

#pragma once

#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceDeviceSumFacKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

// Helper functions

/// @brief Workspace of a whole block, per component: none, the
/// intermediates living in the dynamic shared memory
/// IProductWRTPhysTraceSharedMemorySize() sizes.
template <typename Implementation, typename TTraceSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTPhysTraceWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Workspace of a whole block, per component: none, the
/// intermediates living in the dynamic shared memory
/// IProductWRTPhysTraceSharedMemorySize() sizes.
template <typename Implementation, typename TTraceSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTPhysTraceWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter2D sizeParam2D,
    [[maybe_unused]] const unsigned int ntrace = 2u)
{
    return 0;
}

/// @brief Workspace of a whole block, per component: none, the
/// intermediates living in the dynamic shared memory
/// IProductWRTPhysTraceSharedMemorySize() sizes.
template <typename Implementation, typename TTraceSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTPhysTraceWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter3D sizeParam3D)
{
    return 0;
}

/// @brief Dynamic shared memory one block needs, in elements: none, a
/// segment's traces being points carrying no surface measure.
template <typename Implementation, typename TTraceSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTPhysTraceSharedMemorySize(
    [[maybe_unused]] const TTraceSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Dynamic shared memory one block needs, in elements: @p ntrace
/// trace-mode buffers, one element being resident per block at a time.
///
/// @param   ntrace  Edges the buffer must hold at once: two on the bulk
///                  path, which integrates an edge pair in one call,
///                  and one on the per-trace path.
template <typename Implementation, typename TTraceSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTPhysTraceSharedMemorySize(
    const TTraceSizeParameter2D sizeParam2D, const unsigned int ntrace = 2u)
{
    return ntrace * IProductWRTPhysTraceEdgeModeBlockSize(sizeParam2D);
}

/// @brief Dynamic shared memory one block needs, in elements: the face
/// mode block followed by the contraction scratch, the split the kernels
/// re-derive from the same two helpers. The bulk and the per-trace path
/// take the same.
template <typename Implementation, typename TTraceSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTPhysTraceSharedMemorySize(
    const TTraceSizeParameter3D sizeParam3D)
{
    return IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D) +
           IProductWRTPhysTraceFaceScratchSize(sizeParam3D);
}

// -----------------------------------------------------------------------------
// 2D KERNELS
// -----------------------------------------------------------------------------

/**
 * @brief Leaf kernel, stage one in two dimensions: integrate an edge, or
 * a pair of edges, against the hat functions of the in-edge direction.
 *
 * The one-block-per-element twin of IPWRTPhysEdgeKernel. The threads of
 * the block own the flattened (edge, mode) pairs and each carries its
 * own quadrature reduction to completion, so the sum stays in a register
 * and no thread reads an entry another one is writing. For every edge in
 * [@p edg, @p nedge) and every volume point @em q of the in-edge
 * direction this forms
 * \f$F_q = \sum_i w_i\, h_q(\xi^{tr}_i)\, J\, \hat f_i\f$ over the
 * @p tnq trace quadrature points. Under @p isCollocated the trace points
 * are the volume points (`nm == tnq`, which holds by construction of
 * the collocation flags), the table is the identity and the contraction
 * degenerates to a pointwise multiply by the weight and the Jacobian.
 *
 * @p out is the caller's trace-mode workspace, not the volume field: it
 * receives @p nm entries per edge, edge-major, starting at zero. For
 * regular geometry the Jacobian slot is `e - edg`, that is relative to
 * the first edge of this call, the caller having already offset
 * @p tjac to the right face. The kernel closes on a barrier, so the
 * workspace is visible to the whole block on return.
 *
 * @tparam DEFORMED      Trace Jacobian varies point by point.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   edg         Position of the first edge within its direction.
 * @param   nedge       One past the position of the last edge.
 * @param   nm          Volume quadrature points of the in-edge
 *                      direction, hence entries produced per edge.
 * @param   tnq         Trace quadrature points of the edge.
 * @param   tbasis      eInterp table \f$h_q(\xi^{tr}_i)\f$, indexed
 *                      `tbasis[q * tnq + i]`.
 * @param   tw          Trace quadrature weights.
 * @param   tjac        Trace Jacobian of this block's element.
 * @param   in          Packed trace values of the edges covered.
 * @param   out         Trace-mode workspace in shared memory.
 * @param   isCollocated  Trace points coincide with the volume points.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysEdgeTOPKernel(
    const unsigned edg, const unsigned nedge, const unsigned nm,
    const unsigned tnq, const TData *NEK_RESTRICT tbasis,
    const TData *NEK_RESTRICT tw, const TData *NEK_RESTRICT tjac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    const unsigned int nloc   = nedge - edg;

    for (unsigned int cnt = idx0; cnt < nloc * nm; cnt += stride)
    {
        const unsigned int q  = cnt % nm;
        const unsigned int le = cnt / nm;

        if (isCollocated)
        {
            const TData jac_val = DEFORMED ? tjac[cnt] : tjac[le];
            out[cnt]            = in[cnt] * jac_val * tw[q];
        }
        else
        {
            const unsigned int qOffset = le * tnq;
            const unsigned int bOffset = q * tnq;
            TData sum_q                = 0.0;
            for (unsigned int i = 0; i < tnq; ++i)
            {
                const TData jac_val = DEFORMED ? tjac[qOffset + i] : tjac[le];
                sum_q +=
                    in[qOffset + i] * tbasis[bOffset + i] * jac_val * tw[i];
            }
            out[cnt] = sum_q;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Leaf kernel, stage two in two dimensions: lift the results of a
 * direction-0 normal edge into the volume field.
 *
 * The one-block-per-element twin of AddEdgeN0ToVolKernel. Threads own
 * the output modes and accumulate the contributions of every edge of the
 * group in a register, so the `+=` onto @p out never races; the SumFac
 * version, owning a whole element per lane, loops the edges outermost
 * instead. Applies \f$\Lambda_{iq} \mathrel{+}= h_i(\pm 1)\, F_q\f$, the
 * volume field being indexed `nm0 * q + i`. Under
 * @p END_PTS_COLLOCATED0 the interpolation is a delta and the edge lands
 * entirely in the boundary plane `i = e * (nm0 - 1)`, the loop variable
 * @em e being the position of the edge within its direction (0 lower, 1
 * upper) rather than its index within this call, which is why the caller
 * passes absolute positions.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   edg     Position of the first edge within direction 0.
 * @param   nedge   One past the position of the last edge.
 * @param   tstride Row stride of @p nbasis, that is the number of
 *                  traces direction 0 has, independent of how many this
 *                  call covers.
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_i(\pm 1)\f$ of direction 0.
 * @param   in      Trace-mode workspace written by
 *                  IPWRTPhysEdgeTOPKernel.
 * @param   out     Volume field of this element, accumulated onto.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED0, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddEdgeN0ToVolTOPKernel(
    const unsigned edg, const unsigned nedge, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const TData *NEK_RESTRICT nbasis,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    const unsigned int nloc   = nedge - edg;

    if constexpr (END_PTS_COLLOCATED0)
    {
        for (unsigned int cnt = idx0; cnt < nloc * nm1; cnt += stride)
        {
            const unsigned int q  = cnt % nm1;
            const unsigned int le = cnt / nm1;
            const unsigned int e  = edg + le;
            out[nm0 * q + e * (nm0 - 1u)] += in[cnt];
        }
    }
    else
    {
        for (unsigned int idx = idx0; idx < nm0 * nm1; idx += stride)
        {
            const unsigned int i = idx % nm0;
            const unsigned int q = idx / nm0;

            TData acc = 0.0;
            for (unsigned int le = 0; le < nloc; ++le)
            {
                acc += in[le * nm1 + q] * nbasis[i * tstride + (edg + le)];
            }
            out[idx] += acc;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Leaf kernel, stage two in two dimensions: lift the results of a
 * direction-1 normal edge into the volume field.
 *
 * The direction-1 twin of AddEdgeN0ToVolTOPKernel, and the
 * one-block-per-element twin of AddEdgeN1ToVolKernel: here the edge
 * carries @p nm0 entries and the lift runs along direction 1, so under
 * the endpoint fast path the whole edge lands in the row
 * `j = e * (nm1 - 1)` and the general arm accumulates
 * `out[nm0 * q + i] += in[cnt + i] * h_q(\pm 1)`.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the
 *                            domain endpoints.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   edg     Position of the first edge within direction 1.
 * @param   nedge   One past the position of the last edge.
 * @param   tstride Row stride of @p nbasis, that is the number of
 *                  traces direction 1 has (two for a quadrilateral, one
 *                  for the triangles), independent of how many this
 *                  call covers.
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   in      Trace-mode workspace written by
 *                  IPWRTPhysEdgeTOPKernel.
 * @param   out     Volume field of this element, accumulated onto.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED1, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddEdgeN1ToVolTOPKernel(
    const unsigned edg, const unsigned nedge, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const TData *NEK_RESTRICT nbasis,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    const unsigned int nloc   = nedge - edg;

    if constexpr (END_PTS_COLLOCATED1)
    {
        for (unsigned int cnt = idx0; cnt < nloc * nm0; cnt += stride)
        {
            const unsigned int q  = cnt % nm0;
            const unsigned int le = cnt / nm0;
            const unsigned int e  = edg + le;
            out[e * nm0 * (nm1 - 1u) + q] += in[cnt];
        }
    }
    else // project to interior
    {
        for (unsigned int idx = idx0; idx < nm0 * nm1; idx += stride)
        {
            const unsigned int i = idx % nm0;
            const unsigned int q = idx / nm0;

            TData acc = 0.0;
            for (unsigned int le = 0; le < nloc; ++le)
            {
                acc += in[le * nm0 + i] * nbasis[q * tstride + (edg + le)];
            }
            out[idx] += acc;
        }
    }

    localBarrier(threadBlock);
}

/**
 * @brief Glue for a direction-0 normal edge: integrate, then lift.
 *
 * The one-block-per-element twin of IPWRTPhysTraceEdgeN0Kernel. Routes
 * the direction's tables and counts into the two leaf stages, which
 * carry their own barriers. The in-edge direction of a direction-0
 * normal is direction 1, so the integration produces @p nm1 entries per
 * edge into @p wsp, which the lift then spreads along direction 0.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   edg     Position of the first edge within direction 0.
 * @param   nedge   One past the position of the last edge.
 * @param   tstride Row stride of @p nbasis: the number of traces
 *                  direction 0 has.
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_i(\pm 1)\f$ of direction 0.
 * @param   tnq     Trace quadrature points of the edge.
 * @param   tbasis  eInterp table to the trace points of direction 0.
 * @param   tw      Trace quadrature weights of direction 0.
 * @param   tjac    Trace Jacobian of this block's element.
 * @param   wsp     Trace-mode scratch in shared memory, at least `nm1`
 *                  entries per edge covered.
 * @param   in      Packed trace values of the edges covered.
 * @param   out     Volume field of this element, accumulated onto.
 * @param   isCollocated  Trace points coincide with the volume points.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED0, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceEdgeN0TOPKernel(
    const unsigned edg, const unsigned nedge, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const TData *NEK_RESTRICT nbasis,
    const unsigned tnq, const TData *NEK_RESTRICT tbasis,
    const TData *NEK_RESTRICT tw, const TData *NEK_RESTRICT tjac, TData *wsp,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated, const TthreadBlock &threadBlock)
{
    IPWRTPhysEdgeTOPKernel<DEFORMED>(edg, nedge, nm1, tnq, tbasis, tw, tjac, in,
                                     wsp, isCollocated, threadBlock);

    AddEdgeN0ToVolTOPKernel<END_PTS_COLLOCATED0>(edg, nedge, tstride, nm0, nm1,
                                                 nbasis, wsp, out, threadBlock);
}

/**
 * @brief Glue for a direction-1 normal edge: integrate, then lift.
 *
 * The twin of IPWRTPhysTraceEdgeN0TOPKernel one direction over, and the
 * one-block-per-element twin of IPWRTPhysTraceEdgeN1Kernel. The in-edge
 * direction is now direction 0, so the integration produces @p nm0
 * entries per edge into @p wsp and the lift spreads them along
 * direction 1.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   edg     Position of the first edge within direction 1.
 * @param   nedge   One past the position of the last edge.
 * @param   tstride Row stride of @p nbasis: the number of traces
 *                  direction 1 has (two for a quadrilateral, one for
 *                  the triangles).
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   tnq     Trace quadrature points of the edge.
 * @param   tbasis  eInterp table to the trace points of direction 1.
 * @param   tw      Trace quadrature weights of direction 1.
 * @param   tjac    Trace Jacobian of this block's element.
 * @param   wsp     Trace-mode scratch in shared memory, at least `nm0`
 *                  entries per edge covered.
 * @param   in      Packed trace values of the edges covered.
 * @param   out     Volume field of this element, accumulated onto.
 * @param   isCollocated  Trace points coincide with the volume points.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED1, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceEdgeN1TOPKernel(
    const unsigned edg, const unsigned nedge, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const TData *NEK_RESTRICT nbasis,
    const unsigned tnq, const TData *NEK_RESTRICT tbasis,
    const TData *NEK_RESTRICT tw, const TData *NEK_RESTRICT tjac,
    TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated,
    const TthreadBlock &threadBlock)
{
    IPWRTPhysEdgeTOPKernel<DEFORMED>(edg, nedge, nm0, tnq, tbasis, tw, tjac, in,
                                     wsp, isCollocated, threadBlock);

    AddEdgeN1ToVolTOPKernel<END_PTS_COLLOCATED1>(edg, nedge, tstride, nm0, nm1,
                                                 nbasis, wsp, out, threadBlock);
}

/**
 * @brief Zero one element's volume field across the thread block.
 *
 * The block-wide counterpart of the per-lane zeroing the SumFac wrappers
 * open with: threads stride the @p nmodes entries and the kernel closes
 * on a barrier, so the field is ready for accumulation on return. Called
 * only when @c APPEND is false.
 *
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   nmodes      Entries of the volume field, `nm0 * nm1` in two
 *                      dimensions and `nm0 * nm1 * nm2` in three.
 * @param   out         Volume field of this element.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ZeroModesTOPKernel(
    const unsigned int nmodes, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    for (unsigned int idx = idx0; idx < nmodes; idx += stride)
    {
        out[idx] = 0.0;
    }
    localBarrier(threadBlock);
}

/**
 * @brief All edges of one two-dimensional element: zero the volume
 * field, then walk the packed traces.
 *
 * The bulk wrapper, and the one-block-per-element twin of
 * IProductWRTPhysTrace2DKernel. It applies the direction-0 pair first,
 * then advances @p in by the pair's `2 * tnq00` values and @p tjac by
 * the same number of points when deformed or by two face slots when
 * regular, and applies the direction-1 traces: a pair, or a single edge
 * when @c SHAPE_TYPE is exactly @c Tri, whose direction-1 upper end is
 * the collapsed vertex. Quadrilateral and triangle run the identical
 * leaf kernels; the hypotenuse is simply the \f$\eta_0 = +1\f$ line of
 * the collapsed square and its edge Jacobian carries the physical metric
 * of the slant.
 *
 * The @c endPtsCollocated flags are runtime booleans, so each call site
 * is written out twice to instantiate both arms of the leaf template.
 *
 * @tparam SHAPE_TYPE    Quad, Tri or NodalTri.
 * @tparam DEFORMED      Trace Jacobian varies point by point.
 * @tparam APPEND        Accumulate onto @p out instead of zeroing it
 *                       first.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   nm0,nm1         Volume quadrature points per direction.
 * @param   nbasis0,nbasis1 eInterp tables \f$h_p(\pm 1)\f$ per
 *                          direction.
 * @param   tnq00,tnq10     Trace quadrature points of the direction-0
 *                          and direction-1 edges.
 * @param   tbasis0,tbasis1 eInterp tables to the trace points, per
 *                          direction.
 * @param   tw00,tw10       Trace quadrature weights per direction.
 * @param   tjac            Trace Jacobian of this block's element.
 * @param   wsp             Trace-mode scratch in shared memory,
 *                          `2 * max(nm0,nm1)` entries.
 * @param   in              Packed trace values of this element.
 * @param   out             Volume field of this element.
 * @param   isCollocated0   Direction-0 trace points coincide with the
 *                          volume points.
 * @param   isCollocated1   Direction-1 trace points coincide with the
 *                          volume points.
 * @param   endPtsCollocated0  Direction-0 volume rule contains the
 *                          domain endpoints.
 * @param   endPtsCollocated1  Direction-1 volume rule contains the
 *                          domain endpoints.
 * @param   threadBlock     Thread block cooperating on this element.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool APPEND,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTPhysTrace2DTOPKernel(
    const unsigned nm0, const unsigned nm1, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const unsigned tnq00,
    const unsigned tnq10, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT tw00,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tjac,
    TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in, TData *out,
    const bool isCollocated0, const bool isCollocated1, bool endPtsCollocated0,
    bool endPtsCollocated1, const TthreadBlock &threadBlock)
{
    if constexpr (!APPEND)
    {
        ZeroModesTOPKernel(nm0 * nm1, out, threadBlock);
    }

    if (endPtsCollocated0)
    {
        IPWRTPhysTraceEdgeN0TOPKernel<true, DEFORMED>(
            0, 2, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac, wsp, in,
            out, isCollocated0, threadBlock);
    }
    else
    {
        IPWRTPhysTraceEdgeN0TOPKernel<false, DEFORMED>(
            0, 2, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac, wsp, in,
            out, isCollocated0, threadBlock);
    }

    // edge offset for the following edges (width-1 layout)
    const unsigned offset  = 2 * tnq00;
    const unsigned joffset = (DEFORMED) ? 2 * tnq00 : 2;

    constexpr unsigned nedge1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    if (endPtsCollocated1)
    {
        IPWRTPhysTraceEdgeN1TOPKernel<true, DEFORMED>(
            0, nedge1, nedge1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
            tjac + joffset, wsp, in + offset, out, isCollocated1, threadBlock);
    }
    else
    {
        IPWRTPhysTraceEdgeN1TOPKernel<false, DEFORMED>(
            0, nedge1, nedge1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
            tjac + joffset, wsp, in + offset, out, isCollocated1, threadBlock);
    }
}
// -----------------------------------------------------------------------------
// 3D KERNELS
// -----------------------------------------------------------------------------

/**
 * @brief Leaf kernel, stage one in three dimensions: integrate a face, or
 * a pair of faces, against the hat functions of the two in-face
 * directions.
 *
 * The one-block-per-element twin of IPWRTPhysFaceKernel: the threads own
 * the flattened per-face index and each keeps its reduction in a
 * register. For every face in [@p fac, @p nface) it forms
 * \f[
 *   F_{pq} = \sum_j w_j\, h_q(\xi^{tr}_j) \sum_i w_i\, h_p(\xi^{tr}_i)\,
 *            J(\xi^{tr}_i, \xi^{tr}_j)\, \hat f(\xi^{tr}_i, \xi^{tr}_j),
 * \f]
 * writing `nm0 * nm1` entries per face at `q * nm0 + p`, face-major,
 * into the caller's trace-mode workspace @p out.
 *
 * Four arms cover the collocation cases: both directions collocated is a
 * pointwise multiply by the two weights and the Jacobian; one direction
 * collocated contracts the other only and scales by the collocated
 * direction's weight; neither collocated runs the two contractions in
 * sequence through @p wsp, with a barrier between the stages, which is
 * why the caller must keep @p wsp and @p out distinct.
 *
 * @note @p nm0 and @p nm1 are the in-face counts the glue level
 * supplies, `(nm1,nm2)` for an \f$N_0\f$ face, `(nm0,nm2)` for
 * \f$N_1\f$ and `(nm0,nm1)` for \f$N_2\f$, not the element's
 * direction-0 and direction-1 counts.
 *
 * @tparam DEFORMED      Trace Jacobian varies point by point.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   fac          Position of the first face within its direction.
 * @param   nface        One past the position of the last face.
 * @param   nm0,nm1      Volume quadrature points of the two in-face
 *                       directions, hence entries produced per face.
 * @param   tnq0,tnq1    Trace quadrature points of the two in-face
 *                       directions.
 * @param   tbasis0      eInterp table to the trace points of the first
 *                       in-face direction, `tbasis0[p * tnq0 + i]`.
 * @param   tbasis1      The same for the second in-face direction.
 * @param   tw0,tw1      Trace quadrature weights per in-face direction.
 * @param   tjac         Trace Jacobian of this block's element.
 * @param   wsp          Scratch for the fully general arm in shared
 *                       memory, `nm0 * tnq1` entries per face covered.
 * @param   in           Packed trace values of the faces covered.
 * @param   out          Trace-mode workspace in shared memory.
 * @param   isCollocated0  First in-face direction is collocated.
 * @param   isCollocated1  Second in-face direction is collocated.
 * @param   threadBlock  Thread block cooperating on this element.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysFaceTOPKernel(
    const unsigned fac, const unsigned nface, const unsigned nm0,
    const unsigned nm1, const unsigned tnq0, const unsigned tnq1,
    const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
    const TData *NEK_RESTRICT tw0, const TData *NEK_RESTRICT tw1,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated0, const bool isCollocated1,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    const unsigned int nf     = nface - fac;

    if (isCollocated0 && isCollocated1) // simply multiply by weights
    {
        const unsigned total = nf * tnq1 * tnq0;
        for (unsigned cnt = idx0; cnt < total; cnt += stride)
        {
            const unsigned f    = cnt / (tnq1 * tnq0);
            const unsigned rem  = cnt % (tnq1 * tnq0);
            const unsigned j    = rem / tnq0;
            const unsigned i    = rem % tnq0;
            const TData jac_val = DEFORMED ? tjac[cnt] : tjac[f];
            out[cnt]            = in[cnt] * jac_val * tw0[i] * tw1[j];
        }
    }
    else if (isCollocated0) // full inner product in dir 1
    {
        const unsigned total = nf * nm1 * nm0;
        for (unsigned cnt = idx0; cnt < total; cnt += stride)
        {
            const unsigned f        = cnt / (nm1 * nm0);
            const unsigned rem      = cnt % (nm1 * nm0);
            const unsigned q        = rem / nm0;
            const unsigned i        = rem % nm0;
            const unsigned inOffset = f * tnq0 * tnq1;
            const unsigned qoffset  = q * tnq1;
            TData sum_q             = 0.0;
            for (unsigned j = 0; j < tnq1; ++j)
            {
                const unsigned joffset = j * tnq0 + inOffset;
                const TData jac_val    = DEFORMED ? tjac[joffset + i] : tjac[f];
                sum_q +=
                    in[joffset + i] * jac_val * tbasis1[qoffset + j] * tw1[j];
            }
            out[cnt] = sum_q * tw0[i];
        }
    }
    else if (isCollocated1) // full inner product in dir 0
    {
        const unsigned total = nf * tnq1 * nm0;
        for (unsigned cnt = idx0; cnt < total; cnt += stride)
        {
            const unsigned f       = cnt / (tnq1 * nm0);
            const unsigned rem     = cnt % (tnq1 * nm0);
            const unsigned j       = rem / nm0;
            const unsigned p       = rem % nm0;
            const unsigned joffset = j * tnq0 + f * tnq0 * tnq1;
            const unsigned poffset = p * tnq0;
            TData sum_p            = 0.0;
            for (unsigned i = 0; i < tnq0; ++i)
            {
                const TData jac_val = DEFORMED ? tjac[joffset + i] : tjac[f];
                sum_p +=
                    in[joffset + i] * jac_val * tbasis0[poffset + i] * tw0[i];
            }
            out[cnt] = sum_p * tw1[j];
        }
    }
    else // full inner product, two stages through wsp
    {
        // The two stages run as one flat loop each over the (face, entry)
        // pairs, in the same shape as the collocated arms above, rather
        // than as a per-face loop carrying a barrier. Keeping the
        // barriers in straight-line code costs nothing -- the glue
        // hands this kernel one face at a time -- and leaves the arm
        // with the same single trailing barrier every other arm has.
        const unsigned total0 = nf * tnq1 * nm0;
        for (unsigned cnt = idx0; cnt < total0; cnt += stride)
        {
            const unsigned f       = cnt / (tnq1 * nm0);
            const unsigned rem     = cnt % (tnq1 * nm0);
            const unsigned j       = rem / nm0;
            const unsigned p       = rem % nm0;
            const unsigned joffset = j * tnq0 + f * tnq0 * tnq1;
            const unsigned poffset = p * tnq0;
            TData sum_p            = 0.0;
            for (unsigned i = 0; i < tnq0; ++i)
            {
                const TData jac_val = DEFORMED ? tjac[joffset + i] : tjac[f];
                sum_p +=
                    in[joffset + i] * jac_val * tbasis0[poffset + i] * tw0[i];
            }
            wsp[cnt] = sum_p;
        }
        localBarrier(threadBlock);

        const unsigned total1 = nf * nm1 * nm0;
        for (unsigned cnt = idx0; cnt < total1; cnt += stride)
        {
            const unsigned f       = cnt / (nm1 * nm0);
            const unsigned rem     = cnt % (nm1 * nm0);
            const unsigned q       = rem / nm0;
            const unsigned p       = rem % nm0;
            const unsigned qoffset = q * tnq1;
            const unsigned wspoff  = f * tnq1 * nm0;
            TData sum_q            = 0.0;
            for (unsigned j = 0; j < tnq1; ++j)
            {
                sum_q +=
                    wsp[wspoff + j * nm0 + p] * tbasis1[qoffset + j] * tw1[j];
            }
            out[cnt] = sum_q;
        }
    }
    localBarrier(threadBlock);
}

/**
 * @brief Leaf kernel, stage two in three dimensions: lift the results of
 * a direction-0 normal face into the volume field.
 *
 * The one-block-per-element twin of AddFaceN0ToVolKernel. Threads
 * own the output modes and accumulate the contributions of every face of
 * the group in a register, so the `+=` onto @p out never races. Applies
 * \f$\Lambda_{pqr} \mathrel{+}= h_p(\pm 1)\, F_{qr}\f$, the volume field being
 * indexed `r * nm0 * nm1 + q * nm0 + p`. Under @p END_PTS_COLLOCATED0
 * the interpolation is a delta and the face lands entirely in the plane
 * `p = f * (nm0 - 1)`, @em f being the position of the face within its
 * direction rather than its index within this call.
 *
 * The general arm indexes @p nbasis with the stride passed in
 * explicitly, so a direction carrying a single trace reads its
 * one-column table correctly.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   fac       Position of the first face within direction 0.
 * @param   nface     One past the position of the last face.
 * @param   tstride   Row stride of @p nbasis, that is the number of
 *                    traces direction 0 has.
 * @param   nm0,nm1,nm2   Volume quadrature points per direction.
 * @param   nbasis    eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   in        Trace-mode workspace written by
 *                    IPWRTPhysFaceTOPKernel, `nm1 * nm2` entries per face.
 * @param   out       Volume field of this element, accumulated onto.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED0, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddFaceN0ToVolTOPKernel(
    const unsigned fac, const unsigned nface, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const unsigned nm2,
    const TData *NEK_RESTRICT nbasis, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (END_PTS_COLLOCATED0)
    {
        const unsigned nm01  = nm0 * nm1;
        const unsigned total = (nface - fac) * nm2 * nm1;
        for (unsigned cnt = idx0; cnt < total; cnt += stride)
        {
            const unsigned f   = fac + cnt / (nm2 * nm1);
            const unsigned rem = cnt % (nm2 * nm1);
            const unsigned r   = rem / nm1;
            const unsigned q   = rem % nm1;
            out[f * (nm0 - 1) + nm01 * r + q * nm0] += in[cnt];
        }
    }
    else
    {
        const unsigned total = nm2 * nm1 * nm0;
        for (unsigned ocnt = idx0; ocnt < total; ocnt += stride)
        {
            const unsigned r   = ocnt / (nm1 * nm0);
            const unsigned rem = ocnt % (nm1 * nm0);
            const unsigned q   = rem / nm0;
            const unsigned p   = rem % nm0;
            TData acc          = 0.0;
            for (unsigned f = fac; f < nface; ++f)
            {
                acc += in[((f - fac) * nm2 + r) * nm1 + q] *
                       nbasis[p * tstride + f];
            }
            out[ocnt] += acc;
        }
    }
    localBarrier(threadBlock);
}

/**
 * @brief Leaf kernel, stage two in three dimensions: lift the results of
 * a direction-1 normal face into the volume field.
 *
 * The one-block-per-element twin of AddFaceN1ToVolKernel. The direction-1 twin
 * of AddFaceN0ToVolTOPKernel. Threads own the output modes and accumulate the
 * contributions of every face of the group in a register, so the `+=` onto @p
 * out never races. Applies \f$\Lambda_{pqr} \mathrel{+}= h_q(\pm 1)\,
 * F_{pr}\f$, the volume field being indexed `r * nm0 * nm1 + q * nm0 + p`.
 * Under @p END_PTS_COLLOCATED1 the interpolation is a delta and the face lands
 * entirely in the plane `q = f * (nm1 - 1)`, @em f being the position of the
 * face within its direction rather than its index within this call.
 *
 * The general arm indexes @p nbasis with the stride passed in
 * explicitly, so a direction carrying a single trace reads its
 * one-column table correctly.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the
 *                            domain endpoints.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   fac       Position of the first face within direction 1.
 * @param   nface     One past the position of the last face.
 * @param   tstride   Row stride of @p nbasis, that is the number of
 *                    traces direction 1 has (one for a tetrahedron).
 * @param   nm0,nm1,nm2   Volume quadrature points per direction.
 * @param   nbasis    eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   in        Trace-mode workspace written by
 *                    IPWRTPhysFaceTOPKernel, `nm0 * nm2` entries per face.
 * @param   out       Volume field of this element, accumulated onto.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED1, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddFaceN1ToVolTOPKernel(
    const unsigned fac, const unsigned nface, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const unsigned nm2,
    const TData *NEK_RESTRICT nbasis, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (END_PTS_COLLOCATED1)
    {
        const unsigned nm01  = nm0 * nm1;
        const unsigned total = (nface - fac) * nm2 * nm0;
        for (unsigned cnt = idx0; cnt < total; cnt += stride)
        {
            const unsigned f   = fac + cnt / (nm2 * nm0);
            const unsigned rem = cnt % (nm2 * nm0);
            const unsigned r   = rem / nm0;
            const unsigned p   = rem % nm0;
            out[f * nm0 * (nm1 - 1) + nm01 * r + p] += in[cnt];
        }
    }
    else
    {
        const unsigned total = nm2 * nm1 * nm0;
        for (unsigned ocnt = idx0; ocnt < total; ocnt += stride)
        {
            const unsigned r   = ocnt / (nm1 * nm0);
            const unsigned rem = ocnt % (nm1 * nm0);
            const unsigned q   = rem / nm0;
            const unsigned p   = rem % nm0;
            TData acc          = 0.0;
            for (unsigned f = fac; f < nface; ++f)
            {
                // input slabs are face-local F(p,r) of size nm0*nm2
                acc += in[(f - fac) * nm0 * nm2 + r * nm0 + p] *
                       nbasis[q * tstride + f];
            }
            out[ocnt] += acc;
        }
    }
    localBarrier(threadBlock);
}

/**
 * @brief Leaf kernel, stage two in three dimensions: lift the results of
 * a direction-2 normal face into the volume field.
 *
 * The one-block-per-element twin of AddFaceN2ToVolKernel. The direction-2 twin
 * of AddFaceN0ToVolTOPKernel. Threads own the output modes and accumulate the
 * contributions of every face of the group in a register, so the `+=` onto @p
 * out never races. Applies \f$\Lambda_{pqr} \mathrel{+}= h_r(\pm 1)\,
 * F_{pq}\f$, the volume field being indexed `r * nm0 * nm1 + q * nm0 + p`.
 * Under @p END_PTS_COLLOCATED2 the interpolation is a delta and the face lands
 * entirely in the plane `r = f * (nm2 - 1)`, @em f being the position of the
 * face within its direction rather than its index within this call.
 *
 * The general arm indexes @p nbasis with the stride passed in
 * explicitly, so a direction carrying a single trace reads its
 * one-column table correctly.
 *
 * @tparam END_PTS_COLLOCATED2  Direction-2 volume rule contains the
 *                            domain endpoints.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   fac       Position of the first face within direction 2.
 * @param   nface     One past the position of the last face.
 * @param   tstride   Row stride of @p nbasis, that is the number of
 *                    traces direction 2 has (one for everything but a
 * hexahedron).
 * @param   nm0,nm1,nm2   Volume quadrature points per direction.
 * @param   nbasis    eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   in        Trace-mode workspace written by
 *                    IPWRTPhysFaceTOPKernel, `nm0 * nm1` entries per face.
 * @param   out       Volume field of this element, accumulated onto.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED2, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddFaceN2ToVolTOPKernel(
    const unsigned fac, const unsigned nface, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const unsigned nm2,
    const TData *NEK_RESTRICT nbasis, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (END_PTS_COLLOCATED2)
    {
        const unsigned total = (nface - fac) * nm1 * nm0;
        for (unsigned cnt = idx0; cnt < total; cnt += stride)
        {
            const unsigned f   = fac + cnt / (nm1 * nm0);
            const unsigned rem = cnt % (nm1 * nm0);
            out[f * nm0 * nm1 * (nm2 - 1) + rem] += in[cnt];
        }
    }
    else
    {
        const unsigned total = nm2 * nm1 * nm0;
        for (unsigned ocnt = idx0; ocnt < total; ocnt += stride)
        {
            const unsigned r   = ocnt / (nm1 * nm0);
            const unsigned rem = ocnt % (nm1 * nm0);
            TData acc          = 0.0;
            for (unsigned f = fac; f < nface; ++f)
            {
                acc +=
                    in[(f - fac) * nm0 * nm1 + rem] * nbasis[r * tstride + f];
            }
            out[ocnt] += acc;
        }
    }
    localBarrier(threadBlock);
}

/**
 * @brief Glue for a direction-0 normal face: integrate, then lift.
 *
 * The one-block-per-element twin of IPWRTPhysTraceFaceN0Kernel. Routes
 * the direction's tables and counts into the two leaf stages, which
 * carry their own barriers. The in-face directions of an \f$N_0\f$
 * face are 1 and 2, so the face integration is asked for `(nm1, nm2)` entries
 * per face and the lift spreads them along direction 0.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   fac         Position of the first face within direction 0.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of @p nbasis (traces in direction 0).
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis      eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   tnq0,tnq1   Trace quadrature points of the two in-face
 *                      directions.
 * @param   tbasis0,tbasis1   eInterp tables to those trace points.
 * @param   tw0,tw1     Trace quadrature weights of the in-face
 *                      directions.
 * @param   tjac        Trace Jacobian of this block's element.
 * @param   wsp         Trace-mode workspace in shared memory, `(nm1, nm2)`
 *                      entries per face covered.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of the faces covered.
 * @param   out         Volume field of this element, accumulated onto.
 * @param   isCollocated0    First in-face direction is collocated.
 * @param   isCollocated1    Second in-face direction is collocated.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED0, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceFaceN0TOPKernel(
    const unsigned fac, const unsigned nface, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const unsigned nm2,
    const TData *NEK_RESTRICT nbasis, const unsigned tnq0, const unsigned tnq1,
    const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
    const TData *NEK_RESTRICT tw0, const TData *NEK_RESTRICT tw1,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    TData *NEK_RESTRICT wsp1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated0, const bool isCollocated1,
    const TthreadBlock &threadBlock)
{
    IPWRTPhysFaceTOPKernel<DEFORMED>(fac, nface, nm1, nm2, tnq0, tnq1, tbasis0,
                                     tbasis1, tw0, tw1, tjac, wsp1, in, wsp,
                                     isCollocated0, isCollocated1, threadBlock);
    AddFaceN0ToVolTOPKernel<END_PTS_COLLOCATED0>(
        fac, nface, tstride, nm0, nm1, nm2, nbasis, wsp, out, threadBlock);
}

/**
 * @brief Glue for a direction-1 normal face: integrate, then lift.
 *
 * The one-block-per-element twin of IPWRTPhysTraceFaceN1Kernel. Routes
 * the direction's tables and counts into the two leaf stages, which
 * carry their own barriers. The in-face directions of an \f$N_1\f$
 * face are 0 and 2, so the face integration is asked for `(nm0, nm2)` entries
 * per face and the lift spreads them along direction 1.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   fac         Position of the first face within direction 1.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of @p nbasis (traces in direction 1).
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis      eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   tnq0,tnq1   Trace quadrature points of the two in-face
 *                      directions.
 * @param   tbasis0,tbasis1   eInterp tables to those trace points.
 * @param   tw0,tw1     Trace quadrature weights of the in-face
 *                      directions.
 * @param   tjac        Trace Jacobian of this block's element.
 * @param   wsp         Trace-mode workspace in shared memory, `(nm0, nm2)`
 *                      entries per face covered.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of the faces covered.
 * @param   out         Volume field of this element, accumulated onto.
 * @param   isCollocated0    First in-face direction is collocated.
 * @param   isCollocated1    Second in-face direction is collocated.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED1, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceFaceN1TOPKernel(
    const unsigned fac, const unsigned nface, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const unsigned nm2,
    const TData *NEK_RESTRICT nbasis, const unsigned tnq0, const unsigned tnq1,
    const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
    const TData *NEK_RESTRICT tw0, const TData *NEK_RESTRICT tw1,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    TData *NEK_RESTRICT wsp1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated0, const bool isCollocated1,
    const TthreadBlock &threadBlock)
{
    IPWRTPhysFaceTOPKernel<DEFORMED>(fac, nface, nm0, nm2, tnq0, tnq1, tbasis0,
                                     tbasis1, tw0, tw1, tjac, wsp1, in, wsp,
                                     isCollocated0, isCollocated1, threadBlock);
    AddFaceN1ToVolTOPKernel<END_PTS_COLLOCATED1>(
        fac, nface, tstride, nm0, nm1, nm2, nbasis, wsp, out, threadBlock);
}

/**
 * @brief Glue for a direction-2 normal face: integrate, then lift.
 *
 * The one-block-per-element twin of IPWRTPhysTraceFaceN2Kernel. Routes
 * the direction's tables and counts into the two leaf stages, which
 * carry their own barriers. The in-face directions of an \f$N_2\f$
 * face are 0 and 1, so the face integration is asked for `(nm0, nm1)` entries
 * per face and the lift spreads them along direction 2.
 *
 * @tparam END_PTS_COLLOCATED2  Direction-2 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TthreadBlock       Thread-block handle of the backend.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   fac         Position of the first face within direction 2.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of @p nbasis (traces in direction 2).
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis      eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   tnq0,tnq1   Trace quadrature points of the two in-face
 *                      directions.
 * @param   tbasis0,tbasis1   eInterp tables to those trace points.
 * @param   tw0,tw1     Trace quadrature weights of the in-face
 *                      directions.
 * @param   tjac        Trace Jacobian of this block's element.
 * @param   wsp         Trace-mode workspace in shared memory, `(nm0, nm1)`
 *                      entries per face covered.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of the faces covered.
 * @param   out         Volume field of this element, accumulated onto.
 * @param   isCollocated0    First in-face direction is collocated.
 * @param   isCollocated1    Second in-face direction is collocated.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool END_PTS_COLLOCATED2, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceFaceN2TOPKernel(
    const unsigned fac, const unsigned nface, const unsigned tstride,
    const unsigned nm0, const unsigned nm1, const unsigned nm2,
    const TData *NEK_RESTRICT nbasis, const unsigned tnq0, const unsigned tnq1,
    const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
    const TData *NEK_RESTRICT tw0, const TData *NEK_RESTRICT tw1,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    TData *NEK_RESTRICT wsp1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated0, const bool isCollocated1,
    const TthreadBlock &threadBlock)
{
    IPWRTPhysFaceTOPKernel<DEFORMED>(fac, nface, nm0, nm1, tnq0, tnq1, tbasis0,
                                     tbasis1, tw0, tw1, tjac, wsp1, in, wsp,
                                     isCollocated0, isCollocated1, threadBlock);
    AddFaceN2ToVolTOPKernel<END_PTS_COLLOCATED2>(
        fac, nface, tstride, nm0, nm1, nm2, nbasis, wsp, out, threadBlock);
}

/**
 * @brief One named face of one three-dimensional element: turn the face
 * id into a direction and a position, then integrate and lift.
 *
 * The one-block-per-element twin of IProductWRTPhysTraceFaceKernel: the
 * single-trace worker, used when the solver drives one trace at a time,
 * and also the per-face step of the bulk wrapper. @p in and @p tjac
 * already point at that face's block, so the position within the
 * direction is expressed through the [@p fac, @p nface) window handed to
 * the glue kernels.
 *
 * The face to (direction, position) map is `4->(0,0)`, `2->(0,1)`,
 * `1->(1,0)`, `3->(1,1)`, `0->(2,0)` and, for the hexahedron alone,
 * `5->(2,1)`. A tetrahedron differs in one place: its face 3 is the
 * lower direction-0 face, `3->(0,0)`, which is why case 3 branches on
 * the shape. It serves every three-dimensional shape.
 *
 * The @c nbasis row strides are derived from the shape rather than
 * assumed: two in direction 0 always, two in direction 1 except for a
 * tetrahedron, and two in direction 2 only for a hexahedron.
 *
 * @note An unrecognised @p face is ignored silently: device code cannot
 * raise a host error.
 *
 * @tparam DEFORMED      Trace Jacobian varies point by point.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   face        Nektar face id of the trace to apply.
 * @param   shape       Hex, Prism, Pyr, Tet, NodalTet or NodalPrism.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis0,nbasis1,nbasis2   eInterp tables \f$h_p(\pm 1)\f$
 *                      per direction.
 * @param   tnq00,tnq01,tnq10,tnq11,tnq20,tnq21   Trace quadrature
 *                      points, per normal and tangential direction.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points.
 * @param   tw00,tw01,tw10,tw11,tw20,tw21   Trace quadrature weights.
 * @param   tjac        Trace Jacobian of this block's element, already
 *                      offset to this face.
 * @param   wsp         Trace-mode workspace of one face, in shared
 *                      memory.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values, already offset to this face.
 * @param   out         Volume field of this element.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2   Volume
 *                      rule of that direction contains the domain
 *                      endpoints.
 * @param   append      Accumulate onto @p out instead of zeroing it
 *                      first; the bulk wrapper passes true, having
 *                      zeroed the field itself.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTPhysTraceFaceTOPKernel(
    const unsigned face, const LibUtilities::ShapeType shape,
    const unsigned nm0, const unsigned nm1, const unsigned nm2,
    const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
    const TData *NEK_RESTRICT nbasis2, const unsigned tnq00,
    const unsigned tnq01, const unsigned tnq10, const unsigned tnq11,
    const unsigned tnq20, const unsigned tnq21,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
    const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    TData *NEK_RESTRICT wsp1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated00,
    const bool isCollocated01, const bool isCollocated10,
    const bool isCollocated11, const bool isCollocated20,
    const bool isCollocated21, bool endPtsCollocated0, bool endPtsCollocated1,
    bool endPtsCollocated2, bool append, const TthreadBlock &threadBlock)
{
    // per-direction trace counts: the nbasis eInterp tables are stored
    // tightly with this stride (see BasisDataWarehouse eInterp)
    const bool tsIsTet =
        (shape == LibUtilities::Tet || shape == LibUtilities::NodalTet);
    const unsigned ts0 = 2u;
    const unsigned ts1 = tsIsTet ? 1u : 2u;
    const unsigned ts2 = (shape == LibUtilities::Hex) ? 2u : 1u;

    if (!append)
    {
        ZeroModesTOPKernel(nm0 * nm1 * nm2, out, threadBlock);
    }

    switch (face)
    {
        case 4:
            if (endPtsCollocated0)
            {
                IPWRTPhysTraceFaceN0TOPKernel<true, DEFORMED>(
                    0, 1, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                    tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01, threadBlock);
            }
            else
            {
                IPWRTPhysTraceFaceN0TOPKernel<false, DEFORMED>(
                    0, 1, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                    tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01, threadBlock);
            }
            break;
        case 2:
            if (endPtsCollocated0)
            {
                IPWRTPhysTraceFaceN0TOPKernel<true, DEFORMED>(
                    1, 2, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                    tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01, threadBlock);
            }
            else
            {
                IPWRTPhysTraceFaceN0TOPKernel<false, DEFORMED>(
                    1, 2, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                    tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01, threadBlock);
            }
            break;
        case 1:
            if (endPtsCollocated1)
            {
                IPWRTPhysTraceFaceN1TOPKernel<true, DEFORMED>(
                    0, 1, ts1, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10,
                    tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11, threadBlock);
            }
            else
            {
                IPWRTPhysTraceFaceN1TOPKernel<false, DEFORMED>(
                    0, 1, ts1, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10,
                    tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11, threadBlock);
            }
            break;
        case 3:
            if (tsIsTet)
            {
                if (endPtsCollocated0)
                {
                    IPWRTPhysTraceFaceN0TOPKernel<true, DEFORMED>(
                        0, 1, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01,
                        tbasis00, tbasis01, tw00, tw01, tjac, wsp, wsp1, in,
                        out, isCollocated00, isCollocated01, threadBlock);
                }
                else
                {
                    IPWRTPhysTraceFaceN0TOPKernel<false, DEFORMED>(
                        0, 1, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01,
                        tbasis00, tbasis01, tw00, tw01, tjac, wsp, wsp1, in,
                        out, isCollocated00, isCollocated01, threadBlock);
                }
            }
            else if (endPtsCollocated1)
            {
                IPWRTPhysTraceFaceN1TOPKernel<true, DEFORMED>(
                    1, 2, ts1, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10,
                    tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11, threadBlock);
            }
            else
            {
                IPWRTPhysTraceFaceN1TOPKernel<false, DEFORMED>(
                    1, 2, ts1, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10,
                    tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11, threadBlock);
            }
            break;
        case 0:
            if (endPtsCollocated2)
            {
                IPWRTPhysTraceFaceN2TOPKernel<true, DEFORMED>(
                    0, 1, ts2, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20,
                    tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21, threadBlock);
            }
            else
            {
                IPWRTPhysTraceFaceN2TOPKernel<false, DEFORMED>(
                    0, 1, ts2, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20,
                    tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21, threadBlock);
            }
            break;
        case 5:
            if (endPtsCollocated2)
            {
                IPWRTPhysTraceFaceN2TOPKernel<true, DEFORMED>(
                    1, 2, ts2, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20,
                    tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21, threadBlock);
            }
            else
            {
                IPWRTPhysTraceFaceN2TOPKernel<false, DEFORMED>(
                    1, 2, ts2, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20,
                    tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21, threadBlock);
            }
            break;
        default:
            // Unrecognised face input (cannot raise host error in device code)
            break;
    }
}

/**
 * @brief All faces of one three-dimensional element: zero the volume
 * field, then walk the packed traces one face at a time.
 *
 * The bulk wrapper, and the one-block-per-element twin of
 * IProductWRTPhysTrace3DFaceKernel. The face order is the packed trace
 * order, hex `{4,2,1,3,0,5}`, prism and pyramid `{4,2,1,3,0}` and
 * tetrahedron `{3,2,1,0}`, with the matching normal directions; the
 * input pointer advances by that face's `tnq * tnq` values and the
 * Jacobian pointer by the same number of points when deformed or by one
 * face slot when regular, which is exactly the regular-Jacobian slot
 * order.
 *
 * Each face is applied through the single-face worker with
 * `append = true`, this wrapper having done the zeroing itself. Faces
 * are processed one at a time rather than a pair at a time so that the
 * trace-mode workspace stays within its budget: batching a pair would
 * spill the second face's entries into the scratch its own general
 * integration is using. Pair batching is a possible follow-up with a
 * doubled workspace.
 *
 * @tparam SHAPE_TYPE    Hex, Prism, Pyr, Tet, NodalTet or NodalPrism.
 * @tparam DEFORMED      Trace Jacobian varies point by point.
 * @tparam APPEND        Accumulate onto @p out instead of zeroing it
 *                       first.
 * @tparam TthreadBlock  Thread-block handle of the backend.
 * @tparam TData         Floating-point type of the field data.
 *
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis0,nbasis1,nbasis2   eInterp tables \f$h_p(\pm 1)\f$
 *                      per direction.
 * @param   tnq00,tnq01 Trace quadrature points of an \f$N_0\f$ face.
 * @param   tnq10,tnq11 Trace quadrature points of an \f$N_1\f$ face.
 * @param   tnq20,tnq21 Trace quadrature points of an \f$N_2\f$ face.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points, per normal
 *                      direction and tangential direction.
 * @param   tw00,tw01,tw10,tw11,tw20,tw21   Trace quadrature weights in
 *                      the same order.
 * @param   tjac        Trace Jacobian of this block's element.
 * @param   wsp         Trace-mode workspace of one face, in shared
 *                      memory.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of this element.
 * @param   out         Volume field of this element.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2   Volume
 *                      rule of that direction contains the domain
 *                      endpoints.
 * @param   threadBlock Thread block cooperating on this element.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool APPEND,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void IProductWRTPhysTrace3DFaceTOPKernel(
    const unsigned nm0, const unsigned nm1, const unsigned nm2,
    const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
    const TData *NEK_RESTRICT nbasis2, const unsigned tnq00,
    const unsigned tnq01, const unsigned tnq10, const unsigned tnq11,
    const unsigned tnq20, const unsigned tnq21,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
    const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    TData *NEK_RESTRICT wsp1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated00,
    const bool isCollocated01, const bool isCollocated10,
    const bool isCollocated11, const bool isCollocated20,
    const bool isCollocated21, bool endPtsCollocated0, bool endPtsCollocated1,
    bool endPtsCollocated2, const TthreadBlock &threadBlock)
{
    if constexpr (!APPEND)
    {
        ZeroModesTOPKernel(nm0 * nm1 * nm2, out, threadBlock);
    }

    constexpr bool IS_TET = (SHAPE_TYPE == LibUtilities::Tet ||
                             SHAPE_TYPE == LibUtilities::NodalTet);
    constexpr unsigned nTraces =
        (SHAPE_TYPE == LibUtilities::Hex) ? 6 : (IS_TET ? 4 : 5);
    constexpr unsigned faceHex[6] = {4, 2, 1, 3, 0, 5};
    constexpr unsigned face5[5]   = {4, 2, 1, 3, 0};
    constexpr unsigned faceTet[4] = {3, 2, 1, 0};
    constexpr unsigned dirHex[6]  = {0, 0, 1, 1, 2, 2};
    constexpr unsigned dir5[5]    = {0, 0, 1, 1, 2};
    constexpr unsigned dirTet[4]  = {0, 0, 1, 2};

    const unsigned fsz[3] = {tnq00 * tnq01, tnq10 * tnq11, tnq20 * tnq21};
    unsigned inoff = 0, joff = 0;
    for (unsigned t = 0; t < nTraces; ++t)
    {
        const unsigned face = (SHAPE_TYPE == LibUtilities::Hex)
                                  ? faceHex[t]
                                  : (IS_TET ? faceTet[t] : face5[t]);
        const unsigned dir  = (SHAPE_TYPE == LibUtilities::Hex)
                                  ? dirHex[t]
                                  : (IS_TET ? dirTet[t] : dir5[t]);
        IProductWRTPhysTraceFaceTOPKernel<DEFORMED>(
            face, SHAPE_TYPE, nm0, nm1, nm2, nbasis0, nbasis1, nbasis2, tnq00,
            tnq01, tnq10, tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10,
            tbasis11, tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20, tw21,
            tjac + joff, wsp, wsp1, in + inoff, out, isCollocated00,
            isCollocated01, isCollocated10, isCollocated11, isCollocated20,
            isCollocated21, endPtsCollocated0, endPtsCollocated1,
            endPtsCollocated2, true, threadBlock);
        inoff += fsz[dir];
        joff += DEFORMED ? fsz[dir] : 1;
    }
}

/**
 * @brief Bulk path, one dimension: lift both end vertices of every
 * segment of the block into its volume modes.
 *
 * @details
 * The TOP counterpart of the one-dimensional
 * IProductWRTPhysTraceKernel. One block takes one element at a time
 * from the grid-stride loop and its threads share the mode loop.
 *
 * Element @em e of component @em c starts at
 * `numDataIn * (nelmt * c + e)` in the packed trace input and at
 * `numDataOut * (nelmt * c + e)` in the volume output: the width-one
 * specialisation of the SumFac kernels' addressing.
 *
 * A segment's traces are points and carry no surface measure, so the
 * trace Jacobian and the workspace go unread here, as they do on the
 * SumFac route.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TTraceSizeParameter1D, typename TData,
          typename TthreadBlock,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter1D>()))
    IProductWRTPhysTraceKernelLauncher(
        const TTraceSizeParameter1D sizeParam1D,
        const TData *NEK_RESTRICT nbasis0, const size_t nelmt,
        const unsigned int numDataIn, const unsigned int numDataOut,
        [[maybe_unused]] const unsigned int numDataJac,
        [[maybe_unused]] const TData *NEK_RESTRICT tjac,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, const bool endPtsCollocated, const bool append,
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
        const TData *inptr = in + numDataIn * (nelmt * c + e);
        TData *outptr      = out + numDataOut * (nelmt * c + e);

        for (unsigned int p = idx0; p < nm0; p += stride)
        {
            TData acc = append ? outptr[p] : TData(0.0);

            if (endPtsCollocated)
            {
                if (p == 0u)
                {
                    acc += inptr[0];
                }
                if (p == nm0 - 1u)
                {
                    acc += inptr[1];
                }
            }
            else
            {
                acc += nbasis0[2 * p] * inptr[0];
                acc += nbasis0[2 * p + 1] * inptr[1];
            }
            outptr[p] = acc;
        }

        localBarrier(threadBlock);
        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief Bulk path, two dimensions: lift every edge of every element of
 * the block into its volume modes.
 *
 * @details
 * The TOP counterpart of the two-dimensional
 * IProductWRTPhysTraceKernel, walking the normal directions through
 * IProductWRTPhysTrace2DTOPKernel.
 *
 * The trace-mode buffer lives in dynamic shared memory, the two edge
 * blocks IProductWRTPhysTraceSharedMemorySize() asks for, sized against
 * the same helper so the two cannot drift. @p wsp
 * is therefore unread.
 *
 * The accumulate flag is honoured, as it is on the SumFac route, by
 * branching to the APPEND instantiation of the core.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TTraceSizeParameter2D, typename TData,
          typename TthreadBlock,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter2D>()))
    IProductWRTPhysTraceKernelLauncher(
        const TTraceSizeParameter2D sizeParam2D,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const size_t nelmt, const unsigned int numDataIn,
        const unsigned int numDataOut, const unsigned int numDataJac,
        const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tjac,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
        TData *out, const bool isCollocated0, const bool isCollocated1,
        const bool endPtsCollocated0, const bool endPtsCollocated1,
        const bool append, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);
    TData *s_wsp = (TData *)shmemptr;

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int tnq00 = sizeParam2D.nq00();
    const unsigned int tnq10 = sizeParam2D.nq10();

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr  = in + numDataIn * (nelmt * c + e);
        TData *outptr       = out + numDataOut * (nelmt * c + e);
        const TData *jacptr = tjac + numDataJac * e;

        if (append)
        {
            IProductWRTPhysTrace2DTOPKernel<SHAPE_TYPE, DEFORMED, true>(
                nm0, nm1, nbasis0, nbasis1, tnq00, tnq10, tbasis0, tbasis1,
                tw00, tw10, jacptr, s_wsp, inptr, outptr, isCollocated0,
                isCollocated1, endPtsCollocated0, endPtsCollocated1,
                threadBlock);
        }
        else
        {
            IProductWRTPhysTrace2DTOPKernel<SHAPE_TYPE, DEFORMED, false>(
                nm0, nm1, nbasis0, nbasis1, tnq00, tnq10, tbasis0, tbasis1,
                tw00, tw10, jacptr, s_wsp, inptr, outptr, isCollocated0,
                isCollocated1, endPtsCollocated0, endPtsCollocated1,
                threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief Per-trace path, two dimensions: lift one named edge of every
 * element of the block, accumulating into the volume modes.
 *
 * The single-trace counterpart of the two-dimensional
 * IProductWRTPhysTraceKernelLauncher. The edge id is turned into a normal
 * direction and an edge slot by the same mapping the SumFac route
 * uses, and the caller has already offset @p in and @p tjac to this
 * trace, so the kernel reads them from zero.
 */
template <typename Implementation, bool DEFORMED,
          typename TTraceSizeParameter2D, typename TData, typename TthreadBlock,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter2D>()))
    IProductWRTPhysTraceTraceKernelLauncher(
        const unsigned edge, const LibUtilities::ShapeType shape,
        const TTraceSizeParameter2D sizeParam2D,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const size_t nelmt, const unsigned int numDataIn,
        const unsigned int numDataOut, const unsigned int numDataJac,
        const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tjac,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, const bool isCollocated0,
        const bool isCollocated1, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const bool append,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);
    TData *s_wsp = (TData *)shmemptr;

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int tnq00 = sizeParam2D.nq00();
    const unsigned int tnq10 = sizeParam2D.nq10();

    // Edge -> (normal dir, edge slot) mapping, identical to the
    // per-lane IProductWRTPhysTraceEdgeKernel switch.
    const bool isQuad = (shape == LibUtilities::Quad);
    unsigned nd = 0, edg = 0, nedg = 1;
    if (isQuad)
    {
        switch (edge)
        {
            case 0:
                nd   = 1;
                edg  = 0;
                nedg = 1;
                break;
            case 1:
                nd   = 0;
                edg  = 1;
                nedg = 2;
                break;
            case 2:
                nd   = 1;
                edg  = 1;
                nedg = 2;
                break;
            case 3:
                nd   = 0;
                edg  = 0;
                nedg = 1;
                break;
            default:
                nd = 3;
                break;
        }
    }
    else
    {
        switch (edge)
        {
            case 0:
                nd   = 1;
                edg  = 0;
                nedg = 1;
                break;
            case 1:
                nd   = 0;
                edg  = 1;
                nedg = 2;
                break;
            case 2:
                nd   = 0;
                edg  = 0;
                nedg = 1;
                break;
            default:
                nd = 3;
                break;
        }
    }

    if (nd > 1u)
    {
        return;
    }

    const unsigned int tstride = isQuad || nd == 0 ? 2 : 1;

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr  = in + numDataIn * (nelmt * c + e);
        TData *outptr       = out + numDataOut * (nelmt * c + e);
        const TData *jacptr = tjac + numDataJac * e;

        if (!append)
        {
            ZeroModesTOPKernel(nm0 * nm1, outptr, threadBlock);
        }

        if (nd == 0)
        {
            if (endPtsCollocated0)
            {
                IPWRTPhysTraceEdgeN0TOPKernel<true, DEFORMED>(
                    edg, nedg, tstride, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                    jacptr, s_wsp, inptr, outptr, isCollocated0, threadBlock);
            }
            else
            {
                IPWRTPhysTraceEdgeN0TOPKernel<false, DEFORMED>(
                    edg, nedg, tstride, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                    jacptr, s_wsp, inptr, outptr, isCollocated0, threadBlock);
            }
        }
        else
        {
            if (endPtsCollocated1)
            {
                IPWRTPhysTraceEdgeN1TOPKernel<true, DEFORMED>(
                    edg, nedg, tstride, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
                    jacptr, s_wsp, inptr, outptr, isCollocated1, threadBlock);
            }
            else
            {
                IPWRTPhysTraceEdgeN1TOPKernel<false, DEFORMED>(
                    edg, nedg, tstride, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
                    jacptr, s_wsp, inptr, outptr, isCollocated1, threadBlock);
            }
        }

        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief Bulk path, three dimensions: lift every face of every element
 * of the block into its volume modes.
 *
 * @details
 * The TOP counterpart of the three-dimensional
 * IProductWRTPhysTraceKernel, walking the faces in packed trace order
 * through IProductWRTPhysTrace3DFaceTOPKernel.
 *
 * Shared memory is partitioned as the SumFac route partitions its
 * per-element global workspace: the face mode block
 * IProductWRTPhysTraceFaceModeBlockSize() sizes, then the contraction
 * scratch IProductWRTPhysTraceFaceScratchSize() sizes. The host sizes
 * the dynamic allocation against the same two helpers.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TTraceSizeParameter3D, typename TData,
          typename TthreadBlock,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter3D>()))
    IProductWRTPhysTraceKernelLauncher(
        const TTraceSizeParameter3D sizeParam3D,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT nbasis2, const size_t nelmt,
        const unsigned int numDataIn, const unsigned int numDataOut,
        const unsigned int numDataJac, const TData *NEK_RESTRICT tbasis00,
        const TData *NEK_RESTRICT tbasis01, const TData *NEK_RESTRICT tbasis10,
        const TData *NEK_RESTRICT tbasis11, const TData *NEK_RESTRICT tbasis20,
        const TData *NEK_RESTRICT tbasis21, const TData *NEK_RESTRICT tw00,
        const TData *NEK_RESTRICT tw01, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tw11, const TData *NEK_RESTRICT tw20,
        const TData *NEK_RESTRICT tw21, const TData *NEK_RESTRICT tjac,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, const bool isCollocated00,
        const bool isCollocated01, const bool isCollocated10,
        const bool isCollocated11, const bool isCollocated20,
        const bool isCollocated21, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const bool endPtsCollocated2,
        const bool append, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    TData *s_wsp0 = (TData *)shmemptr;
    TData *s_wsp1 = s_wsp0 + IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int tnq00 = sizeParam3D.nq00();
    const unsigned int tnq01 = sizeParam3D.nq01();
    const unsigned int tnq10 = sizeParam3D.nq10();
    const unsigned int tnq11 = sizeParam3D.nq11();
    const unsigned int tnq20 = sizeParam3D.nq20();
    const unsigned int tnq21 = sizeParam3D.nq21();

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr  = in + numDataIn * (nelmt * c + e);
        TData *outptr       = out + numDataOut * (nelmt * c + e);
        const TData *jacptr = tjac + numDataJac * e;

        if (append)
        {
            IProductWRTPhysTrace3DFaceTOPKernel<SHAPE_TYPE, DEFORMED, true>(
                nm0, nm1, nm2, nbasis0, nbasis1, nbasis2, tnq00, tnq01, tnq10,
                tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10, tbasis11,
                tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20, tw21, jacptr,
                s_wsp0, s_wsp1, inptr, outptr, isCollocated00, isCollocated01,
                isCollocated10, isCollocated11, isCollocated20, isCollocated21,
                endPtsCollocated0, endPtsCollocated1, endPtsCollocated2,
                threadBlock);
        }
        else
        {
            IProductWRTPhysTrace3DFaceTOPKernel<SHAPE_TYPE, DEFORMED, false>(
                nm0, nm1, nm2, nbasis0, nbasis1, nbasis2, tnq00, tnq01, tnq10,
                tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10, tbasis11,
                tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20, tw21, jacptr,
                s_wsp0, s_wsp1, inptr, outptr, isCollocated00, isCollocated01,
                isCollocated10, isCollocated11, isCollocated20, isCollocated21,
                endPtsCollocated0, endPtsCollocated1, endPtsCollocated2,
                threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief Per-trace path, three dimensions: lift one named face of every
 * element of the block, accumulating into the volume modes.
 *
 * The single-trace counterpart of the three-dimensional
 * IProductWRTPhysTraceKernelLauncher. The caller has already offset @p in
 * and @p tjac to this trace, so the kernel reads them from zero, and
 * IProductWRTPhysTraceFaceTOPKernel takes the accumulate flag at run time.
 */
template <typename Implementation, bool DEFORMED,
          typename TTraceSizeParameter3D, typename TData, typename TthreadBlock,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter3D>()))
    IProductWRTPhysTraceTraceKernelLauncher(
        const unsigned face, const LibUtilities::ShapeType shape,
        const TTraceSizeParameter3D sizeParam3D,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT nbasis2, const size_t nelmt,
        const unsigned int numDataIn, const unsigned int numDataOut,
        const unsigned int numDataJac, const TData *NEK_RESTRICT tbasis00,
        const TData *NEK_RESTRICT tbasis01, const TData *NEK_RESTRICT tbasis10,
        const TData *NEK_RESTRICT tbasis11, const TData *NEK_RESTRICT tbasis20,
        const TData *NEK_RESTRICT tbasis21, const TData *NEK_RESTRICT tw00,
        const TData *NEK_RESTRICT tw01, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tw11, const TData *NEK_RESTRICT tw20,
        const TData *NEK_RESTRICT tw21, const TData *NEK_RESTRICT tjac,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, const bool isCollocated00,
        const bool isCollocated01, const bool isCollocated10,
        const bool isCollocated11, const bool isCollocated20,
        const bool isCollocated21, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const bool endPtsCollocated2,
        const bool append, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    TData *s_wsp0 = (TData *)shmemptr;
    TData *s_wsp1 = s_wsp0 + IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int tnq00 = sizeParam3D.nq00();
    const unsigned int tnq01 = sizeParam3D.nq01();
    const unsigned int tnq10 = sizeParam3D.nq10();
    const unsigned int tnq11 = sizeParam3D.nq11();
    const unsigned int tnq20 = sizeParam3D.nq20();
    const unsigned int tnq21 = sizeParam3D.nq21();

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr  = in + numDataIn * (nelmt * c + e);
        TData *outptr       = out + numDataOut * (nelmt * c + e);
        const TData *jacptr = tjac + numDataJac * e;

        IProductWRTPhysTraceFaceTOPKernel<DEFORMED>(
            face, shape, nm0, nm1, nm2, nbasis0, nbasis1, nbasis2, tnq00, tnq01,
            tnq10, tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10, tbasis11,
            tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20, tw21, jacptr,
            s_wsp0, s_wsp1, inptr, outptr, isCollocated00, isCollocated01,
            isCollocated10, isCollocated11, isCollocated20, isCollocated21,
            endPtsCollocated0, endPtsCollocated1, endPtsCollocated2, append,
            threadBlock);

        e += getBlockRange(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
