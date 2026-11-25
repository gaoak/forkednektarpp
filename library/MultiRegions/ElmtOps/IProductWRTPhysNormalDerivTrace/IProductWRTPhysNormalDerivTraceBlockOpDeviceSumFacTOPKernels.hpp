///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysNormalDerivTraceBlockOpDeviceSumFacTOPKernels.hpp
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
// Description: IProductWRTPhysNormalDerivTrace SumFacTOP device kernels: one
// thread block per element on width-1 element-contiguous storage, threads
// striding the trace points and modes.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysNormalDerivTraceBlockOpDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernels of the surface inner product against the
 * normal derivative of the volume cardinal basis, one thread block per
 * element.
 *
 * @details
 * The thread-over-points counterpart of
 * IProductWRTPhysNormalDerivTraceBlockOpDeviceSumFacKernels.hpp. Both
 * compute one trace inner product per element direction, each the plain
 * trace lift with the direction's table replaced by its derivative, and
 * apply the collapsed \f$2/(1 - \eta)\f$ factors on the volume. Here an
 * element's entries are contiguous at width one and the block's threads
 * share its point and mode loops. The edge and face cores are those of
 * IProductWRTPhysTraceBlockOpDeviceSumFacTOPKernels.hpp.
 *
 * The trace-mode buffer, the face contraction scratch, the element's
 * volume and, for a collapsed shape, the volume buffer of the collapsed
 * factor scaling all live in dynamic shared memory, after the
 * \f$2/(1 - \eta)\f$ tables. Faces go through the cores one at a time, as
 * on the plain trace lift's TOP route, so the buffers are sized for one
 * face.
 *
 * @see IProductWRTPhysNormalDerivTraceBlockOpDeviceSumFac.hpp for the host
 * dispatch that launches these under the SumFacTOP tag.
 * @see IProductWRTPhysNormalDerivTraceBlockOpDeviceSumFacKernels.hpp for the
 * warp-per-lane kernels.
 */

#pragma once

#include <MultiRegions/ElmtOps/IProductWRTPhysNormalDerivTrace/IProductWRTPhysNormalDerivTraceBlockOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysNormalDerivTrace/IProductWRTPhysNormalDerivTraceBlockOpDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceBlockOpDeviceSumFacTOPKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/// @brief Workspace of a whole block, per component: none, the
/// intermediates living in the dynamic shared memory
/// IProductWRTPhysNormalDerivTraceSharedMemorySize() sizes.
template <typename Implementation, typename TTraceSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTPhysNormalDerivTraceWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Workspace of a whole block, per component: none, the
/// intermediates living in the dynamic shared memory
/// IProductWRTPhysNormalDerivTraceSharedMemorySize() sizes.
template <typename Implementation, typename TTraceSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTPhysNormalDerivTraceWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter2D sizeParam2D)
{
    return 0;
}

/// @brief Workspace of a whole block, per component: none, the
/// intermediates living in the dynamic shared memory
/// IProductWRTPhysNormalDerivTraceSharedMemorySize() sizes.
template <typename Implementation, typename TTraceSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTPhysNormalDerivTraceWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter3D sizeParam3D)
{
    return 0;
}

/// @brief Dynamic shared memory one block needs, in elements: none, a
/// segment's traces being points.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TTraceSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTPhysNormalDerivTraceSharedMemorySize(
    [[maybe_unused]] const TTraceSizeParameter1D sizeParam1D,
    [[maybe_unused]] const unsigned int elmtsPerBlock)
{
    return 0;
}

/// @brief Dynamic shared memory one block needs, in elements: the
/// \f$2/(1 - \eta_1)\f$ table of a collapsed shape, then, for each of the
/// @p elmtsPerBlock elements of the block, an edge pair's trace-mode buffer,
/// the element's volume and, for a collapsed shape, the volume buffer of
/// the collapsed factor scaling.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TTraceSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTPhysNormalDerivTraceSharedMemorySize(
    const TTraceSizeParameter2D sizeParam2D, const unsigned int elmtsPerBlock)
{
    constexpr bool COLLAPSED = IsCollapsedDir(SHAPE_TYPE, 1);

    const unsigned int nmTot = sizeParam2D.nmTot();

    return (COLLAPSED ? sizeParam2D.nm1() : 0u) +
           elmtsPerBlock *
               (2u * IProductWRTPhysTraceEdgeModeBlockSize(sizeParam2D) +
                (COLLAPSED ? 2u : 1u) * nmTot);
}

/// @brief Dynamic shared memory one block needs, in elements: the
/// \f$2/(1 - \eta)\f$ tables of the shape's collapsed directions, then, for
/// each of the @p elmtsPerBlock elements of the block, one face's mode
/// block, the face contraction scratch, the element's volume and, for a
/// collapsed shape, the volume buffer of the collapsed factor scaling.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TTraceSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTPhysNormalDerivTraceSharedMemorySize(
    const TTraceSizeParameter3D sizeParam3D, const unsigned int elmtsPerBlock)
{
    constexpr bool COLLAPSED1 = IsCollapsedDir(SHAPE_TYPE, 1);
    constexpr bool COLLAPSED2 = IsCollapsedDir(SHAPE_TYPE, 2);

    const unsigned int nmTot = sizeParam3D.nmTot();

    return (COLLAPSED1 ? sizeParam3D.nm1() : 0u) +
           (COLLAPSED2 ? sizeParam3D.nm2() : 0u) +
           elmtsPerBlock * (IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D) +
                            IProductWRTPhysTraceFaceScratchSize(sizeParam3D) +
                            ((COLLAPSED1 || COLLAPSED2) ? 2u : 1u) * nmTot);
}

/**
 * @brief The one-dimensional lift: \f$\langle \partial_n \phi, g \rangle\f$
 * over a segment's two vertices, one thread block per element.
 *
 * The TOP counterpart of the one-dimensional SumFac launcher: the block's
 * threads share the volume points of the point evaluation
 * \f[ out_p \mathrel{+}= \sum_{v=0,1} \partial h_p(\xi_v)\, F_v\, g_v . \f]
 *
 * @tparam SHAPE_TYPE   Seg; unread.
 * @tparam Implementation   SumFacTOP.
 * @tparam DEFORMED     Unread: a segment carries one factor per vertex
 *                      whatever its geometry.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TTraceSizeParameter1D, typename TData,
          typename TthreadBlock,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter1D>()))
    IProductWRTPhysNormalDerivTraceKernelLauncher(
        const TTraceSizeParameter1D sizeParam1D,
        [[maybe_unused]] const TData *NEK_RESTRICT nbasis0,
        const TData *NEK_RESTRICT dnbasis0, const size_t nelmt,
        const unsigned int numDataIn, const unsigned int numDataOut,
        const unsigned int numDataJac,
        [[maybe_unused]] const size_t jacCompStride,
        [[maybe_unused]] const TData *NEK_RESTRICT twoOver0,
        const TData *NEK_RESTRICT jac, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        const bool append, [[maybe_unused]] const bool endPtsCollocated0,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter1D or "
                  "TemplatedTraceSizeParameter1D.");

    const unsigned int nm0 = sizeParam1D.nm0();

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr  = in + numDataIn * (nelmt * c + e);
        TData *outptr       = out + numDataOut * (nelmt * c + e);
        const TData *jacptr = jac + numDataJac * e;

        const TData fg0 = jacptr[0] * inptr[0];
        const TData fg1 = jacptr[1] * inptr[1];

        for (unsigned int p = idx0; p < nm0; p += stride)
        {
            TData acc = append ? outptr[p] : TData(0.0);
            acc += dnbasis0[2 * p] * fg0;
            acc += dnbasis0[2 * p + 1] * fg1;
            outptr[p] = acc;
        }

        localBarrier(threadBlock);
        e += getBlockRange(threadBlock);
    }
}

/// @brief Scale a volume buffer by the collapsed factors a trace group
/// still needs and accumulate it into the element's volume, threads
/// striding the points. In two dimensions @p nm2 is one and @p twoOver2 is
/// unread.
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ScaleAddCollapsedFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const bool useD1, const bool useD2, const TData *NEK_RESTRICT twoOver1,
    const TData *NEK_RESTRICT twoOver2, const TData *NEK_RESTRICT volwsp,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);
    const unsigned int nm01   = nm0 * nm1;

    for (unsigned int idx = idx0; idx < nm01 * nm2; idx += stride)
    {
        const unsigned int k = idx / nm01;
        const unsigned int j = (idx % nm01) / nm0;

        TData f = 1.0;
        if (useD1)
        {
            f *= twoOver1[j];
        }
        if (useD2)
        {
            f *= twoOver2[k];
        }

        out[idx] += volwsp[idx] * f;
    }

    localBarrier(threadBlock);
}

/**
 * @brief All edges of one two-dimensional element: the two terms of the
 * operator, one per element direction, accumulated in turn; one thread
 * block per element.
 *
 * The TOP counterpart of the two-dimensional SumFac launcher, with the
 * same choice of derivative table per term and the same treatment of the
 * collapsed factor: the \f$\eta_0\f$ term's direction-0 edge group of a
 * collapsed shape goes through the volume buffer and is scaled there. The
 * element's volume is accumulated in shared memory and written out once.
 *
 * The parameters are those of the SumFac launcher; @p wsp is unread, the
 * intermediates living in @p shmemptr.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TTraceSizeParameter2D, typename TData,
          typename TthreadBlock,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter2D>()))
    IProductWRTPhysNormalDerivTraceKernelLauncher(
        const TTraceSizeParameter2D sizeParam2D,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT dnbasis0, const TData *NEK_RESTRICT dnbasis1,
        const size_t nelmt, const unsigned int numDataIn,
        const unsigned int numDataOut, const unsigned int numDataJac,
        const size_t jacCompStride, const TData *NEK_RESTRICT tbasis0,
        const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT dtbasis0,
        const TData *NEK_RESTRICT dtbasis1, const TData *NEK_RESTRICT tw0,
        const TData *NEK_RESTRICT tw1,
        [[maybe_unused]] const TData *NEK_RESTRICT twoOver0,
        [[maybe_unused]] const TData *NEK_RESTRICT twoOver1,
        const TData *NEK_RESTRICT jac, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        const unsigned int traceDir0, const unsigned int traceDir1,
        const bool append, const bool isColl0, const bool isColl1,
        const bool endPtsColl0, const bool endPtsColl1, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = nm0 * nm1;
    const unsigned int tnq00 = sizeParam2D.nq00();
    const unsigned int tnq10 = sizeParam2D.nq10();

    constexpr unsigned int nedgeN1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr bool COLLAPSED = IsCollapsedDir(SHAPE_TYPE, 1);

    // Shared memory: the factor table, then the edge pair's trace-mode
    // buffer, the element's volume and the volume buffer.
    TData *s_twoOver1 = (TData *)shmemptr;
    TData *s_wsp      = s_twoOver1 + (COLLAPSED ? nm1 : 0u);
    TData *s_out =
        s_wsp + 2u * IProductWRTPhysTraceEdgeModeBlockSize(sizeParam2D);
    TData *s_vol = s_out + nmTot;

    // Pre-compute factor.
    if constexpr (COLLAPSED)
    {
        // Copy to shared memory.
        LoadToSharedMemory(nm1, twoOver1, s_twoOver1, threadBlock);

        localBarrier(threadBlock);
    }

    // Stepping past the two N0 edges, at width one.
    const unsigned int off1  = 2 * tnq00;
    const unsigned int joff1 = (DEFORMED) ? off1 : 2;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + numDataIn * (nelmt * c + e);
        TData *outptr      = out + numDataOut * (nelmt * c + e);

        // Accumulate the element's volume in shared memory and write it
        // out once, coalesced.
        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            s_out[idx] = append ? outptr[idx] : TData(0);
        }
        localBarrier(threadBlock);

        for (unsigned int d = 0; d < 2; ++d)
        {
            // Only a collapsed shape carries the factor, and only on its
            // eta_0 term.
            const bool useD1 = CompNeedsCollapsedFac(SHAPE_TYPE, d, 1);

            // Components of the factor array sit one whole component apart.
            const TData *jacptr = jac + d * jacCompStride + numDataJac * e;

            // Term d picks the derivative table wherever a slot's element
            // direction is d. A derivative slot is never collocated, and a
            // direction never has its own end points collocated for its own
            // term.

            // Group 0: edges normal to direction 0. Keeps the factor.
            {
                TData *dst = useD1 ? s_vol : s_out;
                if (useD1)
                {
                    ZeroModesTOPKernel(nmTot, s_vol, threadBlock);
                }

                const TData *nb = (0 == d) ? dnbasis0 : nbasis0;
                const TData *tb = (traceDir0 == d) ? dtbasis0 : tbasis0;
                const bool coll = (traceDir0 == d) ? false : isColl0;
                const bool endp = (0 == d) ? false : endPtsColl0;

                if (endp)
                {
                    IPWRTPhysTraceEdgeN0TOPKernel<true, DEFORMED>(
                        0, 2, 2, nm0, nm1, nb, tnq00, tb, tw0, jacptr, s_wsp,
                        inptr, dst, coll, threadBlock);
                }
                else
                {
                    IPWRTPhysTraceEdgeN0TOPKernel<false, DEFORMED>(
                        0, 2, 2, nm0, nm1, nb, tnq00, tb, tw0, jacptr, s_wsp,
                        inptr, dst, coll, threadBlock);
                }

                if (useD1)
                {
                    ScaleAddCollapsedFacTOPKernel(nm0, nm1, 1u, true, false,
                                                  s_twoOver1, s_twoOver1, s_vol,
                                                  s_out, threadBlock);
                }
            }

            // Group 1: edges normal to direction 1. Sits at eta_1 = -1, so it
            // drops the direction 1 factor.
            {
                const TData *nb = (1 == d) ? dnbasis1 : nbasis1;
                const TData *tb = (traceDir1 == d) ? dtbasis1 : tbasis1;
                const bool coll = (traceDir1 == d) ? false : isColl1;
                const bool endp = (1 == d) ? false : endPtsColl1;

                if (endp)
                {
                    IPWRTPhysTraceEdgeN1TOPKernel<true, DEFORMED>(
                        0, nedgeN1, nedgeN1, nm0, nm1, nb, tnq10, tb, tw1,
                        jacptr + joff1, s_wsp, inptr + off1, s_out, coll,
                        threadBlock);
                }
                else
                {
                    IPWRTPhysTraceEdgeN1TOPKernel<false, DEFORMED>(
                        0, nedgeN1, nedgeN1, nm0, nm1, nb, tnq10, tb, tw1,
                        jacptr + joff1, s_wsp, inptr + off1, s_out, coll,
                        threadBlock);
                }
            }
        }

        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            outptr[idx] = s_out[idx];
        }

        e += getBlockRange(threadBlock);
    }
}

/**
 * @brief All faces of one three-dimensional element: the three terms of
 * the operator, one per element direction, accumulated in turn; one thread
 * block per element.
 *
 * The TOP counterpart of the three-dimensional SumFac launcher, with the
 * same choice of derivative table per term and the same treatment of the
 * collapsed factors: group @em k drops the direction @em k factor, and the
 * groups that keep one go through the volume buffer and are scaled there.
 * Each group's faces go through the face cores one at a time, @p in and the
 * factors offset to the face and the face's position within its direction
 * passed as the window, so that the trace-mode buffer and the contraction
 * scratch hold one face. The element's volume is accumulated in shared
 * memory and written out once.
 *
 * The parameters are those of the SumFac launcher; @p wsp is unread, the
 * intermediates living in @p shmemptr.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TTraceSizeParameter3D, typename TData,
          typename TthreadBlock,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter3D>()))
    IProductWRTPhysNormalDerivTraceKernelLauncher(
        const TTraceSizeParameter3D sizeParam3D,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT nbasis2, const TData *NEK_RESTRICT dnbasis0,
        const TData *NEK_RESTRICT dnbasis1, const TData *NEK_RESTRICT dnbasis2,
        const size_t nelmt, const unsigned int numDataIn,
        const unsigned int numDataOut, const unsigned int numDataJac,
        const size_t jacCompStride, const TData *NEK_RESTRICT tbasis00,
        const TData *NEK_RESTRICT tbasis01, const TData *NEK_RESTRICT tbasis10,
        const TData *NEK_RESTRICT tbasis11, const TData *NEK_RESTRICT tbasis20,
        const TData *NEK_RESTRICT tbasis21, const TData *NEK_RESTRICT dtbasis00,
        const TData *NEK_RESTRICT dtbasis01,
        const TData *NEK_RESTRICT dtbasis10,
        const TData *NEK_RESTRICT dtbasis11,
        const TData *NEK_RESTRICT dtbasis20,
        const TData *NEK_RESTRICT dtbasis21, const TData *NEK_RESTRICT tw00,
        const TData *NEK_RESTRICT tw01, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tw11, const TData *NEK_RESTRICT tw20,
        const TData *NEK_RESTRICT tw21,
        [[maybe_unused]] const TData *NEK_RESTRICT twoOver0,
        [[maybe_unused]] const TData *NEK_RESTRICT twoOver1,
        [[maybe_unused]] const TData *NEK_RESTRICT twoOver2,
        const TData *NEK_RESTRICT jac, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        const unsigned int traceDir00, const unsigned int traceDir01,
        const unsigned int traceDir10, const unsigned int traceDir11,
        const unsigned int traceDir20, const unsigned int traceDir21,
        const bool append, const bool isColl00, const bool isColl01,
        const bool isColl10, const bool isColl11, const bool isColl20,
        const bool isColl21, const bool endPtsColl0, const bool endPtsColl1,
        const bool endPtsColl2, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = nm0 * nm1 * nm2;
    const unsigned int tnq00 = sizeParam3D.nq00();
    const unsigned int tnq01 = sizeParam3D.nq01();
    const unsigned int tnq10 = sizeParam3D.nq10();
    const unsigned int tnq11 = sizeParam3D.nq11();
    const unsigned int tnq20 = sizeParam3D.nq20();
    const unsigned int tnq21 = sizeParam3D.nq21();

    constexpr unsigned int nfaceN1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned int nfaceN2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];
    constexpr bool COLLAPSED1 = IsCollapsedDir(SHAPE_TYPE, 1);
    constexpr bool COLLAPSED2 = IsCollapsedDir(SHAPE_TYPE, 2);

    // Shared memory: the factor tables, then one face's mode block, the
    // contraction scratch, the element's volume and the volume buffer.
    TData *s_twoOver1 = (TData *)shmemptr;
    TData *s_twoOver2 = s_twoOver1 + (COLLAPSED1 ? nm1 : 0u);
    TData *s_wsp0     = s_twoOver2 + (COLLAPSED2 ? nm2 : 0u);
    TData *s_wsp1 = s_wsp0 + IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D);
    TData *s_out  = s_wsp1 + IProductWRTPhysTraceFaceScratchSize(sizeParam3D);
    TData *s_vol  = s_out + nmTot;

    // Pre-compute factor.
    if constexpr (COLLAPSED1)
    {
        // Copy to shared memory.
        LoadToSharedMemory(nm1, twoOver1, s_twoOver1, threadBlock);
    }
    if constexpr (COLLAPSED2)
    {
        // Copy to shared memory.
        LoadToSharedMemory(nm2, twoOver2, s_twoOver2, threadBlock);
    }
    if constexpr (COLLAPSED1 || COLLAPSED2)
    {
        localBarrier(threadBlock);
    }

    // Face sizes and face group offsets into the input and the factors, at
    // width one.
    const unsigned int fsz0 = tnq00 * tnq01;
    const unsigned int fsz1 = tnq10 * tnq11;
    const unsigned int fsz2 = tnq20 * tnq21;
    const unsigned int in1  = 2 * fsz0;
    const unsigned int jac1 = DEFORMED ? in1 : 2;
    const unsigned int in2  = in1 + nfaceN1 * fsz1;
    const unsigned int jac2 = jac1 + (DEFORMED ? nfaceN1 * fsz1 : nfaceN1);

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    size_t e             = getBlockIdx(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + numDataIn * (nelmt * c + e);
        TData *outptr      = out + numDataOut * (nelmt * c + e);

        // Accumulate the element's volume in shared memory and write it
        // out once, coalesced.
        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            s_out[idx] = append ? outptr[idx] : TData(0);
        }
        localBarrier(threadBlock);

        for (unsigned int d = 0; d < 3; ++d)
        {
            const bool c1 = CompNeedsCollapsedFac(SHAPE_TYPE, d, 1);
            const bool c2 = CompNeedsCollapsedFac(SHAPE_TYPE, d, 2);

            const TData *jacptr = jac + d * jacCompStride + numDataJac * e;

            // Term d picks the derivative table wherever a slot's element
            // direction is d. A derivative slot is never collocated, and a
            // direction never has its own end points collocated for its own
            // term.
            const TData *nbd0 = (0 == d) ? dnbasis0 : nbasis0;
            const TData *nbd1 = (1 == d) ? dnbasis1 : nbasis1;
            const TData *nbd2 = (2 == d) ? dnbasis2 : nbasis2;

            const TData *tbd0 = (traceDir00 == d) ? dtbasis00 : tbasis00;
            const TData *tbd1 = (traceDir01 == d) ? dtbasis01 : tbasis01;
            const TData *tbd2 = (traceDir10 == d) ? dtbasis10 : tbasis10;
            const TData *tbd3 = (traceDir11 == d) ? dtbasis11 : tbasis11;
            const TData *tbd4 = (traceDir20 == d) ? dtbasis20 : tbasis20;
            const TData *tbd5 = (traceDir21 == d) ? dtbasis21 : tbasis21;

            const bool cd0 = (traceDir00 == d) ? false : isColl00;
            const bool cd1 = (traceDir01 == d) ? false : isColl01;
            const bool cd2 = (traceDir10 == d) ? false : isColl10;
            const bool cd3 = (traceDir11 == d) ? false : isColl11;
            const bool cd4 = (traceDir20 == d) ? false : isColl20;
            const bool cd5 = (traceDir21 == d) ? false : isColl21;

            const bool epd0 = (0 == d) ? false : endPtsColl0;
            const bool epd1 = (1 == d) ? false : endPtsColl1;
            const bool epd2 = (2 == d) ? false : endPtsColl2;

            // Group 0: faces normal to direction 0, keeps both factors.
            {
                const bool scale = c1 || c2;
                TData *dst       = scale ? s_vol : s_out;
                if (scale)
                {
                    ZeroModesTOPKernel(nmTot, s_vol, threadBlock);
                }

                for (unsigned int f = 0; f < 2; ++f)
                {
                    const TData *fin  = inptr + f * fsz0;
                    const TData *fjac = jacptr + (DEFORMED ? f * fsz0 : f);

                    if (epd0)
                    {
                        IPWRTPhysTraceFaceN0TOPKernel<true, DEFORMED>(
                            f, f + 1, 2, nm0, nm1, nm2, nbd0, tnq00, tnq01,
                            tbd0, tbd1, tw00, tw01, fjac, s_wsp0, s_wsp1, fin,
                            dst, cd0, cd1, threadBlock);
                    }
                    else
                    {
                        IPWRTPhysTraceFaceN0TOPKernel<false, DEFORMED>(
                            f, f + 1, 2, nm0, nm1, nm2, nbd0, tnq00, tnq01,
                            tbd0, tbd1, tw00, tw01, fjac, s_wsp0, s_wsp1, fin,
                            dst, cd0, cd1, threadBlock);
                    }
                }

                if (scale)
                {
                    ScaleAddCollapsedFacTOPKernel(nm0, nm1, nm2, c1, c2,
                                                  s_twoOver1, s_twoOver2, s_vol,
                                                  s_out, threadBlock);
                }
            }

            // Group 1: faces normal to direction 1, drops the direction 1
            // factor.
            {
                const bool scale = c2;
                TData *dst       = scale ? s_vol : s_out;
                if (scale)
                {
                    ZeroModesTOPKernel(nmTot, s_vol, threadBlock);
                }

                for (unsigned int f = 0; f < nfaceN1; ++f)
                {
                    const TData *fin = inptr + in1 + f * fsz1;
                    const TData *fjac =
                        jacptr + jac1 + (DEFORMED ? f * fsz1 : f);

                    if (epd1)
                    {
                        IPWRTPhysTraceFaceN1TOPKernel<true, DEFORMED>(
                            f, f + 1, nfaceN1, nm0, nm1, nm2, nbd1, tnq10,
                            tnq11, tbd2, tbd3, tw10, tw11, fjac, s_wsp0, s_wsp1,
                            fin, dst, cd2, cd3, threadBlock);
                    }
                    else
                    {
                        IPWRTPhysTraceFaceN1TOPKernel<false, DEFORMED>(
                            f, f + 1, nfaceN1, nm0, nm1, nm2, nbd1, tnq10,
                            tnq11, tbd2, tbd3, tw10, tw11, fjac, s_wsp0, s_wsp1,
                            fin, dst, cd2, cd3, threadBlock);
                    }
                }

                if (scale)
                {
                    ScaleAddCollapsedFacTOPKernel(nm0, nm1, nm2, false, c2,
                                                  s_twoOver1, s_twoOver2, s_vol,
                                                  s_out, threadBlock);
                }
            }

            // Group 2: faces normal to direction 2, drops the direction 2
            // factor.
            {
                const bool scale = c1;
                TData *dst       = scale ? s_vol : s_out;
                if (scale)
                {
                    ZeroModesTOPKernel(nmTot, s_vol, threadBlock);
                }

                for (unsigned int f = 0; f < nfaceN2; ++f)
                {
                    const TData *fin = inptr + in2 + f * fsz2;
                    const TData *fjac =
                        jacptr + jac2 + (DEFORMED ? f * fsz2 : f);

                    if (epd2)
                    {
                        IPWRTPhysTraceFaceN2TOPKernel<true, DEFORMED>(
                            f, f + 1, nfaceN2, nm0, nm1, nm2, nbd2, tnq20,
                            tnq21, tbd4, tbd5, tw20, tw21, fjac, s_wsp0, s_wsp1,
                            fin, dst, cd4, cd5, threadBlock);
                    }
                    else
                    {
                        IPWRTPhysTraceFaceN2TOPKernel<false, DEFORMED>(
                            f, f + 1, nfaceN2, nm0, nm1, nm2, nbd2, tnq20,
                            tnq21, tbd4, tbd5, tw20, tw21, fjac, s_wsp0, s_wsp1,
                            fin, dst, cd4, cd5, threadBlock);
                    }
                }

                if (scale)
                {
                    ScaleAddCollapsedFacTOPKernel(nm0, nm1, nm2, c1, false,
                                                  s_twoOver1, s_twoOver2, s_vol,
                                                  s_out, threadBlock);
                }
            }
        }

        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            outptr[idx] = s_out[idx];
        }

        e += getBlockRange(threadBlock);
    }
}

} // namespace Nektar::MultiRegions::detail
