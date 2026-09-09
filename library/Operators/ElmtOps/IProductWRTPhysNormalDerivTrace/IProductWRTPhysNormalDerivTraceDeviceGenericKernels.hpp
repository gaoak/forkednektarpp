///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysNormalDerivTraceDeviceGenericKernels.hpp
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
// Description: Device kernels of the lift against the normal derivative of
// the test function
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysNormalDerivTraceDeviceGenericKernels.hpp
 * @brief Device kernels of the surface inner product against the normal
 * derivative of the volume cardinal basis, one element per warp lane.
 *
 * @details
 * The same decomposition as
 * IProductWRTPhysNormalDerivTraceSerialAVXGenericKernels.hpp: one trace
 * inner product per element direction, each the plain trace lift with the
 * direction's table replaced by its derivative, and the collapsed
 * \f$2/(1 - \eta)\f$ factors applied on the volume. The edge and face
 * cores are those of IProductWRTPhysTraceDeviceGenericKernels.hpp.
 *
 * Element data is warp interleaved: element @em e sits at lane
 * `e % warpsize` of warp `e / warpsize` and its @em i'th value at
 * `[warpsize * i + ilane]`. Tables are shared by every element of the
 * block and indexed directly. The three
 * IProductWRTPhysNormalDerivTraceKernelLauncher() overloads are launched on a
 * two-dimensional grid, elements along the first axis and components
 * along the second; overload resolution picks the dimension from the
 * number of arguments.
 *
 * @see IProductWRTPhysNormalDerivTraceDeviceGeneric.hpp for the block
 * operator that launches these kernels.
 */

#pragma once

#include "LibUtilities/Backends/Backends_Device_API.hpp"
#include "LibUtilities/BasicUtils/Utils/UtilsDeviceKernels.hpp"

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <Operators/ElmtOps/ElmtHelper.hpp>

#include "Operators/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceDeviceGenericKernels.hpp"

namespace Nektar::Operators::detail
{

/// @brief Workspace of a whole block, per component: none, a segment's
/// traces being points.
template <
    typename TTraceSizeParameter1D,
    std::enable_if_t<IsTraceSizeParameter1D_v<TTraceSizeParameter1D>, bool>
        Enable = true>
inline constexpr size_t IProductWRTPhysNormalDerivTraceWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Workspace of a whole block, per component: an edge pair's
/// trace-mode buffer followed by a volume-sized buffer for the collapsed
/// factor scaling, one of each per element.
template <
    typename TTraceSizeParameter2D,
    std::enable_if_t<IsTraceSizeParameter2D_v<TTraceSizeParameter2D>, bool>
        Enable = true>
inline constexpr size_t IProductWRTPhysNormalDerivTraceWorkSpaceSize(
    const size_t nelmt, const TTraceSizeParameter2D sizeParam2D)
{
    return (2u * IProductWRTPhysTraceEdgeModeBlockSize(sizeParam2D) +
            sizeParam2D.nmTot()) *
           nelmt;
}

/// @brief Workspace of a whole block, per component: the face mode blocks
/// of a face pair, the contraction scratch and a volume-sized buffer for
/// the collapsed factor scaling, one of each per element.
///
/// A face group goes through one call of the face core, which writes one
/// mode block per face covered, so the block is sized for the two faces a
/// direction can hold; the scratch is refilled per face and takes one. This
/// is what the SerialAVX implementation allocates, and the two-dimensional
/// size above does the same for an edge pair.
template <
    typename TTraceSizeParameter3D,
    std::enable_if_t<IsTraceSizeParameter3D_v<TTraceSizeParameter3D>, bool>
        Enable = true>
inline constexpr size_t IProductWRTPhysNormalDerivTraceWorkSpaceSize(
    const size_t nelmt, const TTraceSizeParameter3D sizeParam3D)
{
    return (2u * IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D) +
            IProductWRTPhysTraceFaceScratchSize(sizeParam3D) +
            sizeParam3D.nmTot()) *
           nelmt;
}

/**
 * @brief The one-dimensional lift: \f$\langle \partial_n \phi, g \rangle\f$
 * over a segment's two vertices, one lane per element.
 *
 * A segment's traces are points, so there is no trace quadrature to sum
 * over, no tangential table and no trace Jacobian. What is left is the
 * point evaluation
 * \f[ out_p \mathrel{+}= \sum_{v=0,1} \partial h_p(\xi_v)\, F_v\, g_v , \f]
 * with \f$F_v\f$ the normal derivative factor at that vertex, which the
 * data warehouse supplies as the single component of the factor array. A
 * segment carries one factor per vertex whether or not it is deformed.
 *
 * @tparam SHAPE_TYPE   Seg; unread.
 * @tparam DEFORMED     Unread: a segment carries one factor per vertex
 *                      whatever its geometry.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TTraceSizeParameter1D, typename TData, typename TthreadBlock>
NEK_DEVICE_KERNEL void IProductWRTPhysNormalDerivTraceKernelLauncher(
    const TTraceSizeParameter1D sizeParam1D,
    [[maybe_unused]] const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT dnbasis0, const size_t nelmt,
    const unsigned int numDataIn, const unsigned int numDataOut,
    const unsigned int numDataJac, [[maybe_unused]] const size_t jacCompStride,
    [[maybe_unused]] const TData *NEK_RESTRICT twoOver0,
    const TData *NEK_RESTRICT jac, [[maybe_unused]] TData *NEK_RESTRICT wsp,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    [[maybe_unused]] const bool endPtsCollocated0, const bool append,
    [[maybe_unused]] unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter1D or "
                  "TemplatedTraceSizeParameter1D.");

    const unsigned int nm0          = sizeParam1D.nm0();
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr  = in + numDataIn * (nelmt * c + warpsize * iwarp);
        TData *outptr       = out + numDataOut * (nelmt * c + warpsize * iwarp);
        const TData *jacptr = jac + numDataJac * warpsize * iwarp;

        if (!append)
        {
            for (unsigned int p = 0; p < nm0; ++p)
            {
                outptr[warpsize * p + ilane] = 0.0;
            }
        }

        const TData f0 = jacptr[warpsize * 0 + ilane];
        const TData f1 = jacptr[warpsize * 1 + ilane];

        const TData g0 = inptr[warpsize * 0 + ilane];
        const TData g1 = inptr[warpsize * 1 + ilane];

        for (unsigned int p = 0; p < nm0; ++p)
        {
            TData sum = 0.0;
            sum += dnbasis0[2 * p] * f0 * g0;
            sum += dnbasis0[2 * p + 1] * f1 * g1;

            outptr[warpsize * p + ilane] += sum;
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

/// @brief Scale a volume workspace by the collapsed factors a trace group
/// still needs and accumulate it into the output. The factors are table
/// data shared by the block and indexed directly; the volume arrays are
/// warp interleaved. In two dimensions @p nm2 is one and @p twoOver2 is
/// unread.
template <typename TData>
NEK_DEVICE_INLINE void ScaleAddCollapsedFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const bool useD1, const bool useD2,
    const TData *NEK_RESTRICT twoOver1, const TData *NEK_RESTRICT twoOver2,
    const TData *NEK_RESTRICT volwsp, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    for (unsigned int k = 0; k < nm2; ++k)
    {
        for (unsigned int j = 0; j < nm1; ++j)
        {
            TData f = 1.0;
            if (useD1)
            {
                f *= twoOver1[j];
            }
            if (useD2)
            {
                f *= twoOver2[k];
            }

            const unsigned int o = (k * nm1 + j) * nm0;
            for (unsigned int i = 0; i < nm0; ++i)
            {
                out[warpsize * (o + i) + ilane] +=
                    volwsp[warpsize * (o + i) + ilane] * f;
            }
        }
    }
}

/**
 * @brief All edges of one two-dimensional element: the two terms of the
 * operator, one per element direction, accumulated in turn; one lane per
 * element.
 *
 * Which table carries the derivative is decided per term from the trace
 * direction, exactly as the SerialAVX kernels do. A collapsed shape
 * carries \f$2/(1 - \eta_1)\f$ on its \f$\eta_0\f$ term; that factor is
 * singular at the collapsed apex, a quadrature point of the trace but
 * never of the element, so the affected edge group goes into the volume
 * workspace and is scaled there. The group normal to the collapsed
 * direction sits at \f$\eta_1 = -1\f$ where the factor is one, and is
 * accumulated straight into the output.
 *
 * @param   jac             Factor array of the block, term 0; term 1 sits
 *                          @p jacCompStride further on.
 * @param   wsp             Static workspace of the block, holding the edge
 *                          scratch and the volume buffer for every element.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TTraceSizeParameter2D, typename TData, typename TthreadBlock>
NEK_DEVICE_KERNEL void IProductWRTPhysNormalDerivTraceKernelLauncher(
    const TTraceSizeParameter2D sizeParam2D, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT dnbasis0,
    const TData *NEK_RESTRICT dnbasis1, const size_t nelmt,
    const unsigned int numDataIn, const unsigned int numDataOut,
    const unsigned int numDataJac, const size_t jacCompStride,
    const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
    const TData *NEK_RESTRICT dtbasis0, const TData *NEK_RESTRICT dtbasis1,
    const TData *NEK_RESTRICT tw0, const TData *NEK_RESTRICT tw1,
    [[maybe_unused]] const TData *NEK_RESTRICT twoOver0,
    const TData *NEK_RESTRICT twoOver1, const TData *NEK_RESTRICT jac,
    TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const unsigned int traceDir0,
    const unsigned int traceDir1, const bool isColl0, const bool isColl1,
    const bool endPtsColl0, const bool endPtsColl1, const bool append,
    [[maybe_unused]] unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int tnq00 = sizeParam2D.nq00();
    const unsigned int tnq10 = sizeParam2D.nq10();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    constexpr unsigned int nedgeN1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

    const size_t edgeSize =
        2u * IProductWRTPhysTraceEdgeModeBlockSize(sizeParam2D);

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr = in + numDataIn * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + numDataOut * (nelmt * c + warpsize * iwarp);
        TData *wspptr =
            wsp + (edgeSize + numDataOut) * (nelmt * c + warpsize * iwarp);
        TData *volptr = wspptr + edgeSize * warpsize;

        // The trace data and the factor array are warp interleaved, so
        // stepping past the two N0 edges advances by warpsize entries per
        // point.
        const unsigned int off1  = warpsize * 2 * tnq00;
        const unsigned int joff1 = (DEFORMED) ? off1 : warpsize * 2;

        for (unsigned int d = 0; d < 2; ++d)
        {
            // Only a collapsed shape carries the factor, and only on its
            // eta_0 term.
            const bool useD1 = CompNeedsCollapsedFac(SHAPE_TYPE, d, 1);

            // Components of the factor array sit one whole component apart.
            const TData *jacptr =
                jac + d * jacCompStride + numDataJac * warpsize * iwarp;

            if (!(append || d > 0))
            {
                for (unsigned int p = 0; p < numDataOut; ++p)
                {
                    outptr[warpsize * p + ilane] = 0.0;
                }
            }

            // Term d picks the derivative table wherever a slot's element
            // direction is d. A derivative slot is never collocated, and a
            // direction never has its own end points collocated for its own
            // term.

            // Group 0: edges normal to direction 0. Keeps the factor.
            {
                TData *dst = useD1 ? volptr : outptr;
                if (useD1)
                {
                    for (unsigned int p = 0; p < numDataOut; ++p)
                    {
                        dst[warpsize * p + ilane] = 0.0;
                    }
                }

                const TData *nb = (0 == d) ? dnbasis0 : nbasis0;
                const TData *tb = (traceDir0 == d) ? dtbasis0 : tbasis0;
                const bool coll = (traceDir0 == d) ? false : isColl0;
                const bool endp = (0 == d) ? false : endPtsColl0;

                if (endp)
                {
                    IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                        ilane, 0, 2, 2, nm0, nm1, nb, tnq00, tb, tw0, jacptr,
                        wspptr, inptr, dst, coll);
                }
                else
                {
                    IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                        ilane, 0, 2, 2, nm0, nm1, nb, tnq00, tb, tw0, jacptr,
                        wspptr, inptr, dst, coll);
                }

                if (useD1)
                {
                    ScaleAddCollapsedFacKernel(ilane, nm0, nm1, 1u, true, false,
                                               twoOver1, twoOver1, volptr,
                                               outptr);
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
                    IPWRTPhysTraceEdgeN1Kernel<true, DEFORMED>(
                        ilane, 0, nedgeN1, nedgeN1, nm0, nm1, nb, tnq10, tb,
                        tw1, jacptr + joff1, wspptr, inptr + off1, outptr,
                        coll);
                }
                else
                {
                    IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
                        ilane, 0, nedgeN1, nedgeN1, nm0, nm1, nb, tnq10, tb,
                        tw1, jacptr + joff1, wspptr, inptr + off1, outptr,
                        coll);
                }
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief All faces of one three-dimensional element: the three terms of
 * the operator, one per element direction, accumulated in turn; one lane
 * per element.
 *
 * A face group normal to a collapsed direction sits at \f$\eta = -1\f$,
 * where that direction's factor is one, so group @em k drops the
 * direction @em k factor; the groups that keep one go through the volume
 * workspace and are scaled on the element's own points, away from the
 * singular apex.
 *
 * @param   jac             Factor array of the block, term 0; term @em d
 *                          sits `d * jacCompStride` further on.
 * @param   wsp             Static workspace of the block, holding the face
 *                          mode block, the contraction scratch and the
 *                          volume buffer for every element.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TTraceSizeParameter3D, typename TData, typename TthreadBlock>
NEK_DEVICE_KERNEL void IProductWRTPhysNormalDerivTraceKernelLauncher(
    const TTraceSizeParameter3D sizeParam3D, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
    const TData *NEK_RESTRICT dnbasis0, const TData *NEK_RESTRICT dnbasis1,
    const TData *NEK_RESTRICT dnbasis2, const size_t nelmt,
    const unsigned int numDataIn, const unsigned int numDataOut,
    const unsigned int numDataJac, const size_t jacCompStride,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    const TData *NEK_RESTRICT dtbasis00, const TData *NEK_RESTRICT dtbasis01,
    const TData *NEK_RESTRICT dtbasis10, const TData *NEK_RESTRICT dtbasis11,
    const TData *NEK_RESTRICT dtbasis20, const TData *NEK_RESTRICT dtbasis21,
    const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
    const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
    [[maybe_unused]] const TData *NEK_RESTRICT twoOver0,
    const TData *NEK_RESTRICT twoOver1, const TData *NEK_RESTRICT twoOver2,
    const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT wsp,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const unsigned int traceDir00, const unsigned int traceDir01,
    const unsigned int traceDir10, const unsigned int traceDir11,
    const unsigned int traceDir20, const unsigned int traceDir21,
    const bool isColl00, const bool isColl01, const bool isColl10,
    const bool isColl11, const bool isColl20, const bool isColl21,
    const bool endPtsColl0, const bool endPtsColl1, const bool endPtsColl2,
    const bool append, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int tnq00 = sizeParam3D.nq00();
    const unsigned int tnq01 = sizeParam3D.nq01();
    const unsigned int tnq10 = sizeParam3D.nq10();
    const unsigned int tnq11 = sizeParam3D.nq11();
    const unsigned int tnq20 = sizeParam3D.nq20();
    const unsigned int tnq21 = sizeParam3D.nq21();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    constexpr unsigned int nfaceN1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned int nfaceN2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];

    // The face cores want the same two scratch regions the plain trace
    // lift uses, followed here by the volume buffer. A whole face group
    // goes through one call, so the mode block holds a face pair where the
    // plain lift, integrating one face at a time, holds one.
    const size_t wsp0Size =
        2u * IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D);
    const size_t wsp1Size = IProductWRTPhysTraceFaceScratchSize(sizeParam3D);

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr = in + numDataIn * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + numDataOut * (nelmt * c + warpsize * iwarp);
        TData *wspptr      = wsp + (wsp0Size + wsp1Size + numDataOut) *
                                  (nelmt * c + warpsize * iwarp);
        TData *wsp1ptr = wspptr + wsp0Size * warpsize;
        TData *volptr  = wsp1ptr + wsp1Size * warpsize;

        // Face group offsets into the input and the factors.
        const unsigned int in1  = 2 * tnq00 * tnq01;
        const unsigned int jac1 = DEFORMED ? in1 : 2;
        const unsigned int in2  = in1 + nfaceN1 * tnq10 * tnq11;
        const unsigned int jac2 =
            jac1 + (DEFORMED ? nfaceN1 * tnq10 * tnq11 : nfaceN1);

        for (unsigned int d = 0; d < 3; ++d)
        {
            const bool c1 = CompNeedsCollapsedFac(SHAPE_TYPE, d, 1);
            const bool c2 = CompNeedsCollapsedFac(SHAPE_TYPE, d, 2);

            const TData *jacptr =
                jac + d * jacCompStride + numDataJac * warpsize * iwarp;

            if (!(append || d > 0))
            {
                for (unsigned int p = 0; p < numDataOut; ++p)
                {
                    outptr[warpsize * p + ilane] = 0.0;
                }
            }

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
                TData *dst       = scale ? volptr : outptr;
                if (scale)
                {
                    for (unsigned int p = 0; p < numDataOut; ++p)
                    {
                        dst[warpsize * p + ilane] = 0.0;
                    }
                }

                if (epd0)
                {
                    IPWRTPhysTraceFaceN0Kernel<true, DEFORMED>(
                        ilane, 0, 2, 2, nm0, nm1, nm2, nbd0, tnq00, tnq01, tbd0,
                        tbd1, tw00, tw01, jacptr, wspptr, wsp1ptr, inptr, dst,
                        cd0, cd1);
                }
                else
                {
                    IPWRTPhysTraceFaceN0Kernel<false, DEFORMED>(
                        ilane, 0, 2, 2, nm0, nm1, nm2, nbd0, tnq00, tnq01, tbd0,
                        tbd1, tw00, tw01, jacptr, wspptr, wsp1ptr, inptr, dst,
                        cd0, cd1);
                }

                if (scale)
                {
                    ScaleAddCollapsedFacKernel(ilane, nm0, nm1, nm2, c1, c2,
                                               twoOver1, twoOver2, volptr,
                                               outptr);
                }
            }

            // Group 1: faces normal to direction 1, drops the direction 1
            // factor.
            {
                const bool scale = c2;
                TData *dst       = scale ? volptr : outptr;
                if (scale)
                {
                    for (unsigned int p = 0; p < numDataOut; ++p)
                    {
                        dst[warpsize * p + ilane] = 0.0;
                    }
                }

                if (epd1)
                {
                    IPWRTPhysTraceFaceN1Kernel<true, DEFORMED>(
                        ilane, 0, nfaceN1, nfaceN1, nm0, nm1, nm2, nbd1, tnq10,
                        tnq11, tbd2, tbd3, tw10, tw11, jacptr + jac1 * warpsize,
                        wspptr, wsp1ptr, inptr + in1 * warpsize, dst, cd2, cd3);
                }
                else
                {
                    IPWRTPhysTraceFaceN1Kernel<false, DEFORMED>(
                        ilane, 0, nfaceN1, nfaceN1, nm0, nm1, nm2, nbd1, tnq10,
                        tnq11, tbd2, tbd3, tw10, tw11, jacptr + jac1 * warpsize,
                        wspptr, wsp1ptr, inptr + in1 * warpsize, dst, cd2, cd3);
                }

                if (scale)
                {
                    ScaleAddCollapsedFacKernel(ilane, nm0, nm1, nm2, false, c2,
                                               twoOver1, twoOver2, volptr,
                                               outptr);
                }
            }

            // Group 2: faces normal to direction 2, drops the direction 2
            // factor.
            {
                const bool scale = c1;
                TData *dst       = scale ? volptr : outptr;
                if (scale)
                {
                    for (unsigned int p = 0; p < numDataOut; ++p)
                    {
                        dst[warpsize * p + ilane] = 0.0;
                    }
                }

                if (epd2)
                {
                    IPWRTPhysTraceFaceN2Kernel<true, DEFORMED>(
                        ilane, 0, nfaceN2, nfaceN2, nm0, nm1, nm2, nbd2, tnq20,
                        tnq21, tbd4, tbd5, tw20, tw21, jacptr + jac2 * warpsize,
                        wspptr, wsp1ptr, inptr + in2 * warpsize, dst, cd4, cd5);
                }
                else
                {
                    IPWRTPhysTraceFaceN2Kernel<false, DEFORMED>(
                        ilane, 0, nfaceN2, nfaceN2, nm0, nm1, nm2, nbd2, tnq20,
                        tnq21, tbd4, tbd5, tw20, tw21, jacptr + jac2 * warpsize,
                        wspptr, wsp1ptr, inptr + in2 * warpsize, dst, cd4, cd5);
                }

                if (scale)
                {
                    ScaleAddCollapsedFacKernel(ilane, nm0, nm1, nm2, c1, false,
                                               twoOver1, twoOver2, volptr,
                                               outptr);
                }
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

} // namespace Nektar::Operators::detail
