////////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysTraceDeviceGenericKernels.hpp
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
////////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysTraceDeviceGenericKernels.hpp
 * @brief Device kernels of the trace inner product: one element
 * per warp lane.
 *
 * @details
 * IProductWRTPhysTrace is the lifting term of the discontinuous Galerkin
 * operators. Given a field already evaluated at the quadrature points of
 * an element's traces, typically a numerical flux \f$\hat f\f$, it
 * accumulates a volume physical-shaped array whose entries are surface
 * inner products against the cardinal (hat) basis of the volume
 * quadrature grid,
 * \f[
 *   \Lambda_{\boldsymbol p} \;=\; \sum_{F \in \partial E} \int_F \hat f\,
 *   h_{\boldsymbol p}\big|_F\, J_F\, \mathrm{d}s .
 * \f]
 * Those entries are not modal coefficients and not point values of any
 * function. The modal right-hand side is recovered downstream by a
 * transposed basis contraction with neither quadrature weights nor volume
 * Jacobian, both the surface rule's weights and the face Jacobian being
 * inside \f$\Lambda\f$ already.
 *
 * On a direction-@em a trace the hat basis factorises as
 * \f$h_{\boldsymbol p}|_{\xi_a = \pm 1} = h_{p_a}(\pm 1) \prod_{b \neq a}
 * h_{p_b}(\xi_b)\f$, which splits every face integral into two stages:
 * a sum-factorised inner product over the trace directions, then a
 * rank-one lift along the normal direction. Every kernel in this file is
 * one of those two stages, or glue routing a face to them.
 *
 * @section ipwrttrace_dev_naming Naming
 *
 * @c nm0, @c nm1 and @c nm2 hold @c GetNumPoints, that is volume
 * quadrature counts and @em not mode counts. @c nbasis is the @c eInterp
 * table \f$h_p(\pm 1)\f$ from the volume grid of a direction to that
 * direction's trace positions, and @c tbasis the @c eInterp table
 * \f$h_p(\xi^{tr}_i)\f$ from the volume grid to the trace quadrature
 * points. Neither is an expansion basis evaluated anywhere. Both are
 * stored one row per volume point and one column per output point, so
 * @c tbasis is indexed `tbasis[p * tnq + i]` and @c nbasis
 * `nbasis[p * tstride + f]` with @c tstride the number of traces that
 * direction has: two for every direction of a hexahedron, one for a
 * collapsed direction (direction 1 of a tetrahedron, direction 2 of
 * prism, pyramid and tetrahedron). @c tw are the trace quadrature
 * weights and @c tjac the trace Jacobian.
 *
 * @section ipwrttrace_dev_layout Layout contracts
 *
 * - Warp interleaving. Element \f$e = (i_{warp}, i_{lane})\f$ stores
 *   entry @em n at `buf[warpsize * n + ilane]` within its warp block, and
 *   warp blocks stride by `numData * warpsize`. The top-level kernels
 *   grid-stride over the elements and offset every pointer to the warp
 *   before calling a per-element worker.
 * - Trace packing. The packed input, and the deformed Jacobian, hold the
 *   faces in pair order: the \f$N_0\f$ pair first, then the \f$N_1\f$
 *   pair or single, then \f$N_2\f$, face-major within a pair.
 * - Regular Jacobian. One slot per face in that same order, which gives
 *   the face-id to slot tables hex `{4,2,1,3,0,5}`, prism and pyramid
 *   `{4,2,1,3,0}` and tetrahedron `{3,2,1,0}`.
 * - Face to (direction, position). Hex, prism and pyramid
 *   `4->(0,0)`, `2->(0,1)`, `1->(1,0)`, `3->(1,1)`, `0->(2,0)`, and for
 *   the hexahedron alone `5->(2,1)`; tetrahedron `3->(0,0)`, `2->(0,1)`,
 *   `1->(1,0)`, `0->(2,0)`.
 * - The \f$N_a\f$ trace block spans the two volume quadrature counts
 *   other than \f$Q_a\f$, so the glue level routes purely by direction:
 *   \f$N_0\f$ uses `(nm1,nm2)`, \f$N_1\f$ uses `(nm0,nm2)` and
 *   \f$N_2\f$ uses `(nm0,nm1)`.
 *
 * @section ipwrttrace_dev_fast Fast paths
 *
 * Two flags select them, both decided at setup and passed as runtime
 * booleans. @c endPtsCollocated says the volume rule of the normal direction
 * contains the domain endpoints, so \f$h_p(\pm 1)\f$ is a Kronecker delta
 * and the lift degenerates to writing a single boundary plane. @c isCollocated
 * says the trace
 * quadrature points of a tangential direction coincide with the volume points
 * of that direction, so the interpolation table is the identity and the
 * contraction degenerates to a pointwise multiply by the trace weights
 * and the Jacobian.
 *
 * @section ipwrttrace_dev_layer Layering
 *
 * The live path is four layers deep, mirroring the SerialAVX
 * implementation one dimension up:
 * - top-level kernels (`IProductWRTPhysTraceKernelLauncher`,
 *   `IProductWRTPhysTrace{Edge,Face}Kernel`), which own the grid-stride
 *   loop, the warp mapping and the workspace partitioning;
 * - wrappers (`IProductWRTPhysTrace2D`, `IProductWRTPhysTrace3DFace`),
 *   which zero the volume field and walk the packed traces of one
 *   element;
 * - workers (`IProductWRTPhysTraceEdge`, `IProductWRTPhysTraceFace`),
 *   which turn a face or edge id into a direction and a position;
 * - glue (`IPWRTPhysTraceEdgeN{0,1}Kernel`,
 *   `IPWRTPhysTraceFaceN{0,1,2}Kernel`), which pick the direction's
 *   tables and counts and call the two leaf stages,
 *   `IPWRTPhys{Edge,Face}Kernel` and `AddFaceN{0,1,2}ToVolKernel` or
 *   `AddEdgeN{0,1}ToVolKernel`.
 *
 * @section ipwrttrace_dev_state State of the paths
 *
 * Not everything here is live, and not everything live is correct:
 * - These kernels assume warp-width interleaving and carry no
 *   implementation tag: there is one trace inner-product algorithm and
 *   the block operator registers it under @c Generic. A @c SumFacTOP
 *   block operator's width-one interleave would contradict the
 *   interleaving assumed here, and no template parameter now offers to
 *   accept it.
 * - The retained monolithic predecessors of the decomposed face family
 *   (`IProductWRTPhysTraceFaceDir{0,1,2}Core`,
 *   `IProductWRTPhysTrace3DKernel` and the per-shape
 *   `IProductWRTPhysTrace{Hex,Prism,Pyr,Tet}FaceKernel`) have been
 *   deleted: the decomposed family now serves every 3D shape,
 *   @c NodalPrism included, which shares the prism's trace structure.
 *
 * No kernel in this file uses shared memory; every @c shmemptr parameter
 * is present for the launch macro's signature only.
 *
 * @see IProductWRTPhysTraceDeviceGeneric.hpp for the dispatch and the
 * launchers, IProductWRTPhysTraceSerialAVXGenericKernels.hpp for the
 * SIMD counterparts of the same decomposition, and the PhysTraceExtract
 * kernels for the adjoint operation, which applies the same tables
 * untransposed.
 */

#pragma once

#include "LibUtilities/Backends/Backends_Device_API.hpp"
#include "LibUtilities/BasicUtils/Utils/UtilsDeviceKernels.hpp"

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

// Helper functions

/**
 * @brief Per-element size of one edge's trace-mode buffer: the edge
 * inner product writes it and the lift reads it back.
 *
 * The bulk path holds a whole edge pair at once and so takes twice
 * this; the per-trace path takes one. Host and kernel both size against
 * this, so the two cannot drift apart.
 */
template <
    typename TTraceSizeParameter2D,
    std::enable_if_t<IsTraceSizeParameter2D_v<TTraceSizeParameter2D>, bool>
        Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int IProductWRTPhysTraceEdgeModeBlockSize(
    const TTraceSizeParameter2D sizeParam2D)
{
    return std::max(sizeParam2D.nm0(), sizeParam2D.nm1());
}

/**
 * @brief Per-element size of the first three-dimensional workspace
 * region: one face's mode block, the largest of `nm0 * nm1`,
 * `nm1 * nm2` and `nm0 * nm2`.
 *
 * Sizing this region for a single face is what forces the kernels to
 * process faces one at a time rather than batching a pair.
 */
template <
    typename TTraceSizeParameter3D,
    std::enable_if_t<IsTraceSizeParameter3D_v<TTraceSizeParameter3D>, bool>
        Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int IProductWRTPhysTraceFaceModeBlockSize(
    const TTraceSizeParameter3D sizeParam3D)
{
    const unsigned int nm0 = sizeParam3D.nm0();
    const unsigned int nm1 = sizeParam3D.nm1();
    const unsigned int nm2 = sizeParam3D.nm2();

    return std::max(nm0 * nm1, std::max(nm1 * nm2, nm0 * nm2));
}

/**
 * @brief Per-element size of the second three-dimensional workspace
 * region: the scratch of the general (non-collocated) face
 * contraction, which holds the modes of the first in-trace direction
 * against the trace points of the second.
 */
template <
    typename TTraceSizeParameter3D,
    std::enable_if_t<IsTraceSizeParameter3D_v<TTraceSizeParameter3D>, bool>
        Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int IProductWRTPhysTraceFaceScratchSize(
    const TTraceSizeParameter3D sizeParam3D)
{
    const unsigned int nm0 = sizeParam3D.nm0();
    const unsigned int nm1 = sizeParam3D.nm1();

    return std::max(
        nm1 * sizeParam3D.nq01(),
        std::max(nm0 * sizeParam3D.nq11(), nm0 * sizeParam3D.nq21()));
}

/**
 * @brief Workspace of a whole block, per component: none, a segment's
 * traces being points that carry no surface measure.
 */
template <
    typename TTraceSizeParameter1D,
    std::enable_if_t<IsTraceSizeParameter1D_v<TTraceSizeParameter1D>, bool>
        Enable = true>
inline constexpr size_t IProductWRTPhysTraceWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter1D sizeParam1D)
{
    return 0;
}

/**
 * @brief Workspace of a whole block, per component: @p ntrace
 * trace-mode buffers per element.
 *
 * @param   ntrace  Edges the buffer must hold at once. Defaults to
 *                  two, the bulk path integrating an edge pair in one
 *                  call, so that OperatorNDImpl() can size every
 *                  dimension through one uniform call; the per-trace
 *                  launcher passes one.
 */
template <
    typename TTraceSizeParameter2D,
    std::enable_if_t<IsTraceSizeParameter2D_v<TTraceSizeParameter2D>, bool>
        Enable = true>
inline constexpr size_t IProductWRTPhysTraceWorkSpaceSize(
    const size_t nelmt, const TTraceSizeParameter2D sizeParam2D,
    const unsigned int ntrace = 2u)
{
    return ntrace * IProductWRTPhysTraceEdgeModeBlockSize(sizeParam2D) * nelmt;
}

/// @brief Workspace of a whole block, per component: the face mode
/// block followed by the contraction scratch, one of each per element.
/// The bulk and the per-trace path take the same.
template <
    typename TTraceSizeParameter3D,
    std::enable_if_t<IsTraceSizeParameter3D_v<TTraceSizeParameter3D>, bool>
        Enable = true>
inline constexpr size_t IProductWRTPhysTraceWorkSpaceSize(
    const size_t nelmt, const TTraceSizeParameter3D sizeParam3D)
{
    return (IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D) +
            IProductWRTPhysTraceFaceScratchSize(sizeParam3D)) *
           nelmt;
}

/**
 * @brief Read the trace Jacobian of one element, per point or per face.
 *
 * The trace Jacobian is warp interleaved with one value per trace
 * quadrature point when the geometry is deformed and one value per face,
 * in the packed trace order, when it is regular. This helper hides that
 * choice behind a runtime flag rather than the @c DEFORMED template
 * parameter the rest of the file uses, and is called only by the
 * decomposed kernels throughout this file.
 *
 * @tparam TData    Floating-point type of the field data.
 *
 * @param   deformed    Jacobian varies point by point.
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   jlocoff     Face slot to read when the geometry is regular.
 * @param   pt          Trace point index to read when @p deformed.
 * @param   jac         Trace Jacobian of this lane's warp.
 *
 * @return The Jacobian value to apply at that point.
 */
template <typename TData>
NEK_DEVICE_INLINE TData GetJac(const bool deformed, const unsigned int ilane,
                               const unsigned int jlocoff,
                               const unsigned int pt,
                               const TData *NEK_RESTRICT jac)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    if (deformed)
    {
        return jac[warpsize * pt + ilane];
    }

    return jac[warpsize * jlocoff + ilane];
}

// -----------------------------------------------------------------------------
// 1D KERNELS
// -----------------------------------------------------------------------------

/**
 * @brief One block of segments: lift the two endpoint values onto the
 * volume grid.
 *
 * The traces of a segment are its two vertices, so there is no surface
 * measure, no trace quadrature and no Jacobian to apply: the whole
 * operator is the rank-one lift
 * \f$\Lambda_p \mathrel{+}= h_p(-1)\,\hat f_0 + h_p(+1)\,\hat f_1\f$.
 * Under @p endPtsCollocated the volume rule contains both domain endpoints,
 * \f$h_p(\pm 1)\f$ is a Kronecker delta and the two values drop straight
 * into the first and last volume point. Otherwise the interpolation
 * table is applied in full; a segment has two traces in direction 0,
 * which is the stride of two into @p nbasis0.
 *
 * Each grid-stride iteration takes one element, splits the global index
 * into warp and lane and offsets @p in and @p out to that warp.
 *
 * The one-dimensional arm of the overload set OperatorND() launches;
 * the higher-dimensional arms follow below. It takes the whole
 * family's argument order and so accepts @p numDataJac, @p tjac and
 * @p wsp, which a segment needs for none of: its traces are points,
 * which carry no surface measure, and there is no tangential stage to
 * give a workspace to. The tangential tables, weights and collocation
 * flags are empty packs in one dimension, so none is passed either.
 *
 * @tparam SHAPE_TYPE       Seg; unread, the path being the same for
 *                          every one-dimensional expansion.
 * @tparam DEFORMED         Unread: a segment trace carries no
 *                          Jacobian.
 * @tparam TTraceSizeParameter1D    NonTemplatedTraceSizeParameter1D or
 *                          a TemplatedTraceSizeParameter1D
 *                          instantiation.
 * @tparam TData            Floating-point type of the field data.
 * @tparam TthreadBlock     Thread-block handle type.
 *
 * @param   sizeParam1D Volume quadrature points of the segment, nm0().
 *                      Its nq0() is accepted for symmetry with the
 *                      other size-specialised launchers and not used by
 *                      this kernel, which has no trace quadrature.
 * @param   nelmt       Elements in the block, including padding.
 * @param   numDataIn   Entries per element of @p in (the two vertices).
 * @param   numDataOut  Entries per element of @p out (@c nm0).
 * @param   numDataJac  Unread, see above.
 * @param   tjac        Unread, see above, and null in one dimension.
 * @param   wsp         Unread, see above, and null because
 *                      IProductWRTPhysTraceWorkSpaceSize() asks for
 *                      none.
 * @param   nbasis0     eInterp table \f$h_p(\pm 1)\f$, indexed
 *                      `nbasis0[2 * p + f]`.
 * @param   in          Packed trace values of the block.
 * @param   out         Volume-shaped result of the block.
 * @param   endPtsCollocated  Volume rule contains the domain endpoints.
 * @param   append      Accumulate onto @p out instead of zeroing it
 *                      first.
 * @param   shmemptr    Dynamic shared memory; unused.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TTraceSizeParameter1D, typename TData, typename TthreadBlock>
NEK_DEVICE_KERNEL void IProductWRTPhysTraceKernelLauncher(
    const TTraceSizeParameter1D sizeParam1D, const TData *NEK_RESTRICT nbasis0,
    const size_t nelmt, const unsigned int numDataIn,
    const unsigned int numDataOut,
    [[maybe_unused]] const unsigned int numDataJac,
    [[maybe_unused]] const TData *NEK_RESTRICT tjac,
    [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool endPtsCollocated, const bool append,
    [[maybe_unused]] unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter1D or "
                  "TemplatedTraceSizeParameter1D.");

    const unsigned int nm0 = sizeParam1D.nm0();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    size_t e                        = getGlobalIdx<0>(threadBlock);
    const unsigned int c            = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr = in + numDataIn * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + numDataOut * (nelmt * c + warpsize * iwarp);

        if (!append)
        {
            for (unsigned int p = 0; p < nm0; ++p)
            {
                outptr[warpsize * p + ilane] = 0.0;
            }
        }

        if (endPtsCollocated)
        {
            outptr[warpsize * 0 + ilane] += inptr[warpsize * 0 + ilane];
            outptr[warpsize * (nm0 - 1) + ilane] += inptr[warpsize * 1 + ilane];
        }
        else
        {
            for (unsigned int p = 0; p < nm0; ++p)
            {
                TData sum = 0.0;
                sum += nbasis0[2 * p] * inptr[warpsize * 0 + ilane];
                sum += nbasis0[2 * p + 1] * inptr[warpsize * 1 + ilane];

                outptr[warpsize * p + ilane] += sum;
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

// -----------------------------------------------------------------------------
//  2D KERNELS utilising normal to trace directions
// -----------------------------------------------------------------------------

// Kernel for inner product over edge
/**
 * @brief Leaf kernel, stage one in two dimensions: integrate an edge, or
 * a pair of edges, against the hat functions of the in-edge direction.
 *
 * For every edge in [@p edg, @p nedge) and every volume point @em q of
 * the in-edge direction this forms
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
 * @p tjac to the right face.
 *
 * @tparam DEFORMED   Trace Jacobian varies point by point.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   edg         Position of the first edge within its direction.
 * @param   nedge       One past the position of the last edge.
 * @param   nm          Volume quadrature points of the in-edge
 *                      direction, hence entries produced per edge.
 * @param   tnq         Trace quadrature points of the edge.
 * @param   tbasis      eInterp table \f$h_q(\xi^{tr}_i)\f$, indexed
 *                      `tbasis[q * tnq + i]`.
 * @param   tw          Trace quadrature weights.
 * @param   tjac        Trace Jacobian of this lane's warp.
 * @param   in          Packed trace values of the edges covered.
 * @param   out         Trace-mode workspace, warp interleaved.
 * @param   isCollocated  Trace points coincide with the volume points.
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysEdgeKernel(
    const unsigned int ilane, const unsigned edg, const unsigned nedge,
    const unsigned nm, const unsigned tnq, const TData *NEK_RESTRICT tbasis,
    const TData *NEK_RESTRICT tw, const TData *NEK_RESTRICT tjac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    if (isCollocated)
    {
        unsigned cnt = 0;
        for (unsigned e = edg; e < nedge; ++e)
        {
            for (unsigned q = 0; q < nm; ++q, ++cnt)
            {
                // integrate mode over edge
                TData jac_val;

                if constexpr (DEFORMED)
                {
                    jac_val = tjac[warpsize * cnt + ilane];
                }
                else
                {
                    jac_val = tjac[warpsize * (e - edg) + ilane];
                }

                // wsp is shared by the whole warp, so it must be indexed
                // per-lane like the 3D kernels do - a plain out[cnt] makes
                // all 32 lanes race on the same slots.
                out[warpsize * cnt + ilane] =
                    in[warpsize * cnt + ilane] * jac_val * tw[q];
            }
        }
    }
    else // integrate over whole edge
    {
        unsigned cnt     = 0;
        unsigned qOffset = 0;
        for (unsigned e = edg; e < nedge; ++e)
        {
            for (unsigned q = 0; q < nm; ++q)
            {
                TData sum_q      = 0.0;
                unsigned bOffset = q * tnq;

                for (unsigned i = 0; i < tnq; ++i)
                {
                    TData jac_val;

                    if constexpr (DEFORMED)
                    {
                        jac_val = tjac[warpsize * (qOffset + i) + ilane];
                    }
                    else
                    {
                        jac_val = tjac[warpsize * (e - edg) + ilane];
                    }

                    sum_q += in[warpsize * (qOffset + i) + ilane] *
                             tbasis[bOffset + i] * jac_val * tw[i];
                }
                out[warpsize * cnt + ilane] = sum_q;
                ++cnt;
            }
            qOffset += tnq;
        }
    }
}

// kernels for edge 3 and 1 in quads
/**
 * @brief Leaf kernel, stage two in two dimensions: lift the results of a
 * direction-0 normal edge into the volume field.
 *
 * Applies \f$\Lambda_{iq} \mathrel{+}= h_i(\pm 1)\, F_q\f$, the volume
 * field being indexed `nm0 * q + i`. Under @p END_PTS_COLLOCATED0 the
 * interpolation is a delta and the edge lands entirely in the boundary
 * plane `i = e * (nm0 - 1)`, the loop variable @em e being the position
 * of the edge within its direction (0 lower, 1 upper) rather than its
 * index within this call, which is why the caller passes absolute
 * positions.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane   Lane index within the warp; selects the element.
 * @param   edg     Position of the first edge within direction 0.
 * @param   nedge   One past the position of the last edge.
 * @param   tstride Row stride of @p nbasis, that is the number of
 *                  traces direction 0 has, independent of how many this
 *                  call covers.
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_i(\pm 1)\f$ of direction 0.
 * @param   in      Trace-mode workspace written by IPWRTPhysEdgeKernel.
 * @param   out     Volume field of this lane, accumulated onto.
 */
template <bool END_PTS_COLLOCATED0, typename TData>
NEK_DEVICE_INLINE static void AddEdgeN0ToVolKernel(
    const unsigned int ilane, const unsigned edg, const unsigned nedge,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const TData *NEK_RESTRICT nbasis, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (END_PTS_COLLOCATED0)
    {
        for (unsigned e = edg, cnt = 0; e < nedge; ++e)
        {
            const unsigned offset = e * (nm0 - 1u);
            for (unsigned q = 0; q < nm1; ++q, ++cnt)
            {
                out[warpsize * (nm0 * q + offset) + ilane] +=
                    in[warpsize * cnt + ilane];
            }
        }
    }
    else
    {
        for (unsigned e = edg, cnt = 0; e < nedge; ++e)
        {
            for (unsigned q = 0; q < nm1; ++q, ++cnt)
            {
                for (unsigned i = 0; i < nm0; ++i)
                {
                    out[warpsize * (nm0 * q + i) + ilane] +=
                        in[warpsize * cnt + ilane] * nbasis[i * tstride + e];
                }
            }
        }
    }
}

// kernels for edge 0 and 2 in quads
/**
 * @brief Leaf kernel, stage two in two dimensions: lift the results of a
 * direction-1 normal edge into the volume field.
 *
 * The direction-1 twin of AddEdgeN0ToVolKernel: here the edge carries
 * @p nm0 entries and the lift runs along direction 1, so under the
 * endpoint fast path the whole edge lands in the row
 * `j = e * (nm1 - 1)` and the general arm accumulates
 * `out[nm0 * q + i] += in[cnt + i] * h_q(\pm 1)`.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the
 *                            domain endpoints.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane   Lane index within the warp; selects the element.
 * @param   edg     Position of the first edge within direction 1.
 * @param   nedge   One past the position of the last edge.
 * @param   tstride Row stride of @p nbasis, that is the number of
 *                  traces direction 1 has (two for a quadrilateral, one
 *                  for the triangles), independent of how many this
 *                  call covers.
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   in      Trace-mode workspace written by IPWRTPhysEdgeKernel.
 * @param   out     Volume field of this lane, accumulated onto.
 */
template <bool END_PTS_COLLOCATED1, typename TData>
NEK_DEVICE_INLINE static void AddEdgeN1ToVolKernel(
    const unsigned int ilane, const unsigned edg, const unsigned nedge,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const TData *NEK_RESTRICT nbasis, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (END_PTS_COLLOCATED1)
    {
        for (unsigned e = edg, cnt = 0; e < nedge; ++e)
        {
            const unsigned offset = e * nm0 * (nm1 - 1u);
            for (unsigned q = 0; q < nm0; ++q, ++cnt)
            {
                out[warpsize * (offset + q) + ilane] +=
                    in[warpsize * cnt + ilane];
            }
        }
    }
    else // project to interior
    {
        for (unsigned e = edg, cnt = 0; e < nedge; ++e)
        {
            for (unsigned q = 0; q < nm1; ++q)
            {
                const TData val = nbasis[q * tstride + e];
                for (unsigned i = 0; i < nm0; ++i)
                {
                    out[warpsize * (nm0 * q + i) + ilane] +=
                        in[warpsize * (cnt + i) + ilane] * val;
                }
            }
            cnt += nm0;
        }
    }
}

// IPWRTPhysTrace kernels for edges 3 and 1
/**
 * @brief Glue for a direction-0 normal edge: integrate, then lift.
 *
 * Routes the direction's tables and counts into the two leaf stages. The
 * in-edge direction of a direction-0 normal is direction 1, so the
 * integration produces @p nm1 entries per edge into @p wsp, which the
 * lift then spreads along direction 0.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane   Lane index within the warp; selects the element.
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
 * @param   tjac    Trace Jacobian of this lane's warp.
 * @param   wsp     Trace-mode scratch, at least `nm1` entries per edge
 *                  covered, warp interleaved.
 * @param   in      Packed trace values of the edges covered.
 * @param   out     Volume field of this lane, accumulated onto.
 * @param   isCollocated  Trace points coincide with the volume points.
 */
template <bool END_PTS_COLLOCATED0, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceEdgeN0Kernel(
    const unsigned int ilane, const unsigned edg, const unsigned nedge,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const TData *NEK_RESTRICT nbasis, const unsigned tnq,
    const TData *NEK_RESTRICT tbasis, const TData *NEK_RESTRICT tw,
    const TData *NEK_RESTRICT tjac, TData *wsp, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated)
{
    IPWRTPhysEdgeKernel<DEFORMED>(ilane, edg, nedge, nm1, tnq, tbasis, tw, tjac,
                                  in, wsp, isCollocated);

    AddEdgeN0ToVolKernel<END_PTS_COLLOCATED0>(ilane, edg, nedge, tstride, nm0,
                                              nm1, nbasis, wsp, out);
}

// kernels for edge 0 and 2
/**
 * @brief Glue for a direction-1 normal edge: integrate, then lift.
 *
 * The twin of IPWRTPhysTraceEdgeN0Kernel one direction over. The in-edge
 * direction is now direction 0, so the integration produces @p nm0
 * entries per edge into @p wsp and the lift spreads them along
 * direction 1.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane   Lane index within the warp; selects the element.
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
 * @param   tjac    Trace Jacobian of this lane's warp.
 * @param   wsp     Trace-mode scratch, at least `nm0` entries per edge
 *                  covered, warp interleaved.
 * @param   in      Packed trace values of the edges covered.
 * @param   out     Volume field of this lane, accumulated onto.
 * @param   isCollocated  Trace points coincide with the volume points.
 */
template <bool END_PTS_COLLOCATED1, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceEdgeN1Kernel(
    const unsigned int ilane, const unsigned edg, const unsigned nedge,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const TData *NEK_RESTRICT nbasis, const unsigned tnq,
    const TData *NEK_RESTRICT tbasis, const TData *NEK_RESTRICT tw,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated)
{
    IPWRTPhysEdgeKernel<DEFORMED>(ilane, edg, nedge, nm0, tnq, tbasis, tw, tjac,
                                  in, wsp, isCollocated);

    AddEdgeN1ToVolKernel<END_PTS_COLLOCATED1>(ilane, edg, nedge, tstride, nm0,
                                              nm1, nbasis, wsp, out);
}

// -----------------------------------------------------------------------------
// 2D QUAD & TRI KERNELS
// -----------------------------------------------------------------------------
/**
 * @brief All edges of one two-dimensional element: zero the volume
 * field, then walk the packed traces.
 *
 * The bulk wrapper. It applies the direction-0 pair first, then advances
 * @p in by the pair's `2 * tnq00` values and @p tjac by the same number
 * of points when deformed or by two face slots when regular, and applies
 * the direction-1 traces: a pair, or a single edge when @c SHAPE_TYPE is
 * exactly @c Tri, whose direction-1 upper end is the collapsed vertex.
 * Quadrilateral and triangle run the identical leaf kernels; the
 * hypotenuse is simply the \f$\eta_0 = +1\f$ line of the collapsed square
 * and its edge Jacobian carries the physical metric of the slant.
 *
 * The @c endPtsCollocated flags are runtime booleans, so each call site is
 * written out twice to instantiate both arms of the leaf template.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri.
 * @tparam DEFORMED   Trace Jacobian varies point by point.
 * @tparam APPEND     Accumulate onto @p out instead of zeroing it first.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane           Lane index within the warp; selects the
 *                          element.
 * @param   nm0,nm1         Volume quadrature points per direction.
 * @param   nbasis0,nbasis1 eInterp tables \f$h_p(\pm 1)\f$ per
 *                          direction.
 * @param   tnq00,tnq10     Trace quadrature points of the direction-0
 *                          and direction-1 edges.
 * @param   tbasis0,tbasis1 eInterp tables to the trace points, per
 *                          direction.
 * @param   tw00,tw10       Trace quadrature weights per direction.
 * @param   tjac            Trace Jacobian of this lane's warp.
 * @param   wsp             Trace-mode scratch, `2 * max(nm0,nm1)`
 *                          entries, warp interleaved.
 * @param   in              Packed trace values of this element.
 * @param   out             Volume field of this lane.
 * @param   isCollocated0        Direction-0 trace points coincide with the
 *                          volume points.
 * @param   isCollocated1        Direction-1 trace points coincide with the
 *                          volume points.
 * @param   endPtsCollocated0     Direction-0 volume rule contains the domain
 *                          endpoints.
 * @param   endPtsCollocated1     Direction-1 volume rule contains the domain
 *                          endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool APPEND,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTPhysTrace2D(
    const unsigned int ilane, const unsigned nm0, const unsigned nm1,
    const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
    const unsigned tnq00, const unsigned tnq10,
    const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
    const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw10,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    const TData *NEK_RESTRICT in, TData *out, const bool isCollocated0,
    const bool isCollocated1, const bool endPtsCollocated0,
    const bool endPtsCollocated1)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (!APPEND)
    {
        // zero field (warp-interleaved layout)
        for (unsigned int j = 0; j < nm1; ++j)
        {
            for (unsigned int i = 0; i < nm0; ++i)
            {
                out[warpsize * (j * nm0 + i) + ilane] = 0.0;
            }
        }
    }

    if (endPtsCollocated0)
    {
        IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
            ilane, 0, 2, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac, wsp,
            in, out, isCollocated0);
    }
    else
    {
        IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
            ilane, 0, 2, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac, wsp,
            in, out, isCollocated0);
    }

    // The trace data and Jacobians are warp-interleaved, so stepping past the
    // two N0 edges advances by warpsize entries per point - a packed offset
    // only worked under DEVICEONHOST where warpsize is 1.
    unsigned offset  = warpsize * 2 * tnq00; // edge offset for following edge
    unsigned joffset = (DEFORMED) ? warpsize * 2 * tnq00 : warpsize * 2;

    // edge 0 in tris, edges 0 + 2 in quads
    constexpr unsigned ntrace1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

    if (endPtsCollocated1)
    {
        IPWRTPhysTraceEdgeN1Kernel<true, DEFORMED>(
            ilane, 0, ntrace1, ntrace1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
            tjac + joffset, wsp, in + offset, out, isCollocated1);
    }
    else
    {
        IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
            ilane, 0, ntrace1, ntrace1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
            tjac + joffset, wsp, in + offset, out, isCollocated1);
    }
}

/**
 * @brief Launch entry point for a block of quadrilaterals or
 * triangles: grid-stride over the elements and apply all their edges.
 *
 * Each iteration splits the global element index into warp and lane,
 * offsets @p in, @p out and @p tjac by the warp's `numData * warpsize`
 * blocks and the workspace by `2 * max(nm0,nm1) * warpsize`, then calls
 * IProductWRTPhysTrace2D, @p append selecting between its accumulating
 * and zeroing @c APPEND instantiations.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam DEFORMED         Trace Jacobian varies point by point.
 * @tparam TTraceSizeParameter2D    NonTemplatedTraceSizeParameter2D or
 *                          a TemplatedTraceSizeParameter2D
 *                          instantiation.
 * @tparam TData            Floating-point type of the field data.
 * @tparam TthreadBlock     Thread-block handle type.
 *
 * @param   sizeParam2D     Volume quadrature points per direction,
 *                          nm0() and nm1(), and the trace quadrature
 *                          points per direction, nq00() and nq10().
 * @param   nbasis0,nbasis1 eInterp tables \f$h_p(\pm 1)\f$ per
 *                          direction.
 * @param   nelmt           Elements in the block, including padding.
 * @param   numDataIn       Entries per element of @p in.
 * @param   numDataOut      Entries per element of @p out (`nm0 * nm1`).
 * @param   numDataJac      Entries per element of @p tjac: one slot per
 *                          edge when regular, one per trace point when
 *                          deformed.
 * @param   tbasis0,tbasis1 eInterp tables to the trace points.
 * @param   tw00,tw10       Trace quadrature weights per direction.
 * @param   tjac            Trace Jacobian of the block.
 * @param   wsp             Trace-mode scratch of the block.
 * @param   in              Packed trace values of the block.
 * @param   out             Volume-shaped result of the block.
 * @param   isCollocated0        Direction-0 trace points coincide with the
 *                          volume points.
 * @param   isCollocated1        Direction-1 trace points coincide with the
 *                          volume points.
 * @param   endPtsCollocated0     Direction-0 volume rule contains the domain
 *                          endpoints.
 * @param   endPtsCollocated1     Direction-1 volume rule contains the domain
 *                          endpoints.
 * @param   append          Accumulate onto @p out instead of zeroing it
 *                          first; selects between the two @c APPEND
 *                          instantiations of the inner wrapper.
 * @param   shmemptr        Dynamic shared memory; unused.
 * @param   threadBlock     Thread-block handle supplied by the launch
 *                          macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TTraceSizeParameter2D, typename TData, typename TthreadBlock>
NEK_DEVICE_KERNEL void IProductWRTPhysTraceKernelLauncher(
    const TTraceSizeParameter2D sizeParam2D, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const size_t nelmt,
    const unsigned int numDataIn, const unsigned int numDataOut,
    const unsigned int numDataJac, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT tw00,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tjac,
    TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in, TData *out,
    const bool isCollocated0, const bool isCollocated1,
    const bool endPtsCollocated0, const bool endPtsCollocated1,
    const bool append, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
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
    size_t e                        = getGlobalIdx<0>(threadBlock);
    const unsigned int c            = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr  = in + numDataIn * (nelmt * c + warpsize * iwarp);
        TData *outptr       = out + numDataOut * (nelmt * c + warpsize * iwarp);
        const TData *jacptr = tjac + numDataJac * warpsize * iwarp;
        TData *wspptr =
            wsp + (2u * IProductWRTPhysTraceEdgeModeBlockSize(sizeParam2D)) *
                      (nelmt * c + warpsize * iwarp);

        if (append)
        {
            IProductWRTPhysTrace2D<SHAPE_TYPE, DEFORMED, true>(
                ilane, nm0, nm1, nbasis0, nbasis1, tnq00, tnq10, tbasis0,
                tbasis1, tw00, tw10, jacptr, wspptr, inptr, outptr,
                isCollocated0, isCollocated1, endPtsCollocated0,
                endPtsCollocated1);
        }
        else
        {
            IProductWRTPhysTrace2D<SHAPE_TYPE, DEFORMED, false>(
                ilane, nm0, nm1, nbasis0, nbasis1, tnq00, tnq10, tbasis0,
                tbasis1, tw00, tw10, jacptr, wspptr, inptr, outptr,
                isCollocated0, isCollocated1, endPtsCollocated0,
                endPtsCollocated1);
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief One named edge of one two-dimensional element: turn the edge id
 * into a direction and a position, then integrate and lift.
 *
 * The single-trace worker, used when the solver drives one trace at a
 * time rather than the whole boundary. @p in and @p tjac already point at
 * that edge's block, so the position within the direction is expressed
 * through the [@p edg, @p nedge) window handed to the glue kernels. The
 * edge to (direction, position) map is `3->(0,0)`, `1->(0,1)`,
 * `0->(1,0)`, `2->(1,1)` for a quadrilateral and `2->(0,0)`,
 * `1->(0,1)`, `0->(1,0)` for a triangle, whose direction 1 carries a
 * single trace.
 *
 * @note An unrecognised @p edge is ignored silently: device code cannot
 * raise the host error the SerialAVX twin raises here.
 *
 * @note For the lower edge of a two-trace direction the window is
 * [0, 1), which is not the @c nbasis row stride: the lift kernels take
 * that separately, as @c tstride, so the general (non-@c endPtsCollocated)
 * arm reads the table with the direction's trace count either way.
 *
 * @tparam DEFORMED   Trace Jacobian varies point by point.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane           Lane index within the warp; selects the
 *                          element.
 * @param   edge            Nektar edge id of the trace to apply.
 * @param   nm0,nm1         Volume quadrature points per direction.
 * @param   nbasis0,nbasis1 eInterp tables \f$h_p(\pm 1)\f$ per
 *                          direction.
 * @param   tnq00,tnq10     Trace quadrature points per direction.
 * @param   tbasis0,tbasis1 eInterp tables to the trace points.
 * @param   tw00,tw10       Trace quadrature weights per direction.
 * @param   tjac            Trace Jacobian of this lane's warp, already
 *                          offset to this edge.
 * @param   wsp             Trace-mode scratch, `max(nm0,nm1)` entries.
 * @param   in              Packed trace values, already offset to this
 *                          edge.
 * @param   out             Volume field of this lane.
 * @param   isCollocated0        Direction-0 trace points coincide with the
 *                          volume points.
 * @param   isCollocated1        Direction-1 trace points coincide with the
 *                          volume points.
 * @param   endPtsCollocated0     Direction-0 volume rule contains the domain
 *                          endpoints.
 * @param   endPtsCollocated1     Direction-1 volume rule contains the domain
 *                          endpoints.
 * @param   shape           Quad, or any of the triangles.
 * @param   append          Accumulate onto @p out. The launcher always
 *                          passes true; nothing in the device per-trace
 *                          route zeroes @p out, so the caller must zero
 *                          the volume field before issuing the first
 *                          trace (see
 *                          IProductWRTPhysTraceOp::IProductWRTPhysTrace()).
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTPhysTraceEdge(
    const unsigned int ilane, const unsigned edge, const unsigned nm0,
    const unsigned nm1, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const unsigned tnq00,
    const unsigned tnq10, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT tw00,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tjac,
    TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated0, const bool isCollocated1,
    const bool endPtsCollocated0, const bool endPtsCollocated1,
    LibUtilities::ShapeType shape, const bool append)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if (!append)
    {
        // zero field (warp-interleaved layout). Note: the single-trace path is
        // normally driven with append == true (the caller pre-zeroes the field
        // and each trace accumulates), matching the AVX implementation.
        for (unsigned int j = 0; j < nm1; ++j)
        {
            for (unsigned int i = 0; i < nm0; ++i)
            {
                out[warpsize * (j * nm0 + i) + ilane] = 0.0;
            }
        }
    }

    if (shape == Nektar::LibUtilities::Quad)
    {
        switch (edge)
        {
            case 0:
                if (endPtsCollocated1)
                {
                    IPWRTPhysTraceEdgeN1Kernel<true, DEFORMED>(
                        ilane, 0, 1, 2, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
                        tjac, wsp, in, out, isCollocated1);
                }
                else
                {
                    IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
                        ilane, 0, 1, 2, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
                        tjac, wsp, in, out, isCollocated1);
                }
                break;
            case 1:
                if (endPtsCollocated0)
                {

                    IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                        ilane, 1, 2, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                        tjac, wsp, in, out, isCollocated0);
                }
                else
                {

                    IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                        ilane, 1, 2, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                        tjac, wsp, in, out, isCollocated0);
                }
                break;
            case 2:
                if (endPtsCollocated1)
                {
                    IPWRTPhysTraceEdgeN1Kernel<true, DEFORMED>(
                        ilane, 1, 2, 2, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
                        tjac, wsp, in, out, isCollocated1);
                }
                else
                {
                    IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
                        ilane, 1, 2, 2, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
                        tjac, wsp, in, out, isCollocated1);
                }
                break;
            case 3:
                if (endPtsCollocated0)
                {

                    IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                        ilane, 0, 1, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                        tjac, wsp, in, out, isCollocated0);
                }
                else
                {

                    IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                        ilane, 0, 1, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                        tjac, wsp, in, out, isCollocated0);
                }
                break;
            default:
                // Unrecognised edge input (cannot raise host error in device
                // code)
                break;
        }
    }
    else
    {
        switch (edge)
        {
            case 0:
                if (endPtsCollocated1)
                {
                    IPWRTPhysTraceEdgeN1Kernel<true, DEFORMED>(
                        ilane, 0, 1, 1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
                        tjac, wsp, in, out, isCollocated1);
                }
                else
                {
                    IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
                        ilane, 0, 1, 1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10,
                        tjac, wsp, in, out, isCollocated1);
                }
                break;
            case 1:
                if (endPtsCollocated0)
                {

                    IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                        ilane, 1, 2, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                        tjac, wsp, in, out, isCollocated0);
                }
                else
                {

                    IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                        ilane, 1, 2, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                        tjac, wsp, in, out, isCollocated0);
                }
                break;
            case 2:
                if (endPtsCollocated0)
                {

                    IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                        ilane, 0, 1, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                        tjac, wsp, in, out, isCollocated0);
                }
                else
                {

                    IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                        ilane, 0, 1, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00,
                        tjac, wsp, in, out, isCollocated0);
                }
                break;
            default:
                // Unrecognised edge input (cannot raise host error in device
                // code)
                break;
        }
    }
}

/**
 * @brief Launch entry point for one edge of a block of
 * two-dimensional elements.
 *
 * The single-trace counterpart of IProductWRTPhysTraceKernelLauncher. It
 * grid-strides over the block, offsets the pointers to the warp and the
 * workspace by `max(nm0,nm1) * warpsize`, one edge's worth rather than a
 * pair's, and calls IProductWRTPhysTraceEdge. The shape and the edge id
 * are runtime arguments, so one instantiation serves every trace.
 *
 * @tparam DEFORMED         Trace Jacobian varies point by point.
 * @tparam TTraceSizeParameter2D    NonTemplatedTraceSizeParameter2D or
 *                          a TemplatedTraceSizeParameter2D
 *                          instantiation.
 * @tparam TData            Floating-point type of the field data.
 * @tparam TthreadBlock     Thread-block handle type.
 *
 * @param   edge            Nektar edge id of the trace to apply.
 * @param   sizeParam2D     Volume quadrature points per direction,
 *                          nm0() and nm1(), and the trace quadrature
 *                          points per direction, nq00() and nq10().
 * @param   nbasis0,nbasis1 eInterp tables \f$h_p(\pm 1)\f$ per
 *                          direction.
 * @param   nelmt           Elements in the block, including padding.
 * @param   numDataIn       Entries per element of @p in.
 * @param   numDataOut      Entries per element of @p out (`nm0 * nm1`).
 * @param   numDataJac      Entries per element of @p tjac.
 * @param   tbasis0,tbasis1 eInterp tables to the trace points.
 * @param   tw00,tw10       Trace quadrature weights per direction.
 * @param   tjac            Trace Jacobian of the block, already offset
 *                          to this edge's slot or block.
 * @param   wsp             Trace-mode scratch of the block.
 * @param   in              Packed trace values of the block, already
 *                          offset to this edge.
 * @param   out             Volume-shaped result of the block.
 * @param   isCollocated0        Direction-0 trace points coincide with the
 *                          volume points.
 * @param   isCollocated1        Direction-1 trace points coincide with the
 *                          volume points.
 * @param   endPtsCollocated0     Direction-0 volume rule contains the domain
 *                          endpoints.
 * @param   endPtsCollocated1     Direction-1 volume rule contains the domain
 *                          endpoints.
 * @param   shape           Quad, or any of the triangles.
 * @param   append          Accumulate onto @p out; always true from the
 *                          launcher.
 * @param   shmemptr        Dynamic shared memory; unused.
 * @param   threadBlock     Thread-block handle supplied by the launch
 *                          macro.
 */
template <bool DEFORMED, typename TTraceSizeParameter2D, typename TData,
          typename TthreadBlock>
NEK_DEVICE_KERNEL void IProductWRTPhysTraceTraceKernelLauncher(
    const unsigned edge, const LibUtilities::ShapeType shape,
    const TTraceSizeParameter2D sizeParam2D, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const size_t nelmt,
    const unsigned int numDataIn, const unsigned int numDataOut,
    const unsigned int numDataJac, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT tw00,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tjac,
    TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated0, const bool isCollocated1,
    const bool endPtsCollocated0, const bool endPtsCollocated1,
    const bool append, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
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
    size_t e                        = getGlobalIdx<0>(threadBlock);
    const unsigned int c            = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr  = in + numDataIn * (nelmt * c + warpsize * iwarp);
        TData *outptr       = out + numDataOut * (nelmt * c + warpsize * iwarp);
        const TData *jacptr = tjac + numDataJac * warpsize * iwarp;
        TData *wspptr =
            wsp + IProductWRTPhysTraceEdgeModeBlockSize(sizeParam2D) *
                      (nelmt * c + warpsize * iwarp);

        IProductWRTPhysTraceEdge<DEFORMED>(
            ilane, edge, nm0, nm1, nbasis0, nbasis1, tnq00, tnq10, tbasis0,
            tbasis1, tw00, tw10, jacptr, wspptr, inptr, outptr, isCollocated0,
            isCollocated1, endPtsCollocated0, endPtsCollocated1, shape, append);

        e += getGlobalRange<0>(threadBlock);
    }
}

// ----------------------------------------------------------------------------
// DECOMPOSED 3D FACE KERNELS (AVX-style technique ported to device)
// ----------------------------------------------------------------------------

// Integrate a (pair of) face(s) over its quadrature points into nm0*nm1 trace
// modes. Warp-interleaved port of the SerialAVX IPWRTPhysFaceKernel. The
// result is written to "out" (the caller's trace-mode workspace); "wsp" is a
// scratch buffer used by the full (non-collocated) path.
/**
 * @brief Leaf kernel, stage one in three dimensions: integrate a face, or
 * a pair of faces, against the hat functions of the two in-face
 * directions.
 *
 * For every face in [@p fac, @p nface) it forms
 * \f[
 *   F_{pq} = \sum_j w_j\, h_q(\xi^{tr}_j) \sum_i w_i\, h_p(\xi^{tr}_i)\,
 *            J(\xi^{tr}_i, \xi^{tr}_j)\, \hat f(\xi^{tr}_i, \xi^{tr}_j),
 * \f]
 * writing `nm0 * nm1` entries per face at `q * nm0 + p`, face-major, into
 * the caller's trace-mode workspace @p out.
 *
 * Four arms cover the collocation cases: both directions collocated is a
 * pointwise multiply by the two weights and the Jacobian; one direction
 * collocated contracts the other only and scales by the collocated
 * direction's weight; neither collocated runs the two contractions in
 * sequence through @p wsp, which is why the caller must keep @p wsp and
 * @p out distinct.
 *
 * @note @p nm0 and @p nm1 are the in-face counts the glue level supplies,
 * `(nm1,nm2)` for an \f$N_0\f$ face, `(nm0,nm2)` for \f$N_1\f$ and
 * `(nm0,nm1)` for \f$N_2\f$, not the element's direction-0 and
 * direction-1 counts.
 *
 * @tparam DEFORMED   Trace Jacobian varies point by point.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane        Lane index within the warp; selects the element.
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
 * @param   tjac         Trace Jacobian of this lane's warp.
 * @param   wsp          Scratch for the fully general arm,
 *                       `nm0 * tnq1` entries, warp interleaved.
 * @param   in           Packed trace values of the faces covered.
 * @param   out          Trace-mode workspace, warp interleaved.
 * @param   isCollocated0  First in-face direction is collocated.
 * @param   isCollocated1  Second in-face direction is collocated.
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysFaceKernel(
    const unsigned int ilane, const unsigned fac, const unsigned nface,
    const unsigned nm0, const unsigned nm1, const unsigned tnq0,
    const unsigned tnq1, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT tw0,
    const TData *NEK_RESTRICT tw1, const TData *NEK_RESTRICT tjac,
    TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated0, const bool isCollocated1)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if (isCollocated0 && isCollocated1) // simply multiply by weights
    {
        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            for (unsigned j = 0; j < tnq1; ++j)
            {
                for (unsigned i = 0; i < tnq0; ++i, ++cnt)
                {
                    TData jac_val = DEFORMED
                                        ? tjac[warpsize * cnt + ilane]
                                        : tjac[warpsize * (f - fac) + ilane];
                    out[warpsize * cnt + ilane] =
                        in[warpsize * cnt + ilane] * jac_val * tw0[i] * tw1[j];
                }
            }
        }
    }
    else if (isCollocated0) // full inner product in dir 1
    {
        unsigned inOffset = 0, outOffset = 0;
        for (unsigned f = fac; f < nface; ++f)
        {
            for (unsigned i = 0; i < tnq0; ++i)
            {
                for (unsigned q = 0; q < nm1; ++q)
                {
                    unsigned qoffset = q * tnq1;
                    TData sum_q      = 0.0;
                    for (unsigned j = 0; j < tnq1; ++j)
                    {
                        unsigned joffset = j * tnq0 + inOffset;
                        TData jac_val =
                            DEFORMED ? tjac[warpsize * (joffset + i) + ilane]
                                     : tjac[warpsize * (f - fac) + ilane];
                        sum_q += in[warpsize * (joffset + i) + ilane] *
                                 jac_val * tbasis1[qoffset + j] * tw1[j];
                    }
                    out[warpsize * (outOffset + q * nm0 + i) + ilane] =
                        sum_q * tw0[i];
                }
            }
            inOffset += tnq0 * tnq1;
            outOffset += nm0 * nm1;
        }
    }
    else if (isCollocated1) // full inner product in dir 0
    {
        unsigned offset = 0;
        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            for (unsigned j = 0; j < tnq1; ++j)
            {
                unsigned joffset = j * tnq0 + offset;
                for (unsigned p = 0; p < nm0; ++p, ++cnt)
                {
                    unsigned poffset = p * tnq0;
                    TData sum_p      = 0.0;
                    for (unsigned i = 0; i < tnq0; ++i)
                    {
                        TData jac_val =
                            DEFORMED ? tjac[warpsize * (joffset + i) + ilane]
                                     : tjac[warpsize * (f - fac) + ilane];
                        sum_p += in[warpsize * (joffset + i) + ilane] *
                                 jac_val * tbasis0[poffset + i] * tw0[i];
                    }
                    out[warpsize * cnt + ilane] = sum_p * tw1[j];
                }
            }
            offset += tnq0 * tnq1;
        }
    }
    else // full inner product
    {
        unsigned inOffset = 0, outOffset = 0;
        for (unsigned f = fac; f < nface; ++f)
        {
            for (unsigned j = 0; j < tnq1; ++j)
            {
                unsigned joffset = j * tnq0 + inOffset;
                for (unsigned p = 0; p < nm0; ++p)
                {
                    unsigned poffset = p * tnq0;
                    TData sum_p      = 0.0;
                    for (unsigned i = 0; i < tnq0; ++i)
                    {
                        TData jac_val =
                            DEFORMED ? tjac[warpsize * (joffset + i) + ilane]
                                     : tjac[warpsize * (f - fac) + ilane];
                        sum_p += in[warpsize * (joffset + i) + ilane] *
                                 jac_val * tbasis0[poffset + i] * tw0[i];
                    }
                    wsp[warpsize * (j * nm0 + p) + ilane] = sum_p;
                }
            }

            for (unsigned p = 0; p < nm0; ++p)
            {
                for (unsigned q = 0; q < nm1; ++q)
                {
                    unsigned qoffset = q * tnq1;
                    TData sum_q      = 0.0;
                    for (unsigned j = 0; j < tnq1; ++j)
                    {
                        sum_q += wsp[warpsize * (j * nm0 + p) + ilane] *
                                 tbasis1[qoffset + j] * tw1[j];
                    }
                    out[warpsize * (outOffset + q * nm0 + p) + ilane] = sum_q;
                }
            }
            inOffset += tnq0 * tnq1;
            outOffset += nm0 * nm1;
        }
    }
}

// Project N0 (x-normal) face trace modes into the volume via nbasis in dir 0.
/**
 * @brief Leaf kernel, stage two in three dimensions: lift the results of
 * a direction-0 normal face into the volume field.
 *
 * Applies \f$\Lambda_{pqr} \mathrel{+}= h_p(\pm 1)\, F_{qr}\f$, the
 * volume field being indexed `r * nm0 * nm1 + q * nm0 + p`. Under
 * @p END_PTS_COLLOCATED0 the interpolation is a delta and the face lands
 * entirely in the plane `p = f * (nm0 - 1)`, @em f being the position of
 * the face within its direction rather than its index within this call.
 *
 * The general arm indexes `nbasis[p * tstride + f]` with the stride
 * passed in explicitly, so a direction carrying a single trace reads its
 * one-column table correctly.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane     Lane index within the warp; selects the element.
 * @param   fac       Position of the first face within direction 0.
 * @param   nface     One past the position of the last face.
 * @param   tstride   Row stride of @p nbasis, that is the number of
 *                    traces direction 0 has.
 * @param   nm0,nm1,nm2   Volume quadrature points per direction.
 * @param   nbasis    eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   in        Trace-mode workspace written by IPWRTPhysFaceKernel,
 *                    `nm1 * nm2` entries per face.
 * @param   out       Volume field of this lane, accumulated onto.
 */
template <bool END_PTS_COLLOCATED0, typename TData>
NEK_DEVICE_INLINE static void AddFaceN0ToVolKernel(
    const unsigned int ilane, const unsigned fac, const unsigned nface,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const TData *NEK_RESTRICT nbasis,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (END_PTS_COLLOCATED0)
    {
        unsigned nm01 = nm0 * nm1;
        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            unsigned offset = f * (nm0 - 1);
            for (unsigned r = 0; r < nm2; ++r)
            {
                unsigned outOffset = offset + nm01 * r;
                for (unsigned q = 0; q < nm1; ++q, ++cnt)
                {
                    out[warpsize * (outOffset + q * nm0) + ilane] +=
                        in[warpsize * cnt + ilane];
                }
            }
        }
    }
    else
    {
        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            for (unsigned r = 0, ocnt = 0; r < nm2; ++r)
            {
                for (unsigned q = 0; q < nm1; ++q, ++cnt)
                {
                    for (unsigned p = 0; p < nm0; ++p, ++ocnt)
                    {
                        out[warpsize * ocnt + ilane] +=
                            in[warpsize * cnt + ilane] *
                            nbasis[p * tstride + f];
                    }
                }
            }
        }
    }
}

// Project N1 (y-normal) face trace modes into the volume via nbasis in dir 1.
/**
 * @brief Leaf kernel, stage two in three dimensions: lift the results of
 * a direction-1 normal face into the volume field.
 *
 * The direction-1 twin of AddFaceN0ToVolKernel. The face carries
 * `nm0 * nm2` entries laid out face-local as `r * nm0 + p`, the endpoint
 * fast path writes the plane `q = f * (nm1 - 1)`, and the general arm
 * accumulates `h_q(\pm 1) * F_{pr}` over every @em q.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the
 *                            domain endpoints.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane     Lane index within the warp; selects the element.
 * @param   fac       Position of the first face within direction 1.
 * @param   nface     One past the position of the last face.
 * @param   tstride   Row stride of @p nbasis, that is the number of
 *                    traces direction 1 has (one for a tetrahedron).
 * @param   nm0,nm1,nm2   Volume quadrature points per direction.
 * @param   nbasis    eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   in        Trace-mode workspace written by IPWRTPhysFaceKernel,
 *                    `nm0 * nm2` entries per face.
 * @param   out       Volume field of this lane, accumulated onto.
 */
template <bool END_PTS_COLLOCATED1, typename TData>
NEK_DEVICE_INLINE static void AddFaceN1ToVolKernel(
    const unsigned int ilane, const unsigned fac, const unsigned nface,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const TData *NEK_RESTRICT nbasis,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (END_PTS_COLLOCATED1)
    {
        unsigned nm01 = nm0 * nm1;
        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            unsigned offset = f * nm0 * (nm1 - 1);
            for (unsigned r = 0; r < nm2; ++r)
            {
                unsigned outOffset = offset + nm01 * r;
                for (unsigned p = 0; p < nm0; ++p, ++cnt)
                {
                    out[warpsize * (outOffset + p) + ilane] +=
                        in[warpsize * cnt + ilane];
                }
            }
        }
    }
    else
    {
        for (unsigned f = fac; f < nface; ++f)
        {
            // input slabs are face-local F(p,r) of size nm0*nm2
            unsigned offset = (f - fac) * nm0 * nm2;
            for (unsigned r = 0, ocnt = 0; r < nm2; ++r)
            {
                unsigned cnt = r * nm0 + offset;
                for (unsigned q = 0; q < nm1; ++q)
                {
                    TData val = nbasis[q * tstride + f];
                    for (unsigned p = 0; p < nm0; ++p, ++ocnt)
                    {
                        out[warpsize * ocnt + ilane] +=
                            in[warpsize * (cnt + p) + ilane] * val;
                    }
                }
            }
        }
    }
}

// Project N2 (z-normal) face trace modes into the volume via nbasis in dir 2.
/**
 * @brief Leaf kernel, stage two in three dimensions: lift the results of
 * a direction-2 normal face into the volume field.
 *
 * The direction-2 twin of AddFaceN0ToVolKernel. The face carries
 * `nm0 * nm1` entries laid out face-local as `q * nm0 + p`, the endpoint
 * fast path writes the plane `r = f * (nm2 - 1)`, and the general arm
 * accumulates `h_r(\pm 1) * F_{pq}` over every @em r.
 *
 * @tparam END_PTS_COLLOCATED2  Direction-2 volume rule contains the
 *                            domain endpoints.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane     Lane index within the warp; selects the element.
 * @param   fac       Position of the first face within direction 2.
 * @param   nface     One past the position of the last face.
 * @param   tstride   Row stride of @p nbasis, that is the number of
 *                    traces direction 2 has (one for everything but a
 *                    hexahedron).
 * @param   nm0,nm1,nm2   Volume quadrature points per direction.
 * @param   nbasis    eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   in        Trace-mode workspace written by IPWRTPhysFaceKernel,
 *                    `nm0 * nm1` entries per face.
 * @param   out       Volume field of this lane, accumulated onto.
 */
template <bool END_PTS_COLLOCATED2, typename TData>
NEK_DEVICE_INLINE static void AddFaceN2ToVolKernel(
    const unsigned int ilane, const unsigned fac, const unsigned nface,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const TData *NEK_RESTRICT nbasis,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (END_PTS_COLLOCATED2)
    {
        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            unsigned offset = f * nm0 * nm1 * (nm2 - 1);
            for (unsigned q = 0; q < nm1; ++q)
            {
                for (unsigned p = 0; p < nm0; ++p, ++cnt)
                {
                    out[warpsize * offset + ilane] +=
                        in[warpsize * cnt + ilane];
                    ++offset;
                }
            }
        }
    }
    else
    {
        unsigned offset = 0;
        for (unsigned f = fac; f < nface; ++f)
        {
            for (unsigned r = 0, ocnt = 0; r < nm2; ++r)
            {
                TData val = nbasis[r * tstride + f];
                for (unsigned q = 0, cnt = 0; q < nm1; ++q)
                {
                    for (unsigned p = 0; p < nm0; ++p, ++ocnt, ++cnt)
                    {
                        out[warpsize * ocnt + ilane] +=
                            in[warpsize * (offset + cnt) + ilane] * val;
                    }
                }
            }
            offset += nm0 * nm1;
        }
    }
}

/**
 * @brief Glue for a direction-0 normal face: integrate, then lift.
 *
 * Routes the direction's tables and counts into the two leaf stages. The
 * in-face directions of an \f$N_0\f$ face are 1 and 2, so the face
 * integration is asked for `(nm1, nm2)` entries per face and the lift
 * spreads them along direction 0.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
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
 * @param   tjac        Trace Jacobian of this lane's warp.
 * @param   wsp         Trace-mode workspace, `nm1 * nm2` entries per
 *                      face covered.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of the faces covered.
 * @param   out         Volume field of this lane, accumulated onto.
 * @param   isCollocated0    First in-face direction is collocated.
 * @param   isCollocated1    Second in-face direction is collocated.
 */
template <bool END_PTS_COLLOCATED0, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceFaceN0Kernel(
    const unsigned int ilane, const unsigned fac, const unsigned nface,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const TData *NEK_RESTRICT nbasis, const unsigned tnq0,
    const unsigned tnq1, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT tw0,
    const TData *NEK_RESTRICT tw1, const TData *NEK_RESTRICT tjac,
    TData *NEK_RESTRICT wsp, TData *NEK_RESTRICT wsp1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated0, const bool isCollocated1)
{
    IPWRTPhysFaceKernel<DEFORMED>(ilane, fac, nface, nm1, nm2, tnq0, tnq1,
                                  tbasis0, tbasis1, tw0, tw1, tjac, wsp1, in,
                                  wsp, isCollocated0, isCollocated1);
    AddFaceN0ToVolKernel<END_PTS_COLLOCATED0>(ilane, fac, nface, tstride, nm0,
                                              nm1, nm2, nbasis, wsp, out);
}

/**
 * @brief Glue for a direction-1 normal face: integrate, then lift.
 *
 * The in-face directions of an \f$N_1\f$ face are 0 and 2, so the face
 * integration is asked for `(nm0, nm2)` entries per face and the lift
 * spreads them along direction 1.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
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
 * @param   tjac        Trace Jacobian of this lane's warp.
 * @param   wsp         Trace-mode workspace, `nm0 * nm2` entries per
 *                      face covered.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of the faces covered.
 * @param   out         Volume field of this lane, accumulated onto.
 * @param   isCollocated0    First in-face direction is collocated.
 * @param   isCollocated1    Second in-face direction is collocated.
 */
template <bool END_PTS_COLLOCATED1, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceFaceN1Kernel(
    const unsigned int ilane, const unsigned fac, const unsigned nface,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const TData *NEK_RESTRICT nbasis, const unsigned tnq0,
    const unsigned tnq1, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT tw0,
    const TData *NEK_RESTRICT tw1, const TData *NEK_RESTRICT tjac,
    TData *NEK_RESTRICT wsp, TData *NEK_RESTRICT wsp1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated0, const bool isCollocated1)
{
    IPWRTPhysFaceKernel<DEFORMED>(ilane, fac, nface, nm0, nm2, tnq0, tnq1,
                                  tbasis0, tbasis1, tw0, tw1, tjac, wsp1, in,
                                  wsp, isCollocated0, isCollocated1);
    AddFaceN1ToVolKernel<END_PTS_COLLOCATED1>(ilane, fac, nface, tstride, nm0,
                                              nm1, nm2, nbasis, wsp, out);
}

/**
 * @brief Glue for a direction-2 normal face: integrate, then lift.
 *
 * The in-face directions of an \f$N_2\f$ face are 0 and 1, so the face
 * integration is asked for `(nm0, nm1)` entries per face and the lift
 * spreads them along direction 2.
 *
 * @tparam END_PTS_COLLOCATED2  Direction-2 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam TData              Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
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
 * @param   tjac        Trace Jacobian of this lane's warp.
 * @param   wsp         Trace-mode workspace, `nm0 * nm1` entries per
 *                      face covered.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of the faces covered.
 * @param   out         Volume field of this lane, accumulated onto.
 * @param   isCollocated0    First in-face direction is collocated.
 * @param   isCollocated1    Second in-face direction is collocated.
 */
template <bool END_PTS_COLLOCATED2, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IPWRTPhysTraceFaceN2Kernel(
    const unsigned int ilane, const unsigned fac, const unsigned nface,
    const unsigned tstride, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const TData *NEK_RESTRICT nbasis, const unsigned tnq0,
    const unsigned tnq1, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT tw0,
    const TData *NEK_RESTRICT tw1, const TData *NEK_RESTRICT tjac,
    TData *NEK_RESTRICT wsp, TData *NEK_RESTRICT wsp1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated0, const bool isCollocated1)
{
    IPWRTPhysFaceKernel<DEFORMED>(ilane, fac, nface, nm0, nm1, tnq0, tnq1,
                                  tbasis0, tbasis1, tw0, tw1, tjac, wsp1, in,
                                  wsp, isCollocated0, isCollocated1);
    AddFaceN2ToVolKernel<END_PTS_COLLOCATED2>(ilane, fac, nface, tstride, nm0,
                                              nm1, nm2, nbasis, wsp, out);
}

// forward declaration: single-face worker (defined below)
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTPhysTraceFace(
    const unsigned int ilane, const unsigned face,
    const LibUtilities::ShapeType shape, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
    const unsigned tnq00, const unsigned tnq01, const unsigned tnq10,
    const unsigned tnq11, const unsigned tnq20, const unsigned tnq21,
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
    const bool isCollocated21, const bool endPtsCollocated0,
    const bool endPtsCollocated1, const bool endPtsCollocated2,
    const bool append);

/**
 * @brief All faces of one three-dimensional element: zero the volume
 * field, then walk the packed traces one face at a time.
 *
 * The bulk wrapper. The face order is the packed trace order, hex
 * `{4,2,1,3,0,5}`, prism and pyramid `{4,2,1,3,0}` and tetrahedron
 * `{3,2,1,0}`, with the matching normal directions; the input pointer
 * advances by that face's `tnq * tnq` values and the Jacobian pointer by
 * the same number of points when deformed or by one face slot when
 * regular, which is exactly the regular-Jacobian slot order.
 *
 * Each face is applied through the single-face worker with
 * `append = true`, this wrapper having done the zeroing itself. Faces are
 * processed one at a time rather than a pair at a time so that the
 * trace-mode workspace stays within its budget: batching a pair would
 * spill the second face's entries into the scratch its own general
 * integration is using. Pair batching is a possible follow-up with a
 * doubled workspace.
 *
 * @tparam SHAPE_TYPE Hex, Prism, Pyr, Tet, NodalTet or NodalPrism.
 * @tparam DEFORMED   Trace Jacobian varies point by point.
 * @tparam APPEND     Accumulate onto @p out instead of zeroing it first.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
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
 * @param   tjac        Trace Jacobian of this lane's warp.
 * @param   wsp         Trace-mode workspace of one face.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of this element.
 * @param   out         Volume field of this lane.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2   Volume
 *                      rule of that direction contains the domain
 *                      endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool APPEND,
          typename TData>
NEK_DEVICE_INLINE static void IProductWRTPhysTrace3DFace(
    const unsigned int ilane, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
    const unsigned tnq00, const unsigned tnq01, const unsigned tnq10,
    const unsigned tnq11, const unsigned tnq20, const unsigned tnq21,
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
    const bool isCollocated21, const bool endPtsCollocated0,
    const bool endPtsCollocated1, const bool endPtsCollocated2)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (!APPEND)
    {
        for (unsigned int k = 0; k < nm2; ++k)
        {
            for (unsigned int j = 0; j < nm1; ++j)
            {
                for (unsigned int i = 0; i < nm0; ++i)
                {
                    out[warpsize * (k * nm0 * nm1 + j * nm0 + i) + ilane] = 0.0;
                }
            }
        }
    }

    // Bulk apply as per-face applications of the single-face path
    // (pair-ordered packed traces; per-face jac slots in the same order).
    // Processing faces one at a time keeps the trace-mode buffer within its
    // wsp budget; batching a pair would spill the second face's modes into
    // the wsp1 scratch that its own non-collocated integration is using.
    constexpr bool IS_TET      = (SHAPE_TYPE == LibUtilities::Tet ||
                             SHAPE_TYPE == LibUtilities::NodalTet);
    constexpr unsigned nTraces = LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];
    // These permutations match the m_shapeTraceIDtoJacOff table in the
    // host class and must be kept in sync with it.
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
        IProductWRTPhysTraceFace<DEFORMED>(
            ilane, face, SHAPE_TYPE, nm0, nm1, nm2, nbasis0, nbasis1, nbasis2,
            tnq00, tnq01, tnq10, tnq11, tnq20, tnq21, tbasis00, tbasis01,
            tbasis10, tbasis11, tbasis20, tbasis21, tw00, tw01, tw10, tw11,
            tw20, tw21, tjac + joff * warpsize, wsp, wsp1,
            in + inoff * warpsize, out, isCollocated00, isCollocated01,
            isCollocated10, isCollocated11, isCollocated20, isCollocated21,
            endPtsCollocated0, endPtsCollocated1, endPtsCollocated2, true);
        inoff += fsz[dir];
        joff += DEFORMED ? fsz[dir] : 1;
    }
}

/**
 * @brief Launch entry point for a block of three-dimensional
 * elements: grid-stride over the elements and apply all their faces.
 *
 * The bulk three-dimensional path. It sizes the two workspace regions
 * once per launch, `wsp0Size` being the largest face's trace-mode count
 * and `wsp1Size` the largest intermediate of the general face
 * contraction, then per element offsets @p in, @p out and @p tjac to the
 * warp and splits the workspace into the two regions before calling
 * IProductWRTPhysTrace3DFace. @p append selects between the accumulating
 * and the zero-then-write instantiation of that wrapper.
 *
 * @tparam SHAPE_TYPE       Hex, Prism, Pyr, Tet, NodalTet or
 *                          NodalPrism.
 * @tparam DEFORMED         Trace Jacobian varies point by point.
 * @tparam TTraceSizeParameter3D    NonTemplatedTraceSizeParameter3D or
 *                          a TemplatedTraceSizeParameter3D
 *                          instantiation.
 * @tparam TData            Floating-point type of the field data.
 * @tparam TthreadBlock     Thread-block handle type.
 *
 * @param   sizeParam3D Volume quadrature points per direction, nm0(),
 *                      nm1() and nm2(), and the trace quadrature points
 *                      nq00() to nq21(), per normal direction and
 *                      tangential direction.
 * @param   nbasis0,nbasis1,nbasis2   eInterp tables \f$h_p(\pm 1)\f$
 *                      per direction.
 * @param   nelmt       Elements in the block, including padding.
 * @param   numDataIn   Entries per element of @p in, the sum over the
 *                      shape's faces of their trace point counts.
 * @param   numDataOut  Entries per element of @p out
 *                      (`nm0 * nm1 * nm2`).
 * @param   numDataJac  Entries per element of @p tjac: one slot per face
 *                      when regular, @p numDataIn when deformed.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points.
 * @param   tw00,tw01,tw10,tw11,tw20,tw21   Trace quadrature weights.
 * @param   tjac        Trace Jacobian of the block.
 * @param   wsp         Static workspace of the block, holding both
 *                      regions for every element.
 * @param   in          Packed trace values of the block.
 * @param   out         Volume-shaped result of the block.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2   Volume
 *                      rule of that direction contains the domain
 *                      endpoints.
 * @param   append      Accumulate onto @p out instead of zeroing it.
 * @param   shmemptr    Dynamic shared memory; unused.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TTraceSizeParameter3D, typename TData, typename TthreadBlock>
NEK_DEVICE_KERNEL void IProductWRTPhysTraceKernelLauncher(
    const TTraceSizeParameter3D sizeParam3D, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
    const size_t nelmt, const unsigned int numDataIn,
    const unsigned int numDataOut, const unsigned int numDataJac,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
    const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated00, const bool isCollocated01,
    const bool isCollocated10, const bool isCollocated11,
    const bool isCollocated20, const bool isCollocated21,
    const bool endPtsCollocated0, const bool endPtsCollocated1,
    const bool endPtsCollocated2, const bool append,
    [[maybe_unused]] unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    const size_t wsp0Size = IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D);
    const size_t wsp1Size = IProductWRTPhysTraceFaceScratchSize(sizeParam3D);

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr  = in + numDataIn * (nelmt * c + warpsize * iwarp);
        TData *outptr       = out + numDataOut * (nelmt * c + warpsize * iwarp);
        const TData *jacptr = tjac + numDataJac * warpsize * iwarp;
        TData *wspptr =
            wsp + (wsp0Size + wsp1Size) * (nelmt * c + warpsize * iwarp);
        TData *wsp1ptr = wspptr + wsp0Size * warpsize;

        if (append)
        {
            IProductWRTPhysTrace3DFace<SHAPE_TYPE, DEFORMED, true>(
                ilane, nm0, nm1, nm2, nbasis0, nbasis1, nbasis2, tnq00, tnq01,
                tnq10, tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10,
                tbasis11, tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20,
                tw21, jacptr, wspptr, wsp1ptr, inptr, outptr, isCollocated00,
                isCollocated01, isCollocated10, isCollocated11, isCollocated20,
                isCollocated21, endPtsCollocated0, endPtsCollocated1,
                endPtsCollocated2);
        }
        else
        {
            IProductWRTPhysTrace3DFace<SHAPE_TYPE, DEFORMED, false>(
                ilane, nm0, nm1, nm2, nbasis0, nbasis1, nbasis2, tnq00, tnq01,
                tnq10, tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10,
                tbasis11, tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20,
                tw21, jacptr, wspptr, wsp1ptr, inptr, outptr, isCollocated00,
                isCollocated01, isCollocated10, isCollocated11, isCollocated20,
                isCollocated21, endPtsCollocated0, endPtsCollocated1,
                endPtsCollocated2);
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief One named face of one three-dimensional element: turn the face
 * id into a direction and a position, then integrate and lift.
 *
 * The single-trace worker, used when the solver drives one trace at a
 * time, and also the per-face step of the bulk wrapper. @p in and
 * @p tjac already point at that face's block, so the position within the
 * direction is expressed through the [@p fac, @p nface) window handed to
 * the glue kernels.
 *
 * The face to (direction, position) map is `4->(0,0)`, `2->(0,1)`,
 * `1->(1,0)`, `3->(1,1)`, `0->(2,0)` and, for the hexahedron alone,
 * `5->(2,1)`. A tetrahedron differs in one place: its face 3 is the
 * lower direction-0 face, `3->(0,0)`, which is why case 3 branches on
 * the shape. It serves every three-dimensional shape, not only the
 * hexahedron of the neighbouring comment.
 *
 * The @c nbasis row strides are derived from the shape rather than
 * assumed: two in direction 0 always, two in direction 1 except for a
 * tetrahedron, and two in direction 2 only for a hexahedron.
 *
 * @note An unrecognised @p face is ignored silently: device code cannot
 * raise a host error.
 *
 * @tparam DEFORMED   Trace Jacobian varies point by point.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   face        Nektar face id of the trace to apply.
 * @param   shape       Hex, Prism, Pyr, Tet, NodalTet or NodalPrism.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis0,nbasis1,nbasis2   eInterp tables \f$h_p(\pm 1)\f$
 *                      per direction.
 * @param   tnq00,tnq01,tnq10,tnq11,tnq20,tnq21   Trace quadrature
 *                      points, per normal direction and tangential direction.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points.
 * @param   tw00,tw01,tw10,tw11,tw20,tw21   Trace quadrature weights.
 * @param   tjac        Trace Jacobian of this lane's warp, already
 *                      offset to this face.
 * @param   wsp         Trace-mode workspace of one face.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values, already offset to this face.
 * @param   out         Volume field of this lane.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2   Volume
 *                      rule of that direction contains the domain
 *                      endpoints.
 * @param   append      Accumulate onto @p out. Both the launcher and the
 *                      bulk wrapper pass true. The bulk wrapper has
 *                      zeroed the field itself only in its
 *                      @c APPEND == false instantiation; on the
 *                      single-trace route nothing in the device path
 *                      zeroes @p out and the caller must do so before
 *                      the first trace.
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTPhysTraceFace(
    const unsigned int ilane, const unsigned face,
    const LibUtilities::ShapeType shape, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
    const unsigned tnq00, const unsigned tnq01, const unsigned tnq10,
    const unsigned tnq11, const unsigned tnq20, const unsigned tnq21,
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
    const bool isCollocated21, const bool endPtsCollocated0,
    const bool endPtsCollocated1, const bool endPtsCollocated2,
    const bool append)
{
    // per-direction trace counts: the nbasis eInterp tables are stored
    // tightly with this stride (see BasisDataWarehouse eInterp)
    const bool tsIsTet =
        (shape == LibUtilities::Tet || shape == LibUtilities::NodalTet);
    const unsigned ts0              = 2u;
    const unsigned ts1              = tsIsTet ? 1u : 2u;
    const unsigned ts2              = (shape == LibUtilities::Hex) ? 2u : 1u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if (!append)
    {
        // zero field (warp-interleaved). As in 2D, the single-trace path is
        // normally driven with append == true.
        for (unsigned int k = 0; k < nm2; ++k)
        {
            for (unsigned int j = 0; j < nm1; ++j)
            {
                for (unsigned int i = 0; i < nm0; ++i)
                {
                    out[warpsize * (k * nm0 * nm1 + j * nm0 + i) + ilane] = 0.0;
                }
            }
        }
    }

    switch (face)
    {
        case 4:
            if (endPtsCollocated0)
            {
                IPWRTPhysTraceFaceN0Kernel<true, DEFORMED>(
                    ilane, 0, 1, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01,
                    tbasis00, tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01);
            }
            else
            {
                IPWRTPhysTraceFaceN0Kernel<false, DEFORMED>(
                    ilane, 0, 1, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01,
                    tbasis00, tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01);
            }
            break;
        case 2:
            if (endPtsCollocated0)
            {
                IPWRTPhysTraceFaceN0Kernel<true, DEFORMED>(
                    ilane, 1, 2, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01,
                    tbasis00, tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01);
            }
            else
            {
                IPWRTPhysTraceFaceN0Kernel<false, DEFORMED>(
                    ilane, 1, 2, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01,
                    tbasis00, tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01);
            }
            break;
        case 1:
            if (endPtsCollocated1)
            {
                IPWRTPhysTraceFaceN1Kernel<true, DEFORMED>(
                    ilane, 0, 1, ts1, nm0, nm1, nm2, nbasis1, tnq10, tnq11,
                    tbasis10, tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11);
            }
            else
            {
                IPWRTPhysTraceFaceN1Kernel<false, DEFORMED>(
                    ilane, 0, 1, ts1, nm0, nm1, nm2, nbasis1, tnq10, tnq11,
                    tbasis10, tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11);
            }
            break;
        case 3:
            if (shape == LibUtilities::Tet || shape == LibUtilities::NodalTet)
            {
                if (endPtsCollocated0)
                {
                    IPWRTPhysTraceFaceN0Kernel<true, DEFORMED>(
                        ilane, 0, 1, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01,
                        tbasis00, tbasis01, tw00, tw01, tjac, wsp, wsp1, in,
                        out, isCollocated00, isCollocated01);
                }
                else
                {
                    IPWRTPhysTraceFaceN0Kernel<false, DEFORMED>(
                        ilane, 0, 1, ts0, nm0, nm1, nm2, nbasis0, tnq00, tnq01,
                        tbasis00, tbasis01, tw00, tw01, tjac, wsp, wsp1, in,
                        out, isCollocated00, isCollocated01);
                }
            }
            else if (endPtsCollocated1)
            {
                IPWRTPhysTraceFaceN1Kernel<true, DEFORMED>(
                    ilane, 1, 2, ts1, nm0, nm1, nm2, nbasis1, tnq10, tnq11,
                    tbasis10, tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11);
            }
            else
            {
                IPWRTPhysTraceFaceN1Kernel<false, DEFORMED>(
                    ilane, 1, 2, ts1, nm0, nm1, nm2, nbasis1, tnq10, tnq11,
                    tbasis10, tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11);
            }
            break;
        case 0:
            if (endPtsCollocated2)
            {
                IPWRTPhysTraceFaceN2Kernel<true, DEFORMED>(
                    ilane, 0, 1, ts2, nm0, nm1, nm2, nbasis2, tnq20, tnq21,
                    tbasis20, tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21);
            }
            else
            {
                IPWRTPhysTraceFaceN2Kernel<false, DEFORMED>(
                    ilane, 0, 1, ts2, nm0, nm1, nm2, nbasis2, tnq20, tnq21,
                    tbasis20, tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21);
            }
            break;
        case 5:
            if (endPtsCollocated2)
            {
                IPWRTPhysTraceFaceN2Kernel<true, DEFORMED>(
                    ilane, 1, 2, ts2, nm0, nm1, nm2, nbasis2, tnq20, tnq21,
                    tbasis20, tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21);
            }
            else
            {
                IPWRTPhysTraceFaceN2Kernel<false, DEFORMED>(
                    ilane, 1, 2, ts2, nm0, nm1, nm2, nbasis2, tnq20, tnq21,
                    tbasis20, tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21);
            }
            break;
        default:
            // Unrecognised face input (cannot raise host error in device code)
            break;
    }
}

/**
 * @brief Launch entry point for one face of a block of
 * three-dimensional elements.
 *
 * The single-trace counterpart of IProductWRTPhysTraceKernelLauncher and
 * the three-dimensional counterpart of IProductWRTPhysTraceTraceKernelLauncher.
 * It sizes and splits the workspace exactly as the bulk kernel does,
 * grid-strides over the block and calls IProductWRTPhysTraceFace. The
 * shape and the face id are runtime arguments, so one instantiation
 * serves every trace of every three-dimensional shape.
 *
 * @tparam DEFORMED         Trace Jacobian varies point by point.
 * @tparam TTraceSizeParameter3D    NonTemplatedTraceSizeParameter3D or
 *                          a TemplatedTraceSizeParameter3D
 *                          instantiation.
 * @tparam TData            Floating-point type of the field data.
 * @tparam TthreadBlock     Thread-block handle type.
 *
 * @param   face        Nektar face id of the trace to apply.
 * @param   shape       Hex, Prism, Pyr, Tet, NodalTet or NodalPrism.
 * @param   sizeParam3D Volume quadrature points per direction, nm0(),
 *                      nm1() and nm2(), and the trace quadrature points
 *                      nq00() to nq21(), per normal direction and
 *                      tangential direction.
 * @param   nbasis0,nbasis1,nbasis2   eInterp tables \f$h_p(\pm 1)\f$
 *                      per direction.
 * @param   nelmt       Elements in the block, including padding.
 * @param   numDataIn   Entries per element of @p in.
 * @param   numDataOut  Entries per element of @p out.
 * @param   numDataJac  Entries per element of @p tjac.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points.
 * @param   tw00,tw01,tw10,tw11,tw20,tw21   Trace quadrature weights.
 * @param   tjac        Trace Jacobian of the block, already offset to
 *                      this face's slot or block.
 * @param   wsp         Static workspace of the block.
 * @param   in          Packed trace values of the block, already offset
 *                      to this face.
 * @param   out         Volume-shaped result of the block.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2   Volume
 *                      rule of that direction contains the domain
 *                      endpoints.
 * @param   append      Accumulate onto @p out; always true from the
 *                      launcher.
 * @param   shmemptr    Dynamic shared memory; unused.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <bool DEFORMED, typename TTraceSizeParameter3D, typename TData,
          typename TthreadBlock>
NEK_DEVICE_KERNEL void IProductWRTPhysTraceTraceKernelLauncher(
    const unsigned face, const LibUtilities::ShapeType shape,
    const TTraceSizeParameter3D sizeParam3D, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
    const size_t nelmt, const unsigned int numDataIn,
    const unsigned int numDataOut, const unsigned int numDataJac,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
    const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
    const TData *NEK_RESTRICT tjac, TData *NEK_RESTRICT wsp,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated00, const bool isCollocated01,
    const bool isCollocated10, const bool isCollocated11,
    const bool isCollocated20, const bool isCollocated21,
    const bool endPtsCollocated0, const bool endPtsCollocated1,
    const bool endPtsCollocated2, const bool append,
    [[maybe_unused]] unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    const size_t wsp0Size = IProductWRTPhysTraceFaceModeBlockSize(sizeParam3D);
    const size_t wsp1Size = IProductWRTPhysTraceFaceScratchSize(sizeParam3D);

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr  = in + numDataIn * (nelmt * c + warpsize * iwarp);
        TData *outptr       = out + numDataOut * (nelmt * c + warpsize * iwarp);
        const TData *jacptr = tjac + numDataJac * warpsize * iwarp;
        TData *wspptr =
            wsp + (wsp0Size + wsp1Size) * (nelmt * c + warpsize * iwarp);
        TData *wsp1ptr = wspptr + wsp0Size * warpsize;

        IProductWRTPhysTraceFace<DEFORMED>(
            ilane, face, shape, nm0, nm1, nm2, nbasis0, nbasis1, nbasis2, tnq00,
            tnq01, tnq10, tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10,
            tbasis11, tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20, tw21,
            jacptr, wspptr, wsp1ptr, inptr, outptr, isCollocated00,
            isCollocated01, isCollocated10, isCollocated11, isCollocated20,
            isCollocated21, endPtsCollocated0, endPtsCollocated1,
            endPtsCollocated2, append);

        e += getGlobalRange<0>(threadBlock);
    }
}

} // namespace Nektar::Operators::detail
