///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractDeviceGenericKernels.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysTraceExtractDeviceGenericKernels.hpp
 * @brief Device kernels of the trace extraction: one element per warp
 * lane.
 *
 * @details
 * PhysTraceExtract is the adjoint of IProductWRTPhysTrace. It samples a
 * field held at an element's volume quadrature points on the quadrature
 * points of that element's traces. Expanding the volume field in the
 * cardinal (hat) basis of its own grid,
 * \f$u = \sum_{pq} u_{pq}\, h_p(\xi_0)\, h_q(\xi_1)\f$, its values on the
 * direction-1 trace pair are
 * \f[
 *   u(\xi^{tr}_{0i}, \pm 1) \;=\; \sum_q h_q(\pm 1) \sum_p u_{pq}\,
 *   h_p(\xi^{tr}_{0i}) ,
 * \f]
 * so an extraction is two contractions: one against \f$h_q(\pm 1)\f$ in
 * the normal direction, which picks out the trace, and one against
 * \f$h_p(\xi^{tr})\f$ per tangential direction, which resamples that
 * trace onto its own quadrature rule. The two commute, and every kernel
 * here takes the normal direction first. They use the same two eInterp
 * tables as IProductWRTPhysTrace, applied here untransposed.
 *
 * Nothing in this file accumulates: every output is written rather than
 * added to, so there is no APPEND parameter and no obligation on the
 * caller to zero anything first.
 *
 * @section phystraceextract_dev_naming Naming
 *
 * @c nm0, @c nm1 and @c nm2 hold @c GetNumPoints, that is volume
 * quadrature counts and @em not mode counts. The three-dimensional leaf
 * kernels spell those same three arguments @c nq0, @c nq1 and @c nq2 in
 * their parameter lists, but their callers pass the volume counts: read
 * them as @c nm.
 *
 * @c ntbasis0, @c ntbasis1 and @c ntbasis2 are the @c eInterp tables
 * \f$h_p(\pm 1)\f$ from the volume grid of a direction to that
 * direction's trace positions, and the @c tbasis (spelled @c basis in
 * the two-dimensional bulk core) are the @c eInterp tables
 * \f$h_p(\xi^{tr}_i)\f$ from a volume grid to the trace quadrature
 * points. Neither is an expansion basis evaluated anywhere. Both are
 * stored one row per volume point and one column per output point, so
 * @c tbasis is indexed `tbasis[p * tnq + i]` and @c ntbasis
 * `ntbasis[p * tstride + f]` with @c tstride the number of traces that
 * direction has: two for every direction of a hexahedron, one for a
 * collapsed direction (direction 1 of a tetrahedron, direction 2 of
 * prism, pyramid and tetrahedron).
 *
 * The two-digit suffixes name a normal direction and a tangential direction,
 * never a volume direction: @c tbasis00, @c tbasis01, @c nq00, @c nq01,
 * @c isCollocated00 and @c isCollocated01 all belong to the faces whose
 * normal is direction 0, and the volume directions they interpolate
 * @em from are the two other than 0. A two-dimensional trace has a
 * single tangential direction, so the second digit is always 0 where it
 * survives at all (@c nq00 and @c nq10, @c tbasis00 and @c tbasis10) and
 * the bulk core drops it (@c nq0, @c nq1, @c basis0, @c basis1).
 *
 * @section phystraceextract_dev_layout Layout contracts
 *
 * - Warp interleaving. Element \f$e = (i_{warp}, i_{lane})\f$ stores
 *   entry @em n at `buf[warpsize * n + ilane]` within its warp block,
 *   and warp blocks stride by `numData * warpsize`. The launchers
 *   grid-stride over the elements and offset every pointer to the warp
 *   before calling a per-element worker.
 * - Trace packing. The extracted output holds the traces in pair order:
 *   the \f$N_0\f$ pair first, then the \f$N_1\f$ pair or single, then
 *   \f$N_2\f$, face-major within a pair.
 * - Within one trace the entries run with the lower of the two
 *   remaining volume directions fastest: an \f$N_0\f$ face is
 *   direction-1 fastest and direction-2 slowest, an \f$N_1\f$ face
 *   direction-0 fastest and direction-2 slowest, an \f$N_2\f$ face
 *   direction-0 fastest and direction-1 slowest.
 * - Face to (direction, position). Hex, prism and pyramid `4->(0,0)`,
 *   `2->(0,1)`, `1->(1,0)`, `3->(1,1)`, `0->(2,0)`, and for the
 *   hexahedron alone `5->(2,1)`; tetrahedron `3->(0,0)`, `2->(0,1)`,
 *   `1->(1,0)`, `0->(2,0)`. In two dimensions, quadrilateral
 *   `3->(0,0)`, `1->(0,1)`, `0->(1,0)`, `2->(1,1)`; triangle
 *   `2->(0,0)`, `1->(0,1)`, `0->(1,0)`. These are exactly the
 *   GetTraceFaceDispatch and GetTraceEdgeDispatch tables below.
 * - The \f$N_a\f$ trace block spans the two volume quadrature counts
 *   other than \f$Q_a\f$, so the glue level routes purely by direction:
 *   \f$N_0\f$ uses `(nm1,nm2)`, \f$N_1\f$ uses `(nm0,nm2)` and
 *   \f$N_2\f$ uses `(nm0,nm1)`.
 *
 * @section phystraceextract_dev_fast Fast paths
 *
 * Two flags select them, both decided at setup. @c END_PTS_COLLOCATED says
 * the volume rule of the normal direction contains the domain endpoints,
 * so \f$h_p(\pm 1)\f$ is a Kronecker delta and the normal stage
 * degenerates to selecting a boundary plane; it arrives as a runtime
 * boolean, which the DispatchPhysExtractFaceN{0,1,2}KernelTrace3D
 * helpers turn into the template argument the three-dimensional leaves
 * branch on. @c isCollocated says the trace quadrature points of a
 * tangential direction coincide with the volume points of the direction they
 * come from, so the interpolation table is the identity and that
 * contraction degenerates to a copy. It stays a runtime boolean
 * throughout.
 *
 * @section phystraceextract_dev_layer Layering
 *
 * The names are not a mirror of the IProductWRTPhysTrace family, and the
 * routing differs too: a face or edge id becomes a normal direction and
 * a position through the GetTraceFaceDispatch and GetTraceEdgeDispatch
 * tables rather than through a switch inside a worker. In three
 * dimensions:
 * - launchers (PhysTraceExtractKernelLauncher for the bulk,
 *   PhysTraceExtractTraceKernelLauncher for one trace), which own the
 *   grid-stride loop, the warp mapping and the workspace split;
 * - PhysTraceExtract3DSumFacKernelCore, which walks all of an element's
 *   traces direction by direction and advances the output offset, or
 *   PhysTraceFaceExtractKernel, which routes one named face;
 * - DispatchPhysExtractFaceN{0,1,2}KernelTrace3D, which fix the
 *   endpoint template argument;
 * - PhysExtractFaceN{0,1,2}KernelTrace3D, the per-direction glue, which
 *   either writes straight through the normal stage or runs the normal
 *   stage into a workspace and follows with the tangential one;
 * - the leaves PhysExtractEndFacesN{0,1,2}KernelTrace3D (normal stage)
 *   and PhysInterpFaceKernelTrace (tangential stage).
 *
 * In two dimensions PhysTraceExtract2DSumFacKernelCore and
 * PhysTraceExtractEdgeKernel play the same two roles, and both stages
 * live in the one kernel BwdTransQuadSumFacKernelTrace, templated on
 * the normal direction. In one dimension the trace is a point: the bulk
 * kernel BwdTransSegSumFacKernelTrace writes both vertex values and
 * PhysTraceExtractTraceKernelLauncher writes one, each inlining the whole
 * operation.
 *
 * @section phystraceextract_dev_state State of the paths
 *
 * Not everything here is live, and not everything live is correct:
 * - These kernels index with @c warpSize throughout and map one element
 *   to one lane, so they carry no implementation tag: there is one
 *   trace-extraction algorithm and the block operator registers it
 *   under @c Generic. A @c SumFacTOP block operator's width-one
 *   interleave would contradict the indexing here, and no template
 *   parameter now offers to accept it.
 * - The general (non-endpoint-collocated) arm of the three-dimensional
 *   normal stage carries the @c ntbasis row stride as an explicit
 *   @c tstride argument, separate from the face-loop bound. The two
 *   coincide only when the window spans the whole direction, which the
 *   bulk route does and the per-trace route does not, so one cannot
 *   stand in for the other. Both entry points take the stride from
 *   @c ShapeTypeNumTraceInDir. The extraction unit tests cannot check
 *   this arm: their reference is the legacy trace gather, which reads a
 *   boundary quadrature plane and so has nothing to say where the end
 *   points are not quadrature points. The divergence tests on a Gauss
 *   mesh are what cover it.
 * - The BwdTrans names are inherited only. The @c Trace kernels are
 *   self-contained copies that call nothing from BwdTrans, and the
 *   include of BwdTransDeviceSumFacKernels.hpp below is a leftover of
 *   that lineage: what this file actually takes from it are the
 *   ShapeType and Spaces headers it pulls in transitively, the latter
 *   carrying the backend device API.
 *
 * No kernel in this file uses shared memory; every @c shmemptr parameter
 * is present for the launch macro's signature only. The whole file is
 * compiled only for the device back-ends (NEKTAR_ENABLE_DEVICE together
 * with DEVICE_COMPILE_ONLY).
 *
 * @see PhysTraceExtractDeviceGeneric.hpp for the dispatch and the
 * launches, PhysTraceExtractSerialAVXGenericKernels.hpp for the SIMD
 * counterparts of the same decomposition, and
 * IProductWRTPhysTraceDeviceGenericKernels.hpp for the adjoint operation,
 * which applies the same tables transposed.
 */

#pragma once

#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacKernels.hpp"

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

// Helper functions

/**
 * @brief Per-element workspace of the two-dimensional paths: the edge
 * block the normal-direction stage writes, before the tangential
 * interpolation reads it.
 *
 * `nedge` values for each of `max(nm0, nm1)` in-trace positions. The
 * bound is written for an edge pair, `nedge == 2`, which is what the
 * bulk path fills; a per-trace launch fills a single edge and uses the
 * same expression. Host and kernel both size against this, so the two
 * cannot drift apart.
 */
template <
    typename TTraceSizeParameter2D,
    std::enable_if_t<IsTraceSizeParameter2D_v<TTraceSizeParameter2D>, bool>
        Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int PhysTraceExtractBlockSize(
    const TTraceSizeParameter2D sizeParam2D)
{
    return 2u * std::max(sizeParam2D.nm0(), sizeParam2D.nm1());
}

/**
 * @brief Per-element bound covering both three-dimensional workspace
 * regions.
 *
 * The largest of the three face blocks `2 * nm1 * nm2`, `2 * nm0 * nm2`
 * and `2 * nm0 * nm1`, and of the three partially interpolated blocks
 * `2 * nm2 * nq00`, `2 * nm2 * nq10` and `2 * nm1 * nq20`, each written
 * for a face pair. The kernels give the first region twice this and the
 * scratch of PhysInterpFaceKernelTrace one, which is why
 * PhysTraceExtractWorkSpaceSize() asks for three.
 */
template <
    typename TTraceSizeParameter3D,
    std::enable_if_t<IsTraceSizeParameter3D_v<TTraceSizeParameter3D>, bool>
        Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int PhysTraceExtractBlockSize(
    const TTraceSizeParameter3D sizeParam3D)
{
    const unsigned int nm0  = sizeParam3D.nm0();
    const unsigned int nm1  = sizeParam3D.nm1();
    const unsigned int nm2  = sizeParam3D.nm2();
    const unsigned int nq00 = sizeParam3D.nq00();
    const unsigned int nq10 = sizeParam3D.nq10();
    const unsigned int nq20 = sizeParam3D.nq20();

    return std::max({2u * nm1 * nm2, 2u * nm0 * nm2, 2u * nm0 * nm1,
                     2u * nm2 * nq00, 2u * nm2 * nq10, 2u * nm1 * nq20});
}

/**
 * @brief Workspace of a whole block, per component: none, a segment's
 * traces being its end vertices with nothing to interpolate
 * tangentially.
 */
template <
    typename TTraceSizeParameter1D,
    std::enable_if_t<IsTraceSizeParameter1D_v<TTraceSizeParameter1D>, bool>
        Enable = true>
inline constexpr size_t PhysTraceExtractWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TTraceSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Workspace of a whole block, per component: one edge block per
/// element, which PhysTraceExtractBlockSize() sizes.
template <
    typename TTraceSizeParameter2D,
    std::enable_if_t<IsTraceSizeParameter2D_v<TTraceSizeParameter2D>, bool>
        Enable = true>
inline constexpr size_t PhysTraceExtractWorkSpaceSize(
    const size_t nelmt, const TTraceSizeParameter2D sizeParam2D)
{
    return PhysTraceExtractBlockSize(sizeParam2D) * nelmt;
}

/// @brief Workspace of a whole block, per component: three
/// PhysTraceExtractBlockSize() per element, two for the face block the
/// normal stage writes and one for the tangential scratch.
template <
    typename TTraceSizeParameter3D,
    std::enable_if_t<IsTraceSizeParameter3D_v<TTraceSizeParameter3D>, bool>
        Enable = true>
inline constexpr size_t PhysTraceExtractWorkSpaceSize(
    const size_t nelmt, const TTraceSizeParameter3D sizeParam3D)
{
    return 3u * PhysTraceExtractBlockSize(sizeParam3D) * nelmt;
}

/**
 * @brief Where a named trace sits: its normal direction and its position
 * within that direction, the latter as a half-open window.
 *
 * Filled by GetTraceEdgeDispatch and GetTraceFaceDispatch, its only two
 * producers, and consumed by PhysTraceExtractEdgeKernel and
 * PhysTraceFaceExtractKernel. Both tables always describe exactly one
 * trace, so @c nfac is always `fac + 1` and the loops driven by this
 * struct run a single iteration.
 */
struct TraceExtractDispatch
{
    /// Direction whose normal this trace carries: 0, 1 or 2.
    unsigned int normalDir;
    /// Position within that direction, 0 for the lower trace and 1 for
    /// the upper one; also the column of the @c ntbasis table to read.
    unsigned int fac;
    /// One past #fac, the loop bound the leaf kernels take. It is not
    /// the @c ntbasis row stride: every window here is one trace wide,
    /// so the three-dimensional leaves take that stride separately, as
    /// @c tstride; see PhysExtractEndFacesN0KernelTrace3D.
    unsigned int nfac;
};

/**
 * @brief Two-dimensional trace table: turn an edge id into a normal
 * direction and a one-edge window.
 *
 * Quadrilateral `3->(0,0)`, `1->(0,1)`, `0->(1,0)`, `2->(1,1)`.
 * Triangle `2->(0,0)`, `1->(0,1)` (the hypotenuse, which in the
 * collapsed frame is simply the \f$\eta_0 = +1\f$ line of the square)
 * and `0->(1,0)`, the single direction-1 edge, direction 1 having
 * collapsed to a vertex at \f$\eta_1 = +1\f$.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri; every other shape returns
 *                    false.
 *
 * @param   edge      Nektar edge id of the trace wanted.
 * @param   dispatch  Written only when the lookup succeeds.
 *
 * @return False for an unhandled shape or an out-of-range edge, which
 * the callers turn into a silent no-op: device code cannot raise a host
 * error.
 */
template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_DEVICE_INLINE constexpr bool GetTraceEdgeDispatch(
    const unsigned int edge, TraceExtractDispatch &dispatch)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        switch (edge)
        {
            case 0:
                dispatch = {1u, 0u, 1u};
                return true;
            case 1:
                dispatch = {0u, 1u, 2u};
                return true;
            case 2:
                dispatch = {0u, 0u, 1u};
                return true;
            default:
                return false;
        }
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        switch (edge)
        {
            case 0:
                dispatch = {1u, 0u, 1u};
                return true;
            case 1:
                dispatch = {0u, 1u, 2u};
                return true;
            case 2:
                dispatch = {1u, 1u, 2u};
                return true;
            case 3:
                dispatch = {0u, 0u, 1u};
                return true;
            default:
                return false;
        }
    }
    else
    {
        return false;
    }
}

/**
 * @brief Three-dimensional trace table: turn a face id into a normal
 * direction and a one-face window.
 *
 * Hexahedron, prism and pyramid `4->(0,0)`, `2->(0,1)`, `1->(1,0)`,
 * `3->(1,1)`, `0->(2,0)`, and for the hexahedron alone `5->(2,1)`. A
 * tetrahedron differs in one place, its face 3 being the lower
 * direction-0 face, `3->(0,0)`, with `1->(1,0)` and `0->(2,0)` the
 * single faces of its two collapsed directions.
 *
 * @tparam SHAPE_TYPE Hex, Prism, NodalPrism, Pyr, Tet or NodalTet; every
 *                    other shape returns false. The nodal entries are
 *                    accepted here even though the block operator
 *                    already instantiates the per-trace route under the
 *                    parent enumerator.
 *
 * @param   face      Nektar face id of the trace wanted.
 * @param   dispatch  Written only when the lookup succeeds.
 *
 * @return False for an unhandled shape or an out-of-range face, which
 * the callers turn into a silent no-op: device code cannot raise a host
 * error.
 */
template <LibUtilities::ShapeType SHAPE_TYPE>
NEK_DEVICE_INLINE constexpr bool GetTraceFaceDispatch(
    const unsigned int face, TraceExtractDispatch &dispatch)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                  SHAPE_TYPE == LibUtilities::NodalTet)
    {
        switch (face)
        {
            case 0:
                dispatch = {2u, 0u, 1u};
                return true;
            case 1:
                dispatch = {1u, 0u, 1u};
                return true;
            case 2:
                dispatch = {0u, 1u, 2u};
                return true;
            case 3:
                dispatch = {0u, 0u, 1u};
                return true;
            default:
                return false;
        }
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism ||
                       SHAPE_TYPE == LibUtilities::Pyr)
    {
        switch (face)
        {
            case 0:
                dispatch = {2u, 0u, 1u};
                return true;
            case 1:
                dispatch = {1u, 0u, 1u};
                return true;
            case 2:
                dispatch = {0u, 1u, 2u};
                return true;
            case 3:
                dispatch = {1u, 1u, 2u};
                return true;
            case 4:
                dispatch = {0u, 0u, 1u};
                return true;
            default:
                return false;
        }
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        switch (face)
        {
            case 0:
                dispatch = {2u, 0u, 1u};
                return true;
            case 1:
                dispatch = {1u, 0u, 1u};
                return true;
            case 2:
                dispatch = {0u, 1u, 2u};
                return true;
            case 3:
                dispatch = {1u, 1u, 2u};
                return true;
            case 4:
                dispatch = {0u, 0u, 1u};
                return true;
            case 5:
                dispatch = {2u, 1u, 2u};
                return true;
            default:
                return false;
        }
    }
    else
    {
        return false;
    }
}

// --- 1D ---

/**
 * @brief One segment: sample the volume field at both of its vertices.
 *
 * The traces of a segment are its two end vertices, so there is no trace
 * quadrature and no tangential stage: the whole operator is
 * \f$u(\pm 1) = \sum_p h_p(\pm 1)\, u_p\f$. Under @p endPtsCollocated
 * the volume rule contains both domain endpoints, \f$h_p(\pm 1)\f$ is a
 * Kronecker delta and the first and last volume values are copied out.
 * Otherwise the table is applied in full, with the row stride of two a
 * segment's two direction-0 traces give it.
 *
 * The name is inherited: nothing from the shared BwdTrans kernels is
 * called here.
 *
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane     Lane index within the warp; selects the element.
 * @param   nq0       Volume quadrature points of the segment. The
 *                    caller passes @c nm0 despite the parameter name.
 * @param   ntbasis0  eInterp table \f$h_p(\pm 1)\f$, indexed
 *                    `ntbasis0[p * 2 + e]`.
 * @param   in        Volume field of this lane, warp interleaved.
 * @param   out       The two vertex values, warp interleaved.
 * @param   isCollocated      Accepted for symmetry with the two- and
 *                    three-dimensional kernels and unused: a segment's
 *                    trace is a point and has no tangential direction.
 * @param   endPtsCollocated  Volume rule contains the domain endpoints.
 */
template <typename TData>
NEK_DEVICE_INLINE static void BwdTransSegSumFacKernelTrace(
    const unsigned int ilane, const unsigned int nq0, const TData *ntbasis0,
    const TData *in, TData *out, [[maybe_unused]] const bool isCollocated,
    const bool endPtsCollocated)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if (endPtsCollocated)
    {
        out[warpsize * 0 + ilane] = in[warpsize * 0 + ilane];
        out[warpsize * 1 + ilane] = in[warpsize * (nq0 - 1) + ilane];
    }
    else
    {
        for (unsigned int e = 0u; e < 2; ++e)
        {
            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nq0; ++p)
            {
                tmp += in[warpsize * p + ilane] * ntbasis0[p * 2 + e];
            }
            out[warpsize * e + ilane] = tmp;
        }
    }
}

/**
 * @brief Launch entry point for a block of segments: both
 * vertices of every element.
 *
 * Grid-strides over the block's elements, splits the global index into a
 * warp and a lane, offsets @p in by the segment's volume point count
 * per element and @p out by two, and calls
 * BwdTransSegSumFacKernelTrace. The output stride of two is written
 * into this kernel rather than taken from the output block.
 *
 * The one-dimensional arm of the overload set OperatorND() launches;
 * the higher-dimensional arms follow below. It takes the whole family's
 * argument order and so accepts @p nmTot and @p wsp, which a segment
 * needs for nothing: the input stride is the size parameter's nm0() and
 * there is no tangential stage to give a workspace to. The tangential
 * collocation flags are an empty pack in one dimension, so none is
 * passed either.
 *
 * @tparam SHAPE_TYPE       Seg; unread, the path being the same for
 *                      every one-dimensional expansion.
 * @tparam TTraceSizeParameter1D    NonTemplatedTraceSizeParameter1D or
 *                      a TemplatedTraceSizeParameter1D instantiation.
 * @tparam TthreadBlock     Thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam1D Volume and trace point counts of the segment;
 *                      only nm0() is read here.
 * @param   nelmt       Elements in the block, including padding.
 * @param   nmTot       Entries per element of @p in; unread, see above.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$.
 * @param   in          Volume field of the block, warp interleaved.
 * @param   out         Extracted vertex values of the block, two per
 *                      element, warp interleaved.
 * @param   wsp         Workspace; unread, see above, and null because
 *                      PhysTraceExtractWorkSpaceSize() asks for none.
 * @param   endPtsCollocated0   Volume rule contains the domain
 *                      endpoints.
 * @param   shmemptr    Dynamic shared memory; unused.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter1D,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void PhysTraceExtractKernelLauncher(
    const TTraceSizeParameter1D sizeParam1D, const size_t nelmt,
    [[maybe_unused]] const unsigned int nmTot,
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr = in + nm0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + 2 * (nelmt * c + warpsize * iwarp);

        // A segment has no in-trace direction, so there is no
        // tangential collocation flag to forward; the callee marks the
        // parameter [[maybe_unused]] and the constructor recorded false.
        BwdTransSegSumFacKernelTrace(ilane, nm0, ntbasis0, inptr, outptr, false,
                                     endPtsCollocated0);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Launch entry point for one named vertex of a block of
 * segments.
 *
 * The single-trace counterpart of PhysTraceExtractKernelLauncher, and
 * self-contained: it inlines the same two arms instead of calling
 * BwdTransSegSumFacKernelTrace. Each element contributes one value,
 * written at `out[(numDataOut * iwarp + outOffset) * warpsize + ilane]`, so
 * the caller places the trace within the output block through
 * @p outOffset.
 *
 * @tparam TTraceSizeParameter1D    NonTemplatedTraceSizeParameter1D or
 *                      a TemplatedTraceSizeParameter1D instantiation.
 * @tparam TthreadBlock     Thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   traceid     0 for the lower vertex, 1 for the upper one; also
 *                      the column of @p ntbasis0 the general arm reads.
 * @param   sizeParam1D Volume and trace point counts of the segment;
 *                      only nm0() is read here.
 * @param   nelmt       Elements in the block, including padding.
 * @param   numDataOut    Entries per element of @p out.
 * @param   outOffset   Where this trace's value sits within an element's
 *                      output.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$, indexed
 *                      `ntbasis0[p * 2 + traceid]`.
 * @param   in          Volume field of the block, warp interleaved.
 * @param   out         Trace field of the block, warp interleaved.
 * @param   endPtsCollocated0   Volume rule contains the domain
 *                      endpoints, so the vertex value is a volume value.
 * @param   shmemptr    Dynamic shared memory; unused.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter1D,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void PhysTraceExtractTraceKernelLauncher(
    const unsigned int traceid, const TTraceSizeParameter1D sizeParam1D,
    const size_t nelmt, [[maybe_unused]] const unsigned int nmTot,
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

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr = in + nm0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + numDataOut * (nelmt * c + warpsize * iwarp) +
                        outOffset * warpsize;

        if (endPtsCollocated0)
        {
            const unsigned int p = (traceid == 0) ? 0 : (nm0 - 1);
            outptr[ilane]        = inptr[p * warpsize + ilane];
        }
        else
        {
            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p)
            {
                tmp += inptr[p * warpsize + ilane] * ntbasis0[p * 2 + traceid];
            }
            outptr[ilane] = tmp;
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

// --- 2D ---

/**
 * @brief One two-dimensional element, one normal direction: extract the
 * edges of that direction and resample them onto their trace points.
 *
 * Both stages of the extraction in a single kernel, with the normal
 * direction fixed by @p NormalDir. Stage one takes the volume field to
 * the [@p edg, @p nedg) window of edges of that direction and writes
 * `nedge = nedg - edg` values per in-edge point into @p wsp with the
 * edges interleaved, `wsp[warpsize * (q * nedge + localE) + ilane]`.
 * Stage two contracts the in-edge direction against @p tbasis onto
 * @p npts trace points and writes each edge at its own offset in @p out.
 *
 * Under @p endPtsCollocated stage one is a plane selection, the lower
 * edge of the direction being the first volume plane and the upper one
 * the last. Under @p isCollocated stage two is a copy. The @c ntbasis
 * row stride is a compile-time constant taken from the shape, two in
 * direction 0 for both shapes and two in direction 1 except for a
 * triangle, so it is the direction's trace count on the bulk and the
 * per-trace route alike. That is the contrast with the
 * three-dimensional normal stage, which derives the stride from its loop
 * bound instead.
 *
 * The name is inherited: nothing from the shared BwdTrans kernels is
 * called here.
 *
 * @note The two branches computing the plane offset in the
 *       @p NormalDir 0 endpoint arm agree over the range the callers
 *       use: `(e == 0) ? 0 : (nm0 - 1)` and `e * (nm0 - 1)` are the same
 *       for @em e in {0,1}.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri.
 * @tparam NormalDir  Normal direction of the edges handled: 0 or 1.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
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
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Trace values, warp interleaved, already offset to
 *                      this direction's block.
 * @param   wsp         Scratch between the two stages,
 *                      `2 * max(nm0,nm1)` entries, warp interleaved.
 * @param   isCollocated      Trace points of this direction coincide
 *                      with the volume points of the in-edge direction.
 * @param   endPtsCollocated  Volume rule of the normal direction
 *                      contains the domain endpoints.
 * @param   offset_edge0,offset_edge1     Where each of the two edges
 *                      covered starts within @p out. The per-trace route
 *                      leaves both at zero, its window being one edge
 *                      wide.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, int NormalDir, typename TData>
NEK_DEVICE_INLINE static void BwdTransQuadSumFacKernelTrace(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1, const unsigned int edg,
    const unsigned int nedg, const unsigned int npts,
    const TData *NEK_RESTRICT ntbasis, const TData *NEK_RESTRICT tbasis,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, const bool isCollocated,
    const bool endPtsCollocated, unsigned int offset_edge0 = 0,
    unsigned int offset_edge1 = 0)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    const unsigned int nedge        = nedg - edg;

    if constexpr (NormalDir == 0)
    {
        constexpr unsigned int ntrace = 2u;

        // 1. Normal direction (dir 0) to wsp
        if (endPtsCollocated)
        {
            for (unsigned int e = edg; e < nedg; ++e)
            {
                const unsigned int localE = e - edg;
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    unsigned int offset_in;
                    if constexpr (SHAPE_TYPE == LibUtilities::Tri)
                    {
                        // Tri: Group 0 extracts Edge 2 (x=-1) and Edge 1
                        // (hypotenuse) Edge 2 is at x=-1 (p=0), Edge 1 is at
                        // x=1 (p=nm0-1)
                        offset_in = (e == 0) ? 0 : (nm0 - 1);
                    }
                    else
                    {
                        // Quad: Group 0 extracts Edge 3 (x=-1) and Edge 1 (x=1)
                        offset_in = e * (nm0 - 1);
                    }
                    wsp[warpsize * (q * nedge + localE) + ilane] =
                        in[warpsize * (q * nm0 + offset_in) + ilane];
                }
            }
        }
        else
        {
            for (unsigned int e = edg; e < nedg; ++e)
            {
                const unsigned int localE = e - edg;
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p)
                    {
                        tmp += in[warpsize * (q * nm0 + p) + ilane] *
                               ntbasis[p * ntrace + e];
                    }
                    wsp[warpsize * (q * nedge + localE) + ilane] = tmp;
                }
            }
        }

        // 2. Trace direction (dir 1) to out
        if (isCollocated)
        {
            for (unsigned int e = edg; e < nedg; ++e)
            {
                const unsigned int localE = e - edg;
                const unsigned int offset_out =
                    (localE == 0) ? offset_edge0 : offset_edge1;
                for (unsigned int j = 0u; j < npts; ++j)
                {
                    out[warpsize * (offset_out + j) + ilane] =
                        wsp[warpsize * (j * nedge + localE) + ilane];
                }
            }
        }
        else
        {
            for (unsigned int e = edg; e < nedg; ++e)
            {
                const unsigned int localE = e - edg;
                const unsigned int offset_out =
                    (localE == 0) ? offset_edge0 : offset_edge1;
                for (unsigned int j = 0u; j < npts; ++j)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q)
                    {
                        tmp += wsp[warpsize * (q * nedge + localE) + ilane] *
                               tbasis[q * npts + j];
                    }
                    out[warpsize * (offset_out + j) + ilane] = tmp;
                }
            }
        }
    }
    else // NormalDir == 1
    {
        constexpr unsigned int ntrace =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

        // 1. Normal direction (dir 1) to wsp
        if (endPtsCollocated)
        {
            for (unsigned int e = edg; e < nedg; ++e)
            {
                const unsigned int localE = e - edg;
                unsigned int offset_in    = e * nm0 * (nm1 - 1);
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    wsp[warpsize * (p * nedge + localE) + ilane] =
                        in[warpsize * (offset_in + p) + ilane];
                }
            }
        }
        else
        {
            for (unsigned int e = edg; e < nedg; ++e)
            {
                const unsigned int localE = e - edg;
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q)
                    {
                        tmp += in[warpsize * (q * nm0 + p) + ilane] *
                               ntbasis[q * ntrace + e];
                    }
                    wsp[warpsize * (p * nedge + localE) + ilane] = tmp;
                }
            }
        }

        // 2. Trace direction (dir 0) to out
        if (isCollocated)
        {
            for (unsigned int e = edg; e < nedg; ++e)
            {
                const unsigned int localE = e - edg;
                const unsigned int offset_out =
                    (localE == 0) ? offset_edge0 : offset_edge1;
                for (unsigned int i = 0u; i < npts; ++i)
                {
                    out[warpsize * (offset_out + i) + ilane] =
                        wsp[warpsize * (i * nedge + localE) + ilane];
                }
            }
        }
        else
        {
            for (unsigned int e = edg; e < nedg; ++e)
            {
                const unsigned int localE = e - edg;
                const unsigned int offset_out =
                    (localE == 0) ? offset_edge0 : offset_edge1;
                for (unsigned int i = 0u; i < npts; ++i)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p)
                    {
                        tmp += wsp[warpsize * (p * nedge + localE) + ilane] *
                               tbasis[p * npts + i];
                    }
                    out[warpsize * (offset_out + i) + ilane] = tmp;
                }
            }
        }
    }
}

/**
 * @brief All edges of one two-dimensional element: both directions, in
 * packed trace order.
 *
 * The bulk worker. The direction-0 pair goes into the first `2 * nq0`
 * entries of @p out, lower edge then upper edge, and the direction-1
 * traces follow: a pair for a quadrilateral, at offsets 0 and @p nq1
 * past that, and a single edge for a triangle, whose direction-1 upper
 * end is the collapsed vertex. Both shapes run the same leaf kernel
 * template, its @c Tri instantiation differing only in the direction-0
 * offset branch, which agrees numerically, and in the direction-1
 * @c ntrace constant; the hypotenuse is just the \f$\eta_0 = +1\f$ line
 * of the collapsed square.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   nm0,nm1     Volume quadrature points per direction.
 * @param   nq0,nq1     Trace quadrature points of the direction-0 and
 *                      direction-1 edges.
 * @param   ntbasis0,ntbasis1   eInterp tables \f$h_p(\pm 1)\f$ per
 *                      direction.
 * @param   basis0,basis1       eInterp tables to the trace points, one
 *                      per normal direction; the @c tbasis of the rest
 *                      of the file.
 * @param   wsp         Scratch between the two stages,
 *                      `2 * max(nm0,nm1)` entries, warp interleaved.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         All of this element's trace values, warp
 *                      interleaved.
 * @param   isCollocated0,isCollocated1     Trace points of that normal
 *                      direction coincide with the volume points of the
 *                      in-edge direction.
 * @param   endPtsCollocated0,endPtsCollocated1   Volume rule of that
 *                      direction contains the domain endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TData>
NEK_DEVICE_INLINE static void PhysTraceExtract2DSumFacKernelCore(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const TData *ntbasis0,
    const TData *ntbasis1, const TData *basis0, const TData *basis1, TData *wsp,
    const TData *in, TData *out, const bool isCollocated0,
    const bool isCollocated1, const bool endPtsCollocated0,
    const bool endPtsCollocated1)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int offset_Trace0_Edge0 = 0;
    const unsigned int offset_Trace0_Edge1 = nq0;
    const unsigned int offset_Trace1       = 2 * nq0;

    // Dir 0 normal set
    BwdTransQuadSumFacKernelTrace<SHAPE_TYPE, 0>(
        ilane, nm0, nm1, nq0, nq1, 0, 2, nq0, ntbasis0, basis0, in, out, wsp,
        isCollocated0, endPtsCollocated0, offset_Trace0_Edge0,
        offset_Trace0_Edge1);

    // Dir 1 normal set
    constexpr unsigned int ntrace1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

    BwdTransQuadSumFacKernelTrace<SHAPE_TYPE, 1>(
        ilane, nm0, nm1, nq0, nq1, 0, ntrace1, nq1, ntbasis1, basis1, in,
        out + offset_Trace1 * warpsize, wsp, isCollocated1, endPtsCollocated1,
        0, (ntrace1 == 1u) ? 0u : nq1);
}

/**
 * @brief Launch entry point for a block of quadrilaterals or
 * triangles: every edge of every element.
 *
 * Grid-strides over the block's elements, offsets @p in by @p nmTot
 * values per element, @p out by the element's total trace count and
 * @p wsp by `2 * max(nm0,nm1)`, then calls
 * PhysTraceExtract2DSumFacKernelCore. The output stride is the size
 * parameter's nqTotTrace(), which sums the per-direction edge counts
 * against the trace point counts, matching the packing the core
 * writes.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam TTraceSizeParameter2D    NonTemplatedTraceSizeParameter2D or
 *                          a TemplatedTraceSizeParameter2D
 *                          instantiation.
 * @tparam TthreadBlock     Thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam2D Volume quadrature points per direction, nm0()
 *                      and nm1(), and the trace quadrature points of
 *                      the direction-0 and direction-1 edges, nq00()
 *                      and nq10().
 * @param   nelmt       Elements in the block, including padding.
 * @param   nmTot       Entries per element of @p in.
 * @param   ntbasis0,ntbasis1   eInterp tables \f$h_p(\pm 1)\f$ per
 *                      direction.
 * @param   basis0,basis1       eInterp tables to the trace points, one
 *                      per normal direction.
 * @param   in          Volume field of the block, warp interleaved.
 * @param   out         Trace field of the block, warp interleaved.
 * @param   wsp         Scratch of the block, `2 * max(nm0,nm1)` entries
 *                      per element.
 * @param   isCollocated0,isCollocated1           Tangential collocation
 *                      flags, one per normal direction.
 * @param   endPtsCollocated0,endPtsCollocated1   Volume rule of that
 *                      direction contains the domain endpoints.
 * @param   shmemptr    Dynamic shared memory; unused.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter2D,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void PhysTraceExtractKernelLauncher(
    const TTraceSizeParameter2D sizeParam2D, const size_t nelmt,
    const unsigned int nmTot, const TData *NEK_RESTRICT ntbasis0,
    const TData *NEK_RESTRICT ntbasis1, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const bool isCollocated0,
    const bool isCollocated1, const bool endPtsCollocated0,
    const bool endPtsCollocated1, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam2D.nm0();
    const unsigned int nm1 = sizeParam2D.nm1();
    const unsigned int nq0 = sizeParam2D.nq00();
    const unsigned int nq1 = sizeParam2D.nq10();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int nqTotOut = sizeParam2D.template nqTotTrace<SHAPE_TYPE>();

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTotOut * (nelmt * c + warpsize * iwarp);
        TData *wspptr      = wsp + PhysTraceExtractBlockSize(sizeParam2D) *
                                  (nelmt * c + warpsize * iwarp);

        PhysTraceExtract2DSumFacKernelCore<SHAPE_TYPE>(
            ilane, nm0, nm1, nq0, nq1, ntbasis0, ntbasis1, basis0, basis1,
            wspptr, inptr, outptr, isCollocated0, isCollocated1,
            endPtsCollocated0, endPtsCollocated1);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief One named edge of one two-dimensional element: turn the edge id
 * into a normal direction and a position, then extract.
 *
 * The single-trace worker. GetTraceEdgeDispatch supplies the direction
 * and a one-edge window, and the matching @c NormalDir instantiation of
 * BwdTransQuadSumFacKernelTrace does the work with both output offsets
 * left at their default of zero, @p out already pointing at this trace's
 * slot.
 *
 * @note An unrecognised @p edge is ignored silently, leaving @p out
 * untouched: device code cannot raise a host error.
 *
 * @tparam SHAPE_TYPE Quad, Tri or NodalTri.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   edge        Nektar edge id of the trace to extract.
 * @param   nm0,nm1     Volume quadrature points per direction.
 * @param   nq00,nq10   Trace quadrature points of the direction-0 and
 *                      direction-1 edges.
 * @param   ntbasis0,ntbasis1   eInterp tables \f$h_p(\pm 1)\f$ per
 *                      direction.
 * @param   tbasis00,tbasis10   eInterp tables to the trace points, one
 *                      per normal direction.
 * @param   wsp         Scratch between the two stages,
 *                      `2 * max(nm0,nm1)` entries, warp interleaved.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         This trace's values, warp interleaved.
 * @param   isCollocated00,isCollocated10         Tangential collocation
 *                      flags, one per normal direction.
 * @param   endPtsCollocated0,endPtsCollocated1   Volume rule of that
 *                      direction contains the domain endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TData>
NEK_DEVICE_INLINE static void PhysTraceExtractEdgeKernel(
    const unsigned int ilane, const unsigned int edge, const unsigned int nm0,
    const unsigned int nm1, const unsigned int nq00, const unsigned int nq10,
    const TData *NEK_RESTRICT ntbasis0, const TData *NEK_RESTRICT ntbasis1,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis10,
    TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated00,
    const bool isCollocated10, const bool endPtsCollocated0,
    const bool endPtsCollocated1)
{
    TraceExtractDispatch dispatch{};
    if (!GetTraceEdgeDispatch<SHAPE_TYPE>(edge, dispatch))
    {
        return;
    }

    if (dispatch.normalDir == 0u)
    {
        BwdTransQuadSumFacKernelTrace<SHAPE_TYPE, 0>(
            ilane, nm0, nm1, nq00, nq10, dispatch.fac, dispatch.nfac, nq00,
            ntbasis0, tbasis00, in, out, wsp, isCollocated00,
            endPtsCollocated0);
    }
    else
    {
        BwdTransQuadSumFacKernelTrace<SHAPE_TYPE, 1>(
            ilane, nm0, nm1, nq00, nq10, dispatch.fac, dispatch.nfac, nq10,
            ntbasis1, tbasis10, in, out, wsp, isCollocated10,
            endPtsCollocated1);
    }
}

/**
 * @brief Launch entry point for one named edge of a block of
 * quadrilaterals or triangles.
 *
 * The single-trace counterpart of PhysTraceExtractKernelLauncher. It
 * offsets @p out to `(numDataOut * iwarp + outOffset) * warpsize` so that
 * the caller can place each trace in turn within the output block, and
 * calls PhysTraceExtractEdgeKernel.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri. The block operator
 *                          instantiates a @c NodalTri as @c Tri here.
 * @tparam TTraceSizeParameter2D    NonTemplatedTraceSizeParameter2D or
 *                          a TemplatedTraceSizeParameter2D
 *                          instantiation.
 * @tparam TthreadBlock     Thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   traceid     Nektar edge id of the trace to extract.
 * @param   sizeParam2D Volume quadrature points per direction, nm0()
 *                      and nm1(), and the trace quadrature points of
 *                      the direction-0 and direction-1 edges, nq00()
 *                      and nq10().
 * @param   nelmt       Elements in the block, including padding.
 * @param   nmTot       Entries per element of @p in.
 * @param   numDataOut    Entries per element of @p out.
 * @param   outOffset   Where this trace's block sits within an element's
 *                      output.
 * @param   ntbasis0,ntbasis1   eInterp tables \f$h_p(\pm 1)\f$ per
 *                      direction.
 * @param   tbasis00,tbasis10   eInterp tables to the trace points, one
 *                      per normal direction.
 * @param   in          Volume field of the block, warp interleaved.
 * @param   out         Trace field of the block, warp interleaved.
 * @param   wsp         Scratch of the block, `2 * max(nm0,nm1)` entries
 *                      per element.
 * @param   isCollocated00,isCollocated10         Tangential collocation
 *                      flags, one per normal direction.
 * @param   endPtsCollocated0,endPtsCollocated1   Volume rule of that
 *                      direction contains the domain endpoints.
 * @param   shmemptr    Dynamic shared memory; unused.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter2D,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void PhysTraceExtractTraceKernelLauncher(
    const unsigned int traceid, const TTraceSizeParameter2D sizeParam2D,
    const size_t nelmt, const unsigned int nmTot, const unsigned int numDataOut,
    const unsigned int outOffset, const TData *NEK_RESTRICT ntbasis0,
    const TData *NEK_RESTRICT ntbasis1, const TData *NEK_RESTRICT tbasis00,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const bool isCollocated00,
    const bool isCollocated10, const bool endPtsCollocated0,
    const bool endPtsCollocated1, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0  = sizeParam2D.nm0();
    const unsigned int nm1  = sizeParam2D.nm1();
    const unsigned int nq00 = sizeParam2D.nq00();
    const unsigned int nq10 = sizeParam2D.nq10();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + numDataOut * (nelmt * c + warpsize * iwarp) +
                        outOffset * warpsize;
        TData *wspptr = wsp + PhysTraceExtractBlockSize(sizeParam2D) *
                                  (nelmt * c + warpsize * iwarp);

        PhysTraceExtractEdgeKernel<SHAPE_TYPE>(
            ilane, traceid, nm0, nm1, nq00, nq10, ntbasis0, ntbasis1, tbasis00,
            tbasis10, wspptr, inptr, outptr, isCollocated00, isCollocated10,
            endPtsCollocated0, endPtsCollocated1);
        e += getGlobalRange<0>(threadBlock);
    }
}

// --- 3D ---

/**
 * @brief Leaf kernel, tangential stage in three dimensions: resample a
 * face, or a pair of faces, from the volume grid onto the face's own
 * quadrature points.
 *
 * Takes @p nface faces laid out `(f, q1, q0)`, the first tangential
 * index fastest, and applies the two interpolation tables in turn,
 * \f$\text{out}_{fji} = \sum_q t^1_{qj} \sum_p t^0_{pi}\,
 * \text{in}_{fqp}\f$. Three arms: with @p isCollocated1 only direction 0
 * is interpolated, straight into @p out; with @p isCollocated0 only
 * direction 1 is, again straight into @p out; otherwise direction 0 goes
 * into @p wsp and direction 1 from there into @p out. The
 * both-collocated case never arrives, the glue level having skipped this
 * kernel entirely for it.
 *
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   nface       Faces covered, already the width of the window.
 * @param   nqfrom0,nqfrom1     Volume quadrature counts of the two
 *                      tangential directions, the source grid.
 * @param   nqto0,nqto1 Trace quadrature counts of the face, the target
 *                      grid.
 * @param   tbasis0,tbasis1     eInterp tables to the trace points of the
 *                      two tangential directions, indexed
 *                      `tbasis0[p * nqto0 + i]`.
 * @param   wsp         Scratch between the two contractions,
 *                      `nface * nqfrom1 * nqto0` entries, warp
 *                      interleaved. Untouched by the two collocated
 *                      arms.
 * @param   in          Face values on the volume face grid, warp
 *                      interleaved.
 * @param   out         Face values on the trace grid, warp interleaved.
 * @param   isCollocated0,isCollocated1   That tangential direction's trace
 *                      points coincide with the volume points it comes
 *                      from, so its table is the identity.
 */
template <typename TData>
NEK_DEVICE_INLINE static void PhysInterpFaceKernelTrace(
    const unsigned int ilane, const unsigned int nface,
    const unsigned int nqfrom0, const unsigned int nqfrom1,
    const unsigned int nqto0, const unsigned int nqto1,
    const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
    TData *NEK_RESTRICT wsp, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated0, const bool isCollocated1)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if (isCollocated1)
    {
        // Interpolate dir 0 directly to output.
        for (unsigned int f = 0u; f < nface; ++f)
        {
            for (unsigned int q1 = 0u; q1 < nqfrom1; ++q1)
            {
                for (unsigned int i = 0u; i < nqto0; ++i)
                {
                    TData tmp =
                        in[warpsize * ((f * nqfrom1 + q1) * nqfrom0) + ilane] *
                        tbasis0[i];
                    for (unsigned int p = 1u; p < nqfrom0; ++p)
                    {
                        tmp +=
                            in[warpsize * ((f * nqfrom1 + q1) * nqfrom0 + p) +
                               ilane] *
                            tbasis0[p * nqto0 + i];
                    }
                    out[warpsize * ((f * nqto1 + q1) * nqto0 + i) + ilane] =
                        tmp;
                }
            }
        }
    }
    else if (isCollocated0)
    {
        // Interpolate dir 1 only.
        for (unsigned int f = 0u; f < nface; ++f)
        {
            for (unsigned int i = 0u; i < nqto0; ++i)
            {
                for (unsigned int j = 0u; j < nqto1; ++j)
                {
                    TData tmp =
                        in[warpsize * ((f * nqfrom1) * nqto0 + i) + ilane] *
                        tbasis1[j];
                    for (unsigned int q = 1u; q < nqfrom1; ++q)
                    {
                        tmp += in[warpsize * ((f * nqfrom1 + q) * nqto0 + i) +
                                  ilane] *
                               tbasis1[q * nqto1 + j];
                    }
                    out[warpsize * ((f * nqto1 + j) * nqto0 + i) + ilane] = tmp;
                }
            }
        }
    }
    else
    {
        // Interpolate dir 0 to workspace.
        for (unsigned int f = 0u; f < nface; ++f)
        {
            for (unsigned int q1 = 0u; q1 < nqfrom1; ++q1)
            {
                for (unsigned int i = 0u; i < nqto0; ++i)
                {
                    TData tmp =
                        in[warpsize * ((f * nqfrom1 + q1) * nqfrom0) + ilane] *
                        tbasis0[i];
                    for (unsigned int p = 1u; p < nqfrom0; ++p)
                    {
                        tmp +=
                            in[warpsize * ((f * nqfrom1 + q1) * nqfrom0 + p) +
                               ilane] *
                            tbasis0[p * nqto0 + i];
                    }
                    wsp[warpsize * ((f * nqfrom1 + q1) * nqto0 + i) + ilane] =
                        tmp;
                }
            }
        }

        // Interpolate dir 1 to output.
        for (unsigned int f = 0u; f < nface; ++f)
        {
            for (unsigned int i = 0u; i < nqto0; ++i)
            {
                for (unsigned int j = 0u; j < nqto1; ++j)
                {
                    TData tmp =
                        wsp[warpsize * ((f * nqfrom1) * nqto0 + i) + ilane] *
                        tbasis1[j];
                    for (unsigned int q = 1u; q < nqfrom1; ++q)
                    {
                        tmp += wsp[warpsize * ((f * nqfrom1 + q) * nqto0 + i) +
                                   ilane] *
                               tbasis1[q * nqto1 + j];
                    }
                    out[warpsize * ((f * nqto1 + j) * nqto0 + i) + ilane] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Leaf kernel, normal stage in three dimensions: take the volume
 * field to the direction-0 faces of the [@p fac, @p nface) window.
 *
 * Produces one value per (face, direction-2 point, direction-1 point),
 * direction 1 fastest, either by selecting the boundary plane
 * \f$i = 0\f$ or \f$i = nq_0 - 1\f$ when @p END_PTS_COLLOCATED0, or by
 * contracting direction 0 against \f$h_p(\pm 1)\f$ in the general arm.
 * The result is the face grid sampled at the volume quadrature points,
 * which the caller either keeps, the tangential directions being collocated,
 * or hands to PhysInterpFaceKernelTrace.
 *
 * @tparam END_PTS_COLLOCATED0    Direction-0 volume rule contains the
 *                      domain endpoints.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   fac         Position of the first face within direction 0.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of the @c ntbasis table: the number of
 *                      traces the direction has, which is not @p nface
 *                      when the call covers part of the direction.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Face values on the volume face grid,
 *                      `(nface - fac) * nq2 * nq1` entries, warp
 *                      interleaved.
 */
template <bool END_PTS_COLLOCATED0, typename TData>
NEK_DEVICE_INLINE static void PhysExtractEndFacesN0KernelTrace3D(
    const unsigned int ilane, const unsigned int fac, const unsigned int nface,
    const unsigned int tstride, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT ntbasis0,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (END_PTS_COLLOCATED0)
    {
        unsigned int nq0nq1 = nq0 * nq1;
        for (unsigned int f = fac, cnt_fkj = 0; f < nface; ++f)
        {
            unsigned int offset = f * (nq0 - 1);
            for (unsigned int k = 0u; k < nq2; ++k)
            {
                unsigned int offset_fk = k * nq0nq1 + offset;
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_fkj)
                {
                    out[warpsize * cnt_fkj + ilane] =
                        in[warpsize * (offset_fk + j * nq0) + ilane];
                }
            }
        }
    }
    else
    {
        unsigned int nq0nq1 = nq0 * nq1;
        for (unsigned int f = fac, cnt_fkj = 0; f < nface; ++f)
        {
            for (unsigned int k = 0u; k < nq2; ++k)
            {
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_fkj)
                {
                    unsigned int offset_fkp = k * nq0nq1 + j * nq0;
                    TData tmp = in[warpsize * offset_fkp + ilane] * ntbasis0[f];
                    for (unsigned int p = 1u; p < nq0; ++p)
                    {
                        tmp += in[warpsize * (offset_fkp + p) + ilane] *
                               ntbasis0[p * tstride + f];
                    }
                    out[warpsize * cnt_fkj + ilane] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Leaf kernel, normal stage in three dimensions: the direction-1
 * twin of PhysExtractEndFacesN0KernelTrace3D.
 *
 * Produces one value per (face, direction-2 point, direction-0 point),
 * direction 0 fastest, either by selecting the boundary plane
 * \f$j = 0\f$ or \f$j = nq_1 - 1\f$ when @p END_PTS_COLLOCATED1, or by
 * contracting direction 1 against \f$h_q(\pm 1)\f$.
 *
 * @tparam END_PTS_COLLOCATED1    Direction-1 volume rule contains the
 *                      domain endpoints.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   fac         Position of the first face within direction 1.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of the @c ntbasis table: the number of
 *                      traces the direction has, which is not @p nface
 *                      when the call covers part of the direction.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis1    eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Face values on the volume face grid,
 *                      `(nface - fac) * nq2 * nq0` entries, warp
 *                      interleaved.
 */
template <bool END_PTS_COLLOCATED1, typename TData>
NEK_DEVICE_INLINE static void PhysExtractEndFacesN1KernelTrace3D(
    const unsigned int ilane, const unsigned int fac, const unsigned int nface,
    const unsigned int tstride, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT ntbasis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (END_PTS_COLLOCATED1)
    {
        unsigned int nq0nq1 = nq0 * nq1;
        for (unsigned int f = fac, cnt_fki = 0; f < nface; ++f)
        {
            unsigned int offset = f * nq0 * (nq1 - 1);
            for (unsigned int k = 0u; k < nq2; ++k)
            {
                unsigned int offset_ki = k * nq0nq1 + offset;
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_fki)
                {
                    out[warpsize * cnt_fki + ilane] =
                        in[warpsize * (offset_ki + i) + ilane];
                }
            }
        }
    }
    else
    {
        unsigned int nq0nq1 = nq0 * nq1;
        for (unsigned int f = fac, cnt_fki = 0; f < nface; ++f)
        {
            for (unsigned int k = 0u; k < nq2; ++k)
            {
                unsigned int offset_k = k * nq0nq1;
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_fki)
                {
                    TData tmp =
                        in[warpsize * (offset_k + i) + ilane] * ntbasis1[f];
                    for (unsigned int q = 1u; q < nq1; ++q)
                    {
                        tmp += in[warpsize * (offset_k + q * nq0 + i) + ilane] *
                               ntbasis1[q * tstride + f];
                    }
                    out[warpsize * cnt_fki + ilane] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Leaf kernel, normal stage in three dimensions: the direction-2
 * twin of PhysExtractEndFacesN0KernelTrace3D.
 *
 * Produces one value per (face, direction-1 point, direction-0 point),
 * direction 0 fastest, either by selecting the boundary plane
 * \f$k = 0\f$ or \f$k = nq_2 - 1\f$ when @p END_PTS_COLLOCATED2, or by
 * contracting direction 2 against \f$h_r(\pm 1)\f$. Direction 2 carries
 * a trace pair on a hexahedron only; every other three-dimensional shape
 * collapses it to a single face.
 *
 * @tparam END_PTS_COLLOCATED2    Direction-2 volume rule contains the
 *                      domain endpoints.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   fac         Position of the first face within direction 2.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of the @c ntbasis table: the number of
 *                      traces the direction has, which is not @p nface
 *                      when the call covers part of the direction.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis2    eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Face values on the volume face grid,
 *                      `(nface - fac) * nq1 * nq0` entries, warp
 *                      interleaved.
 */
template <bool END_PTS_COLLOCATED2, typename TData>
NEK_DEVICE_INLINE static void PhysExtractEndFacesN2KernelTrace3D(
    const unsigned int ilane, const unsigned int fac, const unsigned int nface,
    const unsigned int tstride, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT ntbasis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (END_PTS_COLLOCATED2)
    {
        for (unsigned int k = fac, cnt_kji = 0; k < nface; ++k)
        {
            unsigned int offset = k * nq0 * nq1 * (nq2 - 1);
            for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
            {
                for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji, ++cnt_ji)
                {
                    out[warpsize * cnt_kji + ilane] =
                        in[warpsize * (cnt_ji + offset) + ilane];
                }
            }
        }
    }
    else
    {
        unsigned int nq01 = nq0 * nq1;
        for (unsigned int k = fac, cnt_kji = 0; k < nface; ++k)
        {
            for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
            {
                for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji, ++cnt_ji)
                {
                    TData tmp = in[warpsize * cnt_ji + ilane] * ntbasis2[k];
                    for (unsigned int r = 1u; r < nq2; ++r)
                    {
                        tmp += in[warpsize * (cnt_ji + r * nq01) + ilane] *
                               ntbasis2[r * tstride + k];
                    }
                    out[warpsize * cnt_kji + ilane] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Glue for the direction-0 normal faces: normal stage, then the
 * tangential stage if it is needed.
 *
 * When both tangential directions are collocated the trace grid @em is the
 * volume face grid, and PhysExtractEndFacesN0KernelTrace3D writes
 * straight into @p out. Otherwise it writes into @p wsp1 and
 * PhysInterpFaceKernelTrace resamples `(nq1,nq2)` onto `(nq00,nq01)`,
 * using @p wsp2 when neither direction is collocated.
 *
 * @tparam END_PTS_COLLOCATED0    Direction-0 volume rule contains the
 *                      domain endpoints.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   fac         Position of the first face within direction 0.
 * @param   nfac        One past the position of the last face.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   nq00,nq01   Trace quadrature points of the two tangential
 *                      directions of a direction-0 face.
 * @param   tbasis00,tbasis01   eInterp tables to those trace points.
 * @param   wsp1        Face values between the two stages,
 *                      `(nfac - fac) * nq2 * nq1` entries.
 * @param   wsp2        Scratch of the fully general tangential stage.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Face values on the trace grid, warp interleaved.
 * @param   isCollocated00,isCollocated01   That tangential direction's trace
 *                      points coincide with the volume points it comes
 *                      from.
 */
template <bool END_PTS_COLLOCATED0, typename TData>
NEK_DEVICE_INLINE static void PhysExtractFaceN0KernelTrace3D(
    const unsigned int ilane, const unsigned int fac, const unsigned int nfac,
    const unsigned int tstride, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT ntbasis0,
    const unsigned int nq00, const unsigned int nq01,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    TData *NEK_RESTRICT wsp1, TData *NEK_RESTRICT wsp2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated00, const bool isCollocated01)
{
    if (isCollocated00 && isCollocated01)
    {
        PhysExtractEndFacesN0KernelTrace3D<END_PTS_COLLOCATED0>(
            ilane, fac, nfac, tstride, nq0, nq1, nq2, ntbasis0, in, out);
    }
    else
    {
        PhysExtractEndFacesN0KernelTrace3D<END_PTS_COLLOCATED0>(
            ilane, fac, nfac, tstride, nq0, nq1, nq2, ntbasis0, in, wsp1);

        PhysInterpFaceKernelTrace(ilane, nfac - fac, nq1, nq2, nq00, nq01,
                                  tbasis00, tbasis01, wsp2, wsp1, out,
                                  isCollocated00, isCollocated01);
    }
}

/**
 * @brief Glue for the direction-1 normal faces: the direction-1 twin of
 * PhysExtractFaceN0KernelTrace3D.
 *
 * Identical structure, with the normal stage
 * PhysExtractEndFacesN1KernelTrace3D and the tangential grids
 * `(nq0,nq2)` to `(nq10,nq11)`.
 *
 * @tparam END_PTS_COLLOCATED1    Direction-1 volume rule contains the
 *                      domain endpoints.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   fac         Position of the first face within direction 1.
 * @param   nfac        One past the position of the last face.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis1    eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   nq10,nq11   Trace quadrature points of the two tangential
 *                      directions of a direction-1 face.
 * @param   tbasis10,tbasis11   eInterp tables to those trace points.
 * @param   wsp1        Face values between the two stages,
 *                      `(nfac - fac) * nq2 * nq0` entries.
 * @param   wsp2        Scratch of the fully general tangential stage.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Face values on the trace grid, warp interleaved.
 * @param   isCollocated10,isCollocated11   That tangential direction's trace
 *                      points coincide with the volume points it comes
 *                      from.
 */
template <bool END_PTS_COLLOCATED1, typename TData>
NEK_DEVICE_INLINE static void PhysExtractFaceN1KernelTrace3D(
    const unsigned int ilane, const unsigned int fac, const unsigned int nfac,
    const unsigned int tstride, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT ntbasis1,
    const unsigned int nq10, const unsigned int nq11,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    TData *NEK_RESTRICT wsp1, TData *NEK_RESTRICT wsp2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated10, const bool isCollocated11)
{
    if (isCollocated10 && isCollocated11)
    {
        PhysExtractEndFacesN1KernelTrace3D<END_PTS_COLLOCATED1>(
            ilane, fac, nfac, tstride, nq0, nq1, nq2, ntbasis1, in, out);
    }
    else
    {
        PhysExtractEndFacesN1KernelTrace3D<END_PTS_COLLOCATED1>(
            ilane, fac, nfac, tstride, nq0, nq1, nq2, ntbasis1, in, wsp1);

        PhysInterpFaceKernelTrace(ilane, nfac - fac, nq0, nq2, nq10, nq11,
                                  tbasis10, tbasis11, wsp2, wsp1, out,
                                  isCollocated10, isCollocated11);
    }
}

/**
 * @brief Glue for the direction-2 normal faces: the direction-2 twin of
 * PhysExtractFaceN0KernelTrace3D.
 *
 * Identical structure, with the normal stage
 * PhysExtractEndFacesN2KernelTrace3D and the tangential grids
 * `(nq0,nq1)` to `(nq20,nq21)`.
 *
 * @tparam END_PTS_COLLOCATED2    Direction-2 volume rule contains the
 *                      domain endpoints.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   fac         Position of the first face within direction 2.
 * @param   nfac        One past the position of the last face.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction: the
 *                      callers pass @c nm0, @c nm1 and @c nm2.
 * @param   ntbasis2    eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   nq20,nq21   Trace quadrature points of the two tangential
 *                      directions of a direction-2 face.
 * @param   tbasis20,tbasis21   eInterp tables to those trace points.
 * @param   wsp1        Face values between the two stages,
 *                      `(nfac - fac) * nq1 * nq0` entries.
 * @param   wsp2        Scratch of the fully general tangential stage.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Face values on the trace grid, warp interleaved.
 * @param   isCollocated20,isCollocated21   That tangential direction's trace
 *                      points coincide with the volume points it comes
 *                      from.
 */
template <bool END_PTS_COLLOCATED2, typename TData>
NEK_DEVICE_INLINE static void PhysExtractFaceN2KernelTrace3D(
    const unsigned int ilane, const unsigned int fac, const unsigned int nfac,
    const unsigned int tstride, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT ntbasis2,
    const unsigned int nq20, const unsigned int nq21,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    TData *NEK_RESTRICT wsp1, TData *NEK_RESTRICT wsp2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated20, const bool isCollocated21)
{
    if (isCollocated20 && isCollocated21)
    {
        PhysExtractEndFacesN2KernelTrace3D<END_PTS_COLLOCATED2>(
            ilane, fac, nfac, tstride, nq0, nq1, nq2, ntbasis2, in, out);
    }
    else
    {
        PhysExtractEndFacesN2KernelTrace3D<END_PTS_COLLOCATED2>(
            ilane, fac, nfac, tstride, nq0, nq1, nq2, ntbasis2, in, wsp1);

        PhysInterpFaceKernelTrace(ilane, nfac - fac, nq0, nq1, nq20, nq21,
                                  tbasis20, tbasis21, wsp2, wsp1, out,
                                  isCollocated20, isCollocated21);
    }
}

/**
 * @brief Turn the runtime endpoint-collocation flag of direction 0 into
 * the template argument PhysExtractFaceN0KernelTrace3D branches on.
 *
 * Both arms are written out so that both instantiations of the glue and
 * of its normal-stage leaf exist. Nothing else differs between them.
 *
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   fac,nfac    Half-open window of faces within direction 0.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   nq00,nq01   Trace quadrature points of the two tangential
 *                      directions.
 * @param   tbasis00,tbasis01   eInterp tables to those trace points.
 * @param   wsp0,wsp1   Face workspace and general-contraction scratch,
 *                      handed on as the glue's @c wsp1 and @c wsp2.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Face values on the trace grid, warp interleaved.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags.
 * @param   endPtsCollocated0   Direction-0 volume rule contains the
 *                      domain endpoints.
 */
template <typename TData>
NEK_DEVICE_INLINE static void DispatchPhysExtractFaceN0KernelTrace3D(
    const unsigned int ilane, const unsigned int fac, const unsigned int nfac,
    const unsigned int tstride, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const TData *NEK_RESTRICT ntbasis0,
    const unsigned int nq00, const unsigned int nq01,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated00, const bool isCollocated01,
    const bool endPtsCollocated0)
{
    if (endPtsCollocated0)
    {
        PhysExtractFaceN0KernelTrace3D<true>(
            ilane, fac, nfac, tstride, nm0, nm1, nm2, ntbasis0, nq00, nq01,
            tbasis00, tbasis01, wsp0, wsp1, in, out, isCollocated00,
            isCollocated01);
    }
    else
    {
        PhysExtractFaceN0KernelTrace3D<false>(
            ilane, fac, nfac, tstride, nm0, nm1, nm2, ntbasis0, nq00, nq01,
            tbasis00, tbasis01, wsp0, wsp1, in, out, isCollocated00,
            isCollocated01);
    }
}

/**
 * @brief Turn the runtime endpoint-collocation flag of direction 1 into
 * the template argument PhysExtractFaceN1KernelTrace3D branches on.
 *
 * The direction-1 twin of DispatchPhysExtractFaceN0KernelTrace3D.
 *
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   fac,nfac    Half-open window of faces within direction 1.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   ntbasis1    eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   nq10,nq11   Trace quadrature points of the two tangential
 *                      directions.
 * @param   tbasis10,tbasis11   eInterp tables to those trace points.
 * @param   wsp0,wsp1   Face workspace and general-contraction scratch,
 *                      handed on as the glue's @c wsp1 and @c wsp2.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Face values on the trace grid, warp interleaved.
 * @param   isCollocated10,isCollocated11   Tangential collocation flags.
 * @param   endPtsCollocated1   Direction-1 volume rule contains the
 *                      domain endpoints.
 */
template <typename TData>
NEK_DEVICE_INLINE static void DispatchPhysExtractFaceN1KernelTrace3D(
    const unsigned int ilane, const unsigned int fac, const unsigned int nfac,
    const unsigned int tstride, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const TData *NEK_RESTRICT ntbasis1,
    const unsigned int nq10, const unsigned int nq11,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated10, const bool isCollocated11,
    const bool endPtsCollocated1)
{
    if (endPtsCollocated1)
    {
        PhysExtractFaceN1KernelTrace3D<true>(
            ilane, fac, nfac, tstride, nm0, nm1, nm2, ntbasis1, nq10, nq11,
            tbasis10, tbasis11, wsp0, wsp1, in, out, isCollocated10,
            isCollocated11);
    }
    else
    {
        PhysExtractFaceN1KernelTrace3D<false>(
            ilane, fac, nfac, tstride, nm0, nm1, nm2, ntbasis1, nq10, nq11,
            tbasis10, tbasis11, wsp0, wsp1, in, out, isCollocated10,
            isCollocated11);
    }
}

/**
 * @brief Turn the runtime endpoint-collocation flag of direction 2 into
 * the template argument PhysExtractFaceN2KernelTrace3D branches on.
 *
 * The direction-2 twin of DispatchPhysExtractFaceN0KernelTrace3D.
 *
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   fac,nfac    Half-open window of faces within direction 2.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   ntbasis2    eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   nq20,nq21   Trace quadrature points of the two tangential
 *                      directions.
 * @param   tbasis20,tbasis21   eInterp tables to those trace points.
 * @param   wsp0,wsp1   Face workspace and general-contraction scratch,
 *                      handed on as the glue's @c wsp1 and @c wsp2.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         Face values on the trace grid, warp interleaved.
 * @param   isCollocated20,isCollocated21   Tangential collocation flags.
 * @param   endPtsCollocated2   Direction-2 volume rule contains the
 *                      domain endpoints.
 */
template <typename TData>
NEK_DEVICE_INLINE static void DispatchPhysExtractFaceN2KernelTrace3D(
    const unsigned int ilane, const unsigned int fac, const unsigned int nfac,
    const unsigned int tstride, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const TData *NEK_RESTRICT ntbasis2,
    const unsigned int nq20, const unsigned int nq21,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const bool isCollocated20, const bool isCollocated21,
    const bool endPtsCollocated2)
{
    if (endPtsCollocated2)
    {
        PhysExtractFaceN2KernelTrace3D<true>(
            ilane, fac, nfac, tstride, nm0, nm1, nm2, ntbasis2, nq20, nq21,
            tbasis20, tbasis21, wsp0, wsp1, in, out, isCollocated20,
            isCollocated21);
    }
    else
    {
        PhysExtractFaceN2KernelTrace3D<false>(
            ilane, fac, nfac, tstride, nm0, nm1, nm2, ntbasis2, nq20, nq21,
            tbasis20, tbasis21, wsp0, wsp1, in, out, isCollocated20,
            isCollocated21);
    }
}

/**
 * @brief All faces of one three-dimensional element: the three normal
 * directions in packed trace order.
 *
 * The bulk worker. The direction-0 pair goes into the first
 * `2 * nq00 * nq01` entries of @p out, then the direction-1 faces, a
 * pair for every shape but the tetrahedron, then the direction-2 faces,
 * a pair for the hexahedron and a single face otherwise. Each group is
 * dispatched as one window, so within a pair both faces come out of a
 * single pass of the normal stage.
 *
 * @tparam SHAPE_TYPE Hex, Prism, NodalPrism, Pyr, Tet or NodalTet.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nq00,nq01,nq10,nq11,nq20,nq21   Trace quadrature points, per
 *                      normal direction and tangential direction.
 * @param   ntbasis0,ntbasis1,ntbasis2      eInterp tables
 *                      \f$h_p(\pm 1)\f$ per direction.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points.
 * @param   wsp0        Face values between the two stages.
 * @param   wsp1        Scratch of the fully general tangential stage.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         All of this element's trace values, warp
 *                      interleaved.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2
 *                      Volume rule of each direction contains the domain
 *                      endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TData>
NEK_DEVICE_INLINE static void PhysTraceExtract3DSumFacKernelCore(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq00, const unsigned int nq01,
    const unsigned int nq10, const unsigned int nq11, const unsigned int nq20,
    const unsigned int nq21, const TData *ntbasis0, const TData *ntbasis1,
    const TData *ntbasis2, const TData *tbasis00, const TData *tbasis01,
    const TData *tbasis10, const TData *tbasis11, const TData *tbasis20,
    const TData *tbasis21, TData *wsp0, TData *wsp1, const TData *in,
    TData *out, const bool isCollocated00, const bool isCollocated01,
    const bool isCollocated10, const bool isCollocated11,
    const bool isCollocated20, const bool isCollocated21,
    const bool endPtsCollocated0, const bool endPtsCollocated1,
    const bool endPtsCollocated2)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    // One ntbasis row per trace of the direction. A call covering part of
    // a direction still indexes at that stride, so it cannot be taken
    // from the loop bound.
    constexpr unsigned int tstride0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned int tstride1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned int tstride2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];

    // Faces with dir-0 normals (trace group tr=2).
    DispatchPhysExtractFaceN0KernelTrace3D(
        ilane, 0, tstride0, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
        tbasis00, tbasis01, wsp0, wsp1, in, out, isCollocated00, isCollocated01,
        endPtsCollocated0);

    unsigned int offset = tstride0 * nq00 * nq01;

    // Faces with dir-1 normals.
    DispatchPhysExtractFaceN1KernelTrace3D(
        ilane, 0, tstride1, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
        tbasis10, tbasis11, wsp0, wsp1, in, out + offset * warpsize,
        isCollocated10, isCollocated11, endPtsCollocated1);
    offset += tstride1 * nq10 * nq11;

    // Faces with dir-2 normals (trace group tr=0).
    DispatchPhysExtractFaceN2KernelTrace3D(
        ilane, 0, tstride2, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21,
        tbasis20, tbasis21, wsp0, wsp1, in, out + offset * warpsize,
        isCollocated20, isCollocated21, endPtsCollocated2);
}

/**
 * @brief Launch entry point for a block of three-dimensional
 * elements: every face of every element.
 *
 * Grid-strides over the block's elements, offsets @p in by @p nmTot
 * values per element and @p out by the element's total trace count
 * `nface0 * nq00 * nq01 + nface1 * nq10 * nq11 + nface2 * nq20 * nq21`,
 * then
 * splits the per-element workspace: @p wsp0 takes `2 * max_wsp_size`
 * values and @p wsp1 the `max_wsp_size` after them, where
 * @c max_wsp_size is the largest of the six face-sized products the
 * three directions can need, `nm1*nm2*2`, `nm0*nm2*2`, `nm0*nm1*2`,
 * `nm2*nq00*2`, `nm2*nq10*2` and `nm1*nq20*2`. The block operator sizes
 * its allocation with the same expression.
 *
 * @tparam SHAPE_TYPE       Hex, Prism, NodalPrism, Pyr, Tet or NodalTet.
 * @tparam TTraceSizeParameter3D    NonTemplatedTraceSizeParameter3D or
 *                          a TemplatedTraceSizeParameter3D
 *                          instantiation.
 * @tparam TthreadBlock     Thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam3D Volume quadrature points per direction, nm0(),
 *                      nm1() and nm2(), and the trace quadrature
 *                      points nq00() to nq21(), per normal direction
 *                      and tangential direction.
 * @param   nelmt       Elements in the block, including padding.
 * @param   nmTot       Entries per element of @p in.
 * @param   ntbasis0,ntbasis1,ntbasis2      eInterp tables
 *                      \f$h_p(\pm 1)\f$ per direction.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points.
 * @param   in          Volume field of the block, warp interleaved.
 * @param   out         Trace field of the block, warp interleaved.
 * @param   wsp         Workspace of the block, `3 * max_wsp_size`
 *                      entries per element.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2
 *                      Volume rule of each direction contains the domain
 *                      endpoints.
 * @param   shmemptr    Dynamic shared memory; unused.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter3D,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void PhysTraceExtractKernelLauncher(
    const TTraceSizeParameter3D sizeParam3D, const size_t nelmt,
    const unsigned int nmTot, const TData *NEK_RESTRICT ntbasis0,
    const TData *NEK_RESTRICT ntbasis1, const TData *NEK_RESTRICT ntbasis2,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, const bool isCollocated00,
    const bool isCollocated01, const bool isCollocated10,
    const bool isCollocated11, const bool isCollocated20,
    const bool isCollocated21, const bool endPtsCollocated0,
    const bool endPtsCollocated1, const bool endPtsCollocated2,
    [[maybe_unused]] unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0  = sizeParam3D.nm0();
    const unsigned int nm1  = sizeParam3D.nm1();
    const unsigned int nm2  = sizeParam3D.nm2();
    const unsigned int nq00 = sizeParam3D.nq00();
    const unsigned int nq01 = sizeParam3D.nq01();
    const unsigned int nq10 = sizeParam3D.nq10();
    const unsigned int nq11 = sizeParam3D.nq11();
    const unsigned int nq20 = sizeParam3D.nq20();
    const unsigned int nq21 = sizeParam3D.nq21();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int nqTotOut = sizeParam3D.template nqTotTrace<SHAPE_TYPE>();

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTotOut * (nelmt * c + warpsize * iwarp);

        const unsigned int max_wsp_size =
            PhysTraceExtractBlockSize(sizeParam3D);
        TData *wspptr0 =
            wsp + (3u * max_wsp_size) * (nelmt * c + warpsize * iwarp);
        TData *wspptr1 = wspptr0 + (2u * max_wsp_size) * warpsize;

        PhysTraceExtract3DSumFacKernelCore<SHAPE_TYPE>(
            ilane, nm0, nm1, nm2, nq00, nq01, nq10, nq11, nq20, nq21, ntbasis0,
            ntbasis1, ntbasis2, tbasis00, tbasis01, tbasis10, tbasis11,
            tbasis20, tbasis21, wspptr0, wspptr1, inptr, outptr, isCollocated00,
            isCollocated01, isCollocated10, isCollocated11, isCollocated20,
            isCollocated21, endPtsCollocated0, endPtsCollocated1,
            endPtsCollocated2);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief One named face of one three-dimensional element: turn the face
 * id into a normal direction and a position, then extract.
 *
 * The single-trace worker. GetTraceFaceDispatch supplies the direction
 * and a one-face window, and the switch picks that direction's table,
 * trace counts and collocation flags. @p out already points at this
 * trace's slot, so the window is the only thing expressing the position
 * within the direction.
 *
 * @note An unrecognised @p face is ignored silently, leaving @p out
 * untouched: device code cannot raise a host error.
 *
 * @note Every window this route dispatches is one face wide, so the
 * normal stage cannot take its @c ntbasis row stride from the loop
 * bound and is handed it separately, as @c tstride; see
 * PhysExtractEndFacesN0KernelTrace3D.
 *
 * All nine collocation flags default to false, that is to the general
 * path, although the only caller passes every one of them.
 *
 * @tparam SHAPE_TYPE Hex, Prism, NodalPrism, Pyr, Tet or NodalTet.
 * @tparam TData      Floating-point type of the field data.
 *
 * @param   ilane       Lane index within the warp; selects the element.
 * @param   face        Nektar face id of the trace to extract.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nq00,nq01,nq10,nq11,nq20,nq21   Trace quadrature points, per
 *                      normal direction and tangential direction.
 * @param   ntbasis0,ntbasis1,ntbasis2      eInterp tables
 *                      \f$h_p(\pm 1)\f$ per direction.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points.
 * @param   wsp0        Face values between the two stages.
 * @param   wsp1        Scratch of the fully general tangential stage.
 * @param   in          Volume field of this lane, warp interleaved.
 * @param   out         This trace's values, warp interleaved.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2
 *                      Volume rule of each direction contains the domain
 *                      endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TData>
NEK_DEVICE_INLINE static void PhysTraceFaceExtractKernel(
    const unsigned int ilane, const unsigned int face, const unsigned int nm0,
    const unsigned int nm1, const unsigned int nm2, const unsigned int nq00,
    const unsigned int nq01, const unsigned int nq10, const unsigned int nq11,
    const unsigned int nq20, const unsigned int nq21,
    const TData *NEK_RESTRICT ntbasis0, const TData *NEK_RESTRICT ntbasis1,
    const TData *NEK_RESTRICT ntbasis2, const TData *NEK_RESTRICT tbasis00,
    const TData *NEK_RESTRICT tbasis01, const TData *NEK_RESTRICT tbasis10,
    const TData *NEK_RESTRICT tbasis11, const TData *NEK_RESTRICT tbasis20,
    const TData *NEK_RESTRICT tbasis21, TData *NEK_RESTRICT wsp0,
    TData *NEK_RESTRICT wsp1, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const bool isCollocated00 = false,
    const bool isCollocated01 = false, const bool isCollocated10 = false,
    const bool isCollocated11 = false, const bool isCollocated20 = false,
    const bool isCollocated21 = false, const bool endPtsCollocated0 = false,
    const bool endPtsCollocated1 = false, const bool endPtsCollocated2 = false)
{
    // One ntbasis row per trace of the direction. A call covering part of
    // a direction still indexes at that stride, so it cannot be taken
    // from the loop bound.
    constexpr unsigned int tstride0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned int tstride1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned int tstride2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];

    TraceExtractDispatch dispatch{};
    if (!GetTraceFaceDispatch<SHAPE_TYPE>(face, dispatch))
    {
        return;
    }

    switch (dispatch.normalDir)
    {
        case 0:
        {
            DispatchPhysExtractFaceN0KernelTrace3D(
                ilane, dispatch.fac, dispatch.nfac, tstride0, nm0, nm1, nm2,
                ntbasis0, nq00, nq01, tbasis00, tbasis01, wsp0, wsp1, in, out,
                isCollocated00, isCollocated01, endPtsCollocated0);
            break;
        }
        case 1:
        {
            DispatchPhysExtractFaceN1KernelTrace3D(
                ilane, dispatch.fac, dispatch.nfac, tstride1, nm0, nm1, nm2,
                ntbasis1, nq10, nq11, tbasis10, tbasis11, wsp0, wsp1, in, out,
                isCollocated10, isCollocated11, endPtsCollocated1);
            break;
        }
        case 2:
        {
            DispatchPhysExtractFaceN2KernelTrace3D(
                ilane, dispatch.fac, dispatch.nfac, tstride2, nm0, nm1, nm2,
                ntbasis2, nq20, nq21, tbasis20, tbasis21, wsp0, wsp1, in, out,
                isCollocated20, isCollocated21, endPtsCollocated2);
            break;
        }
        default:
            break;
    }
}

/**
 * @brief Launch entry point for one named face of a block of
 * three-dimensional elements.
 *
 * The single-trace counterpart of PhysTraceExtractKernelLauncher. It
 * splits the per-element workspace the same way and offsets @p out to
 * `(numDataOut * iwarp + outOffset) * warpsize` so that the caller can
 * place each trace in turn within the output block, then calls
 * PhysTraceFaceExtractKernel.
 *
 * @tparam SHAPE_TYPE       Hex, Prism, Pyr or Tet. The block operator
 *                          instantiates the nodal shapes under their
 *                          parent enumerator on this route.
 * @tparam TTraceSizeParameter3D    NonTemplatedTraceSizeParameter3D or
 *                          a TemplatedTraceSizeParameter3D
 *                          instantiation.
 * @tparam TthreadBlock     Thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   traceid     Nektar face id of the trace to extract.
 * @param   sizeParam3D Volume quadrature points per direction, nm0(),
 *                      nm1() and nm2(), and the trace quadrature
 *                      points nq00() to nq21(), per normal direction
 *                      and tangential direction.
 * @param   nelmt       Elements in the block, including padding.
 * @param   nmTot       Entries per element of @p in.
 * @param   numDataOut    Entries per element of @p out.
 * @param   outOffset   Where this trace's block sits within an element's
 *                      output.
 * @param   ntbasis0,ntbasis1,ntbasis2      eInterp tables
 *                      \f$h_p(\pm 1)\f$ per direction.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points.
 * @param   in          Volume field of the block, warp interleaved.
 * @param   out         Trace field of the block, warp interleaved.
 * @param   wsp         Workspace of the block, `3 * max_wsp_size`
 *                      entries per element.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2
 *                      Volume rule of each direction contains the domain
 *                      endpoints.
 * @param   shmemptr    Dynamic shared memory; unused.
 * @param   threadBlock Thread-block handle supplied by the launch macro.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter3D,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void PhysTraceExtractTraceKernelLauncher(
    const unsigned int traceid, const TTraceSizeParameter3D sizeParam3D,
    const size_t nelmt, const unsigned int nmTot, const unsigned int numDataOut,
    const unsigned int outOffset, const TData *NEK_RESTRICT ntbasis0,
    const TData *NEK_RESTRICT ntbasis1, const TData *NEK_RESTRICT ntbasis2,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, const bool isCollocated00,
    const bool isCollocated01, const bool isCollocated10,
    const bool isCollocated11, const bool isCollocated20,
    const bool isCollocated21, const bool endPtsCollocated0,
    const bool endPtsCollocated1, const bool endPtsCollocated2,
    [[maybe_unused]] unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0  = sizeParam3D.nm0();
    const unsigned int nm1  = sizeParam3D.nm1();
    const unsigned int nm2  = sizeParam3D.nm2();
    const unsigned int nq00 = sizeParam3D.nq00();
    const unsigned int nq01 = sizeParam3D.nq01();
    const unsigned int nq10 = sizeParam3D.nq10();
    const unsigned int nq11 = sizeParam3D.nq11();
    const unsigned int nq20 = sizeParam3D.nq20();
    const unsigned int nq21 = sizeParam3D.nq21();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + numDataOut * (nelmt * c + warpsize * iwarp) +
                        outOffset * warpsize;

        const unsigned int max_wsp_size =
            PhysTraceExtractBlockSize(sizeParam3D);
        TData *wspptr0 =
            wsp + (3u * max_wsp_size) * (nelmt * c + warpsize * iwarp);
        TData *wspptr1 = wspptr0 + (2u * max_wsp_size) * warpsize;

        PhysTraceFaceExtractKernel<SHAPE_TYPE>(
            ilane, traceid, nm0, nm1, nm2, nq00, nq01, nq10, nq11, nq20, nq21,
            ntbasis0, ntbasis1, ntbasis2, tbasis00, tbasis01, tbasis10,
            tbasis11, tbasis20, tbasis21, wspptr0, wspptr1, inptr, outptr,
            isCollocated00, isCollocated01, isCollocated10, isCollocated11,
            isCollocated20, isCollocated21, endPtsCollocated0,
            endPtsCollocated1, endPtsCollocated2);

        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
