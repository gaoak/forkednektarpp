///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractSerialAVXSumFacKernels.hpp
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
 * @file PhysTraceExtractSerialAVXSumFacKernels.hpp
 * @brief Serial/AVX kernels of the trace extraction: one SIMD
 * vector of elements per call.
 *
 * @details
 * PhysTraceExtract is the adjoint of IProductWRTPhysTrace. Where the
 * lift integrates trace data against the cardinal (hat) basis of the
 * volume quadrature grid, this operator samples a volume field at the
 * trace quadrature points of the element's traces. A field given at the
 * volume points is \f$u = \sum_{pq} u_{pq}\, h_p(\xi_0)\, h_q(\xi_1)\f$
 * in the Lagrange interpolants through those points, so its values on
 * the direction-1 trace pair are
 * \f[
 *   u(\xi^{tr}_{0i}, \pm 1) \;=\; \sum_q h_q(\pm 1) \sum_p u_{pq}\,
 *   h_p(\xi^{tr}_{0i}) ,
 * \f]
 * two contractions applied in turn: first the normal direction, which
 * reduces the volume block to one line (2D) or plane (3D) per trace,
 * then the remaining tangential directions, which interpolate that line
 * or plane onto the trace quadrature points. Every kernel in this file
 * is one of those two stages, or glue routing a trace to them. The
 * tables are the same @c eInterp tables IProductWRTPhysTrace consumes,
 * applied untransposed here and transposed there.
 *
 * @section pte_avx_naming Naming
 *
 * @c nm0, @c nm1 and @c nm2 hold @c GetNumPoints, that is volume
 * quadrature counts and @em not mode counts. Only the entry points and
 * the one-dimensional PhysTraceExtractSegKernel spell them that way: the
 * glue and the extraction kernels below take the very same volume counts
 * under the parameter names @c nq0, @c nq1 and @c nq2, and the two
 * interpolation kernels take them as @c nqfrom, or @c nqfrom0 and
 * @c nqfrom1, against the trace counts @c nqto, or @c nqto0 and
 * @c nqto1. A glue or extraction kernel's @c nq0 is therefore a volume
 * count while an entry point's @c nq00 is a trace count. The `nq<ab>`
 * form carries the normal direction as the first digit and the
 * tangential direction as the second, direction 0 always being the
 * lower-numbered
 * of the two remaining volume directions, so in three dimensions the
 * \f$N_0\f$ face runs over `(nm1, nm2)`, \f$N_1\f$ over `(nm0, nm2)` and
 * \f$N_2\f$ over `(nm0, nm1)`.
 *
 * @c ntbasis is the @c eInterp table \f$h_p(\pm 1)\f$ from the volume
 * grid of a direction to that direction's trace positions, and
 * @c tbasis the @c eInterp table \f$h_p(\xi^{tr}_i)\f$ from the volume
 * grid to the trace quadrature points. Neither is an expansion basis
 * evaluated anywhere. Both are stored tight, one row per volume point
 * and one column per output point, so @c tbasis is indexed
 * `tbasis[p * nqto + i]` and @c ntbasis `ntbasis[p * tstride + f]` with
 * @c tstride the number of traces that direction has: two for every
 * direction of a segment, quadrilateral and hexahedron, one for a
 * collapsed direction (direction 1 of a triangle and a tetrahedron,
 * direction 2 of prism, pyramid and tetrahedron).
 *
 * @section pte_avx_flags Fast paths
 *
 * Two flags select them, both decided at setup. The
 * `END_PTS_COLLOCATED<a>` family, derived from the
 * @c PointsTypeNumEndPts table, reach the leaf and glue kernels, and
 * PhysTraceExtractSegKernel, as template parameters, and the entry
 * points as runtime booleans that pick
 * the instantiation: the volume rule of normal direction @em a contains
 * the domain endpoints, \f$h_p(\pm 1)\f$ is a Kronecker delta and the
 * normal-direction stage degenerates to selecting a boundary plane. The
 * `isCollocated<ab>` family are runtime booleans, one per normal
 * direction and tangential direction: the trace quadrature points of that
 * direction are the volume points of the corresponding direction, the
 * interpolation table is the identity and the tangential stage
 * degenerates to a copy.
 *
 * @section pte_avx_layer Layering
 *
 * The file is banner-separated into four sections: the one-, two- and
 * three-dimensional kernels, each holding everything the launcher of
 * that dimension reaches, then the launchers themselves. Within a
 * dimension the order is bottom-up, leaves before the glue that calls
 * them and the per-trace entry point last. Across the sections the
 * layers are:
 *
 * - normal-direction extraction: PhysExtractEdgeN0EndPtsKernel and
 *   PhysExtractEdgeN1EndPtsKernel in two dimensions,
 *   PhysExtractEndFacesN0Kernel, PhysExtractEndFacesN1Kernel and
 *   PhysExtractEndFacesN2Kernel in three;
 * - tangential interpolation: PhysInterpEdgeKernel and
 *   PhysInterpFaceKernel;
 * - glue: PhysExtractEdgeN0Kernel, PhysExtractEdgeN1Kernel and
 *   PhysExtractFaceN0Kernel, PhysExtractFaceN1Kernel,
 *   PhysExtractFaceN2Kernel, which run the extraction into a workspace
 *   and then the interpolation, or let the extraction write straight to
 *   the output when the tangential directions are collocated;
 * - launchers: PhysTraceExtractKernelLauncher, an overload set of three
 *   arms taking the shape's size parameter, gathered at the end of the
 *   file in dimension order. The two- and three-dimensional arms walk
 *   every trace of an element themselves; the one-dimensional arm only
 *   dispatches onto PhysTraceExtractSegKernel, which is the whole
 *   one-dimensional path. PhysTraceExtractEdgeKernel and
 *   PhysTraceFaceExtractKernel are the per-trace entry points, turning
 *   one trace id into a direction and a position.
 *
 * @section pte_avx_layout Layout contracts
 *
 * - SIMD packing. Every buffer is a `simd_type` array, so one call
 *   handles `simd_type::width` elements at once and the tables, shared
 *   by all of them, stay `simd_type::scalarType`. On the Serial
 *   execution space `simd_type` is `tinysimd::scalarT<TData>`, a width-1
 *   vector, and the same kernels serve scalar code.
 * - Trace packing. The traces are written in pair order: the \f$N_0\f$
 *   pair first, then the \f$N_1\f$ pair or single, then \f$N_2\f$,
 *   trace-major within a pair. A trace of normal direction @em a
 *   contributes `nqa0` values in two dimensions and `nqa0 * nqa1` in
 *   three. The single-trace entry points write from index zero of their
 *   output and leave the trace offset to the caller.
 * - Nothing here accumulates except PhysTraceExtractSegKernel, whose
 *   @c APPEND parameter is the only one in the file. Every other kernel
 *   overwrites the entries it owns, so no caller needs to zero anything
 *   first.
 * - This header is included outside any namespace and declares no
 *   includes of its own; the names it needs -- the SIMD types, @c
 *   ASSERTL1, @c NEKERROR -- come from its single includer,
 *   PhysTraceExtractSerialAVXSumFac.hpp.
 *
 * @section pte_avx_state State of the paths
 *
 * The normal-direction kernels carry the @c ntbasis row stride as an
 * explicit @c tstride argument, separate from the face or edge loop
 * bound. The two coincide only when a call spans the whole direction,
 * which the bulk entry points do and the per-trace ones do not, so
 * taking one for the other reads the wrong table row on every partial
 * call. The entry points take the stride from
 * @c ShapeTypeNumTraceInDir, which is also what makes it right for the
 * collapsed directions of a triangle and a tetrahedron.
 *
 * @see PhysTraceExtractDeviceSumFacKernels.hpp for the device kernels of
 * the same decomposition, and
 * IProductWRTPhysTraceSerialAVXSumFacKernels.hpp for the adjoint
 * operation on this execution space.
 */

#pragma once

// For NEK_FORCE_INLINE, previously picked up by whichever header
// happened to include this one first.
#include "LibUtilities/BasicUtils/NekInline.hpp"
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractKernels.hpp>

#include <MultiRegions/ElmtOps/ElmtHelper.hpp>

namespace Nektar::MultiRegions::detail
{

// -----------------------------------------------------------------------------
//  ONE-DIMENSIONAL KERNELS
// -----------------------------------------------------------------------------

// Extract an element's segment expansion onto its two endpoint traces. This is
// the 1D counterpart of PhysTraceExtractEdgeKernel and keeps the operator free
// of any dependency on the shared BwdTrans kernels.
/**
 * @brief One-dimensional extraction: sample a segment's volume field at
 * its two endpoint traces.
 *
 * The traces of a segment are its two vertices, so there is no in-trace
 * direction and nothing to interpolate along: the whole operator is the
 * single contraction \f$v_f = \sum_p u_p\, h_p(\pm 1)\f$ against the
 * direction-0 table. Three branches cover it.
 * - @p END_PTS_COLLOCATED, decided at compile time: the volume rule
 *   contains both domain endpoints, \f$h_p(\pm 1)\f$ is a Kronecker
 *   delta and the first and last volume values drop straight into the
 *   two outputs. The debug assertion pins the two traces this assumes.
 * - @p isCollocated, decided at run time: the first @p nq0 volume values
 *   are copied one for one. The block operator's one-dimensional path
 *   never selects it, its constructor pushing a @c false collocation
 *   flag for the one-dimensional case.
 * - otherwise the table is applied in full,
 *   `out[i] = sum_p in[p] * basis0[p * nq0 + i]`. A segment has two
 *   traces in direction 0, so the @p nq0 the caller passes is also the
 *   table's row stride and the tight-table contract holds.
 *
 * This kernel is the whole one-dimensional path of this implementation.
 * It takes no part in the two- and three-dimensional families and calls
 * nothing outside itself. The per-trace one-dimensional entry has no
 * counterpart at all: the block operator raises an error for a segment
 * there.
 *
 * @tparam APPEND            Accumulate onto @p out instead of
 *                           overwriting it. The block operator
 *                           instantiates only the overwriting form.
 * @tparam END_PTS_COLLOCATED  Direction-0 volume rule contains the domain
 *                           endpoints.
 * @tparam simd_type         SIMD vector type, one element per lane.
 *
 * @param   nm0             Volume quadrature points of the segment.
 * @param   nq0             Trace points to produce: two, the segment's
 *                          vertices.
 * @param   basis0          eInterp table \f$h_p(\pm 1)\f$ of direction
 *                          0, indexed `basis0[p * nq0 + f]`. Unused on
 *                          the two fast paths.
 * @param   in              Volume field of this element group.
 * @param   out             The two endpoint values.
 * @param   isCollocated    Trace points coincide with the volume points.
 */
template <bool APPEND, bool END_PTS_COLLOCATED, typename simd_type>
NEK_FORCE_INLINE static void PhysTraceExtractSegKernel(
    const unsigned int nm0, const unsigned int nq0,
    [[maybe_unused]] const typename simd_type::scalarType *basis0,
    const simd_type *in, simd_type *out, const bool isCollocated = false)
{
    if constexpr (END_PTS_COLLOCATED)
    {
        // The endpoints are element quadrature points, so just pick them up.
        ASSERTL1(nq0 == 2, "This option only works for endpoint extraction");

        if constexpr (APPEND)
        {
            out[0] += in[0];
            out[1] += in[nm0 - 1];
        }
        else
        {
            out[0] = in[0];
            out[1] = in[nm0 - 1];
        }
    }
    else if (isCollocated)
    {
        for (unsigned int i = 0; i < nq0; ++i)
        {
            if constexpr (APPEND)
            {
                out[i] += in[i];
            }
            else
            {
                out[i] = in[i];
            }
        }
    }
    else // interpolate the element expansion onto the trace points
    {
        for (unsigned int i = 0; i < nq0; ++i)
        {
            simd_type tmp = in[0] * basis0[i];

            for (unsigned int p = 1; p < nm0; ++p)
            {
                tmp.fma(in[p], basis0[p * nq0 + i]);
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
    }
}

// -----------------------------------------------------------------------------
//  TWO-DIMENSIONAL KERNELS
// -----------------------------------------------------------------------------

/**
 * @brief Stage one in two dimensions: reduce the volume block to the
 * boundary lines of one or both direction-0 normal edges.
 *
 * For every edge in [@p edg, @p nedg) this applies the direction-0
 * contraction \f$v_q = \sum_p u_{pq}\, h_p(\pm 1)\f$, leaving one value
 * per direction-1 volume point. Under @p END_PTS_COLLOCATED0 the table is
 * a Kronecker delta and the contraction degenerates to reading the
 * boundary column `p = e * (nq0 - 1)`, which is why @p edg and @p nedg
 * are absolute positions within the direction, 0 for the lower edge and
 * 1 for the upper, rather than indices within this call.
 *
 * @p out receives @p nq1 values per edge, edge-major, starting at index
 * zero. On the general path of the glue kernel it is a workspace rather
 * than the trace field.
 *
 * @note The general arm indexes `ntbasis0[p * 2 + e]`, a hard-coded row
 * stride of two. Direction 0 carries a trace pair in both
 * two-dimensional shapes, so that does match the table here; the
 * direction-1 twin makes the same assumption where it does not hold.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the domain
 *                            endpoints.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   edg         Position of the first edge within direction 0.
 * @param   nedg        One past the position of the last edge.
 * @param   nq0,nq1     Volume quadrature points per direction; the glue
 *                      kernel forwards its own nq0 and nq1, which came
 *                      from the entry point's nm0 and nm1.
 * @param   ntbasis0    eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   in          Volume field of this element group.
 * @param   out         Extracted values, @p nq1 per edge.
 */
template <bool END_PTS_COLLOCATED0, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractEdgeN0EndPtsKernel(
    const unsigned edg, const unsigned nedg, const unsigned tstride,
    const unsigned nq0, const unsigned nq1,
    [[maybe_unused]] const typename simd_type::scalarType *ntbasis0,
    const simd_type *in, simd_type *out)
{
    if constexpr (END_PTS_COLLOCATED0) // copy end points values
    {
        for (unsigned e = edg, cnt_eq = 0; e < nedg; ++e)
        {
            unsigned offset = e * (nq0 - 1);
            for (unsigned q = 0; q < nq1; ++q, ++cnt_eq)
            {
                out[cnt_eq] = in[q * nq0 + offset];
            }
        }
    }
    else // interpolate to end points
    {
        for (unsigned e = edg, cnt_eq = 0; e < nedg; ++e)
        {
            for (unsigned q = 0; q < nq1; ++q, ++cnt_eq)
            {
                simd_type tmp = in[q * nq0] * ntbasis0[e];

                for (unsigned p = 1; p < nq0; ++p)
                {
                    tmp.fma(in[p + q * nq0], ntbasis0[p * tstride + e]);
                }
                out[cnt_eq] = tmp;
            }
        }
    }
}

/**
 * @brief Stage one in two dimensions: the direction-1 twin of
 * PhysExtractEdgeN0EndPtsKernel.
 *
 * Applies \f$v_p = \sum_q u_{pq}\, h_q(\pm 1)\f$ over the edges in
 * [@p edg, @p nedg), leaving one value per direction-0 volume point.
 * Under @p END_PTS_COLLOCATED1 the delta property reduces this to reading
 * the boundary row `q = e * (nq1 - 1)`, so again @p edg and @p nedg are
 * absolute positions within the direction. @p out receives @p nq0 values
 * per edge, edge-major, from index zero.
 *
 * @note The general arm indexes `ntbasis1[q * 2 + e]`, a hard-coded row
 * stride of two. That is the table's stride for a quadrilateral, whose
 * direction 1 carries a trace pair, but not for a triangle: there
 * direction 1 has a single trace, the eInterp table has one column, and
 * this arm reads the wrong entries and past the end of the table. A
 * triangle only reaches it if its direction-1 rule contains no domain
 * endpoints; the Radau-type rule the collapsed direction normally
 * carries sets @p END_PTS_COLLOCATED1 and takes the other arm.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the domain
 *                            endpoints.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   edg         Position of the first edge within direction 1.
 * @param   nedg        One past the position of the last edge.
 * @param   nq0,nq1     Volume quadrature points per direction; the glue
 *                      kernel forwards its own nq0 and nq1, which came
 *                      from the entry point's nm0 and nm1.
 * @param   ntbasis1    eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   in          Volume field of this element group.
 * @param   out         Extracted values, @p nq0 per edge.
 */
template <bool END_PTS_COLLOCATED1, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractEdgeN1EndPtsKernel(
    const unsigned edg, const unsigned nedg, const unsigned tstride,
    const unsigned nq0, const unsigned nq1,
    [[maybe_unused]] const typename simd_type::scalarType *ntbasis1,
    const simd_type *in, simd_type *out)
{
    if constexpr (END_PTS_COLLOCATED1) // copy end points values
    {
        for (unsigned e = edg, cnt_ep = 0; e < nedg; ++e)
        {
            unsigned offset = e * (nq0 * (nq1 - 1));
            for (unsigned p = 0; p < nq0; ++p, ++cnt_ep)
            {
                out[cnt_ep] = in[p + offset];
            }
        }
    }
    else // interpolate to end points
    {
        for (unsigned e = edg, cnt_ep = 0; e < nedg; ++e)
        {
            for (unsigned p = 0; p < nq0; ++p, ++cnt_ep)
            {
                simd_type tmp = in[p] * ntbasis1[e];

                for (unsigned q = 1; q < nq1; ++q)
                {
                    tmp.fma(in[p + q * nq0], ntbasis1[q * tstride + e]);
                }
                out[cnt_ep] = tmp;
            }
        }
    }
}

// Extract element to edges in normal to dir 1 direction
//
// for two edges can specify edg = 0; nedg =2
// for one edge can specify edg = 0, nedg = 1 or edge = 1, nedg=2
//
// nq0, nq1 are number of interior lagrange points in dir 0,1
// nq10 is the number of quadrature points in dir 1 normal trace
// tbasis10 is the points to interpolated to within trace with normal in dir = 1
// ntbasis1 is the interpolation of dir 1 points to end points (normal)
/**
 * @brief Glue for the direction-0 normal edges: extract, then
 * interpolate.
 *
 * Runs PhysExtractEdgeN0EndPtsKernel over the edges in [@p edg,
 * @p nedg) and, unless the in-edge direction is collocated, follows it
 * with PhysInterpEdgeKernel from the direction-1 volume points onto the
 * edge's @p nq00 trace points. When it is collocated the extraction
 * writes straight to @p out and @p wsp is untouched. The in-edge
 * direction of a direction-0 normal is direction 1, which is why the
 * interpolation is set up as `nq1 -> nq00` with the direction's tangential-0
 * table.
 *
 * @note @p isCollocated00 is not forwarded to PhysInterpEdgeKernel: the
 * interpolating branch is precisely the one in which the flag is false,
 * so that kernel is always entered on its general arm.
 *
 * @note The plain-comment header above this template names direction 1
 * throughout, and the table and count it names are the direction-1 ones.
 * This kernel serves the direction-0 normals.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the domain
 *                            endpoints; forwarded to stage one.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   edg             Position of the first edge within direction
 *                          0; 0 and @p nedg 2 covers the pair, and one
 *                          edge is either (0, 1) or (1, 2).
 * @param   nedg            One past the position of the last edge; at
 *                          most 2, as the debug assertion states.
 * @param   nq0,nq1         Volume quadrature points per direction; the
 *                          entry point passes its nm0 and nm1.
 * @param   ntbasis0        eInterp table \f$h_p(\pm 1)\f$ of direction
 *                          0.
 * @param   nq00            Trace quadrature points along a direction-0
 *                          normal edge.
 * @param   tbasis00        eInterp table onto those trace points.
 * @param   wsp             Scratch for the extracted values, @p nq1 per
 *                          edge; used only on the interpolating path.
 * @param   in              Volume field of this element group.
 * @param   out             Trace values, @p nq00 per edge on the
 *                          interpolating path and @p nq1 per edge on the
 *                          collocated one, where the two counts agree.
 * @param   isCollocated00  Edge trace points coincide with the
 *                          direction-1 volume points.
 */
template <bool END_PTS_COLLOCATED0, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractEdgeN0Kernel(
    const unsigned edg, const unsigned nedg, const unsigned tstride,
    const unsigned nq0, const unsigned nq1,
    const typename simd_type::scalarType *ntbasis0, const unsigned nq00,
    const typename simd_type::scalarType *tbasis00, simd_type *wsp,
    const simd_type *in, simd_type *out, const bool isCollocated00 = false)
{
    ASSERTL1(nedg <= 2, "nedg > 2 is not valid");

    if (isCollocated00)
    {
        PhysExtractEdgeN0EndPtsKernel<END_PTS_COLLOCATED0>(
            edg, nedg, tstride, nq0, nq1, ntbasis0, in, out);
    }
    else // interpolate
    {
        PhysExtractEdgeN0EndPtsKernel<END_PTS_COLLOCATED0>(
            edg, nedg, tstride, nq0, nq1, ntbasis0, in, wsp);

        PhysInterpEdgeKernel(nedg - edg, nq1, nq00, tbasis00, wsp, out);
    }
}

// Extract element to edges in normal to dir 1 direction
//
// for two edges can specify edg = 0; nedg =2
// for one edge can specify edg = 0, nedg = 1 or edge = 1, nedg=2
//
// nq0, nq1 are number of interior lagrange points in dir 0,1
// nq10 is the number of quadrature points in dir 1 normal trace
// tbasis10 is the points to interpolated to within trace with normal in dir = 1
// ntbasis1 is the interpolation of dir 1 points to end points (normal)
/**
 * @brief Glue for the direction-1 normal edges: the direction-1 twin of
 * PhysExtractEdgeN0Kernel.
 *
 * The in-edge direction of a direction-1 normal is direction 0, so the
 * extraction leaves @p nq0 values per edge and the interpolation runs
 * `nq0 -> nq10` with the direction's tangential-0 table. Everything else,
 * including the collocated shortcut and the unforwarded flag, matches
 * the direction-0 kernel.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the domain
 *                            endpoints; forwarded to stage one.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   edg             Position of the first edge within direction
 *                          1.
 * @param   nedg            One past the position of the last edge; at
 *                          most 2, as the debug assertion states. A
 *                          triangle has a single direction-1 edge and is
 *                          called with (0, 1).
 * @param   nq0,nq1         Volume quadrature points per direction; the
 *                          entry point passes its nm0 and nm1.
 * @param   ntbasis1        eInterp table \f$h_q(\pm 1)\f$ of direction
 *                          1.
 * @param   nq10            Trace quadrature points along a direction-1
 *                          normal edge.
 * @param   tbasis10        eInterp table onto those trace points.
 * @param   wsp             Scratch for the extracted values, @p nq0 per
 *                          edge; used only on the interpolating path.
 * @param   in              Volume field of this element group.
 * @param   out             Trace values, @p nq10 per edge on the
 *                          interpolating path and @p nq0 per edge on the
 *                          collocated one, where the two counts agree.
 * @param   isCollocated10  Edge trace points coincide with the
 *                          direction-0 volume points.
 */
template <bool END_PTS_COLLOCATED1, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractEdgeN1Kernel(
    const unsigned edg, const unsigned nedg, const unsigned tstride,
    const unsigned nq0, const unsigned nq1,
    const typename simd_type::scalarType *ntbasis1, const unsigned nq10,
    const typename simd_type::scalarType *tbasis10, simd_type *wsp,
    const simd_type *in, simd_type *out, const bool isCollocated10 = false)
{
    ASSERTL1(nedg <= 2, "nedg > 2 is not valid");

    if (isCollocated10)
    {
        PhysExtractEdgeN1EndPtsKernel<END_PTS_COLLOCATED1>(
            edg, nedg, tstride, nq0, nq1, ntbasis1, in, out);
    }
    else // enterpolate
    {
        PhysExtractEdgeN1EndPtsKernel<END_PTS_COLLOCATED1>(
            edg, nedg, tstride, nq0, nq1, ntbasis1, in, wsp);

        PhysInterpEdgeKernel(nedg - edg, nq0, nq10, tbasis10, wsp, out);
    }
}

/**
 * @brief Two-dimensional entry point for a single named edge: turn the
 * edge id into a normal direction and a position, then call the glue.
 *
 * The mapping is the face-to-(direction, position) table of the shape.
 * A quadrilateral pairs edges 3 and 1 in direction 0 and edges 0 and 2
 * in direction 1, so `0 -> (1, 0)`, `1 -> (0, 1)`, `2 -> (1, 1)` and
 * `3 -> (0, 0)`. A triangle keeps `2 -> (0, 0)` and `1 -> (0, 1)`, the
 * hypotenuse acting as the direction-0 upper end in the collapsed frame,
 * and has the single direction-1 edge `0 -> (1, 0)`; there is no
 * direction-1 upper edge, the collapsed vertex standing in its place.
 * Any other id raises a fatal error.
 *
 * The position becomes the (@c edg, @c nedg) pair the glue takes, so a
 * lower edge is passed as (0, 1) and an upper edge as (1, 2). @p out is
 * written from index zero; the block operator adds the trace offset to
 * the pointer it hands in.
 *
 * The two runtime endpoint booleans select the compile-time arms of the
 * glue, which is why each case appears twice; the two isCollocated flags
 * are forwarded as runtime arguments.
 *
 * @note A nodal triangle reaches this through the block operator's
 * single-trace entry, which names `LibUtilities::Tri` explicitly for
 * both triangle cases, so the triangle arm is taken. That is not true of
 * the bulk entry point of the same name; see the file note.
 *
 * @tparam SHAPE_TYPE Tri for the triangle arm, any other two-dimensional
 *                    shape for the quadrilateral arm.
 * @tparam simd_type  SIMD vector type, one element per lane.
 *
 * @param   edge                Edge id within the shape.
 * @param   nm0,nm1             Volume quadrature points per direction.
 * @param   nq00,nq10           Trace quadrature points of a direction-0
 *                              and a direction-1 normal edge.
 * @param   ntbasis0,ntbasis1   eInterp tables \f$h(\pm 1)\f$ per
 *                              direction.
 * @param   tbasis00,tbasis10   eInterp tables onto the trace points of a
 *                              direction-0 and a direction-1 normal
 *                              edge.
 * @param   wsp                 Scratch for the extracted values.
 * @param   in                  Volume field of this element group.
 * @param   out                 Trace values of this edge, from index
 *                              zero.
 * @param   isCollocated00      Direction-0 normal edge's trace points
 *                              are the direction-1 volume points.
 * @param   isCollocated10      Direction-1 normal edge's trace points
 *                              are the direction-0 volume points.
 * @param   endPtsCollocated0   Direction-0 volume rule contains the
 *                              domain endpoints.
 * @param   endPtsCollocated1   Direction-1 volume rule contains the
 *                              domain endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename simd_type>
NEK_FORCE_INLINE static void PhysTraceExtractEdgeKernel(
    const unsigned edge, const unsigned nm0, const unsigned nm1,
    const unsigned nq00, const unsigned nq10,
    const typename simd_type::scalarType *ntbasis0,
    const typename simd_type::scalarType *ntbasis1,
    const typename simd_type::scalarType *tbasis00,
    const typename simd_type::scalarType *tbasis10, simd_type *wsp,
    const simd_type *in, simd_type *out, const bool isCollocated00,
    const bool isCollocated10, const bool endPtsCollocated0,
    const bool endPtsCollocated1)
{
    // One ntbasis row per trace of the direction. A call covering part of
    // a direction still indexes at that stride, so it cannot be taken
    // from the loop bound.
    constexpr unsigned tstride0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned tstride1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

    if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        switch (edge)
        {
            case 0:
                if (endPtsCollocated1)
                {
                    PhysExtractEdgeN1Kernel<true>(0, 1, tstride1, nm0, nm1,
                                                  ntbasis1, nq10, tbasis10, wsp,
                                                  in, out, isCollocated10);
                }
                else
                {
                    PhysExtractEdgeN1Kernel<false>(
                        0, 1, tstride1, nm0, nm1, ntbasis1, nq10, tbasis10, wsp,
                        in, out, isCollocated10);
                }
                break;
            case 1:
                if (endPtsCollocated0)
                {
                    PhysExtractEdgeN0Kernel<true>(1, 2, tstride0, nm0, nm1,
                                                  ntbasis0, nq00, tbasis00, wsp,
                                                  in, out, isCollocated00);
                }
                else
                {
                    PhysExtractEdgeN0Kernel<false>(
                        1, 2, tstride0, nm0, nm1, ntbasis0, nq00, tbasis00, wsp,
                        in, out, isCollocated00);
                }
                break;
            case 2:
                if (endPtsCollocated0)
                {
                    PhysExtractEdgeN0Kernel<true>(0, 1, tstride0, nm0, nm1,
                                                  ntbasis0, nq00, tbasis00, wsp,
                                                  in, out, isCollocated00);
                }
                else
                {
                    PhysExtractEdgeN0Kernel<false>(
                        0, 1, tstride0, nm0, nm1, ntbasis0, nq00, tbasis00, wsp,
                        in, out, isCollocated00);
                }
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Unrecognised edge input");
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
                    PhysExtractEdgeN1Kernel<true>(0, 1, tstride1, nm0, nm1,
                                                  ntbasis1, nq10, tbasis10, wsp,
                                                  in, out, isCollocated10);
                }
                else
                {
                    PhysExtractEdgeN1Kernel<false>(
                        0, 1, tstride1, nm0, nm1, ntbasis1, nq10, tbasis10, wsp,
                        in, out, isCollocated10);
                }
                break;
            case 1:
                if (endPtsCollocated0)
                {
                    PhysExtractEdgeN0Kernel<true>(1, 2, tstride0, nm0, nm1,
                                                  ntbasis0, nq00, tbasis00, wsp,
                                                  in, out, isCollocated00);
                }
                else
                {
                    PhysExtractEdgeN0Kernel<false>(
                        1, 2, tstride0, nm0, nm1, ntbasis0, nq00, tbasis00, wsp,
                        in, out, isCollocated00);
                }
                break;
            case 2:
                if (endPtsCollocated1)
                {
                    PhysExtractEdgeN1Kernel<true>(1, 2, tstride1, nm0, nm1,
                                                  ntbasis1, nq10, tbasis10, wsp,
                                                  in, out, isCollocated10);
                }
                else
                {
                    PhysExtractEdgeN1Kernel<false>(
                        1, 2, tstride1, nm0, nm1, ntbasis1, nq10, tbasis10, wsp,
                        in, out, isCollocated10);
                }
                break;
            case 3:
                if (endPtsCollocated0)
                {
                    PhysExtractEdgeN0Kernel<true>(0, 1, tstride0, nm0, nm1,
                                                  ntbasis0, nq00, tbasis00, wsp,
                                                  in, out, isCollocated00);
                }
                else
                {
                    PhysExtractEdgeN0Kernel<false>(
                        0, 1, tstride0, nm0, nm1, ntbasis0, nq00, tbasis00, wsp,
                        in, out, isCollocated00);
                }
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Unrecognised edge input");
                break;
        }
    }
}

// -----------------------------------------------------------------------------
//  THREE-DIMENSIONAL KERNELS
// -----------------------------------------------------------------------------

/**
 * @brief Stage one in three dimensions: reduce the volume block to the
 * boundary planes of one or both direction-0 normal faces.
 *
 * For every face in [@p fac, @p nface) this applies the direction-0
 * contraction \f$v_{jk} = \sum_i u_{ijk}\, h_i(\pm 1)\f$, leaving one
 * value per point of the `(nq1, nq2)` volume plane. Under
 * @p END_PTS_COLLOCATED0 the table is a Kronecker delta and this reduces
 * to reading the boundary plane `i = f * (nq0 - 1)`, so @p fac and
 * @p nface are absolute positions within the direction, 0 for the lower
 * face and 1 for the upper.
 *
 * @p out receives `nq1 * nq2` values per face, face-major, from index
 * zero, with the direction-1 index varying fastest -- the ordering
 * PhysInterpFaceKernel expects when it is given `(nq1, nq2)` as its
 * source counts.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the domain
 *                            endpoints.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   fac         Position of the first face within direction 0.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of @p ntbasis0: the number of traces
 *                      the direction has, which is not @p nface when the
 *                      call covers only part of the direction.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction; the glue
 *                      kernel forwards its own nq0, nq1 and nq2, which
 *                      came from the entry point's nm0, nm1 and nm2.
 * @param   ntbasis0    eInterp table \f$h_i(\pm 1)\f$ of direction 0.
 * @param   in          Volume field of this element group.
 * @param   out         Extracted planes, `nq1 * nq2` per face.
 */
template <bool END_PTS_COLLOCATED0, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractEndFacesN0Kernel(
    const unsigned fac, const unsigned nface, const unsigned tstride,
    const unsigned nq0, const unsigned nq1, const unsigned nq2,
    [[maybe_unused]] const typename simd_type::scalarType *ntbasis0,
    const simd_type *in, simd_type *out)
{
    if constexpr (END_PTS_COLLOCATED0) // copy end faces
    {
        unsigned nq0nq1 = nq0 * nq1;
        for (unsigned f = fac, cnt_fkj = 0; f < nface; ++f)
        {
            unsigned offset = f * (nq0 - 1);

            for (unsigned k = 0; k < nq2; ++k)
            {
                unsigned offset_fk = k * nq0nq1 + offset;

                for (unsigned j = 0; j < nq1; ++j, ++cnt_fkj)
                {
                    out[cnt_fkj] = in[offset_fk + j * nq0];
                }
            }
        }
    }
    else // interpolate in dir 0
    {
        // No f * (nq0 - 1) offset here, unlike the collocated branch above.
        // There it selects the i = 0 or i = nq0 - 1 plane; here the whole
        // dir 0 line is summed for every face and the face is selected by
        // ntbasis0 alone. With the offset, face 1 started at the last point
        // and ran off the end of the row.
        unsigned nq0nq1 = nq0 * nq1;
        for (unsigned f = fac, cnt_fkj = 0; f < nface; ++f)
        {
            for (unsigned k = 0; k < nq2; ++k)
            {
                unsigned offset_fk = k * nq0nq1;
                for (unsigned j = 0; j < nq1; ++j, ++cnt_fkj)
                {
                    unsigned offset_fkp = offset_fk + j * nq0;
                    simd_type tmp       = in[offset_fkp] * ntbasis0[f];
                    for (unsigned p = 1; p < nq0; ++p)
                    {
                        tmp.fma(in[offset_fkp + p], ntbasis0[p * tstride + f]);
                    }
                    out[cnt_fkj] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Stage one in three dimensions: the direction-1 twin of
 * PhysExtractEndFacesN0Kernel.
 *
 * Applies \f$v_{ik} = \sum_j u_{ijk}\, h_j(\pm 1)\f$ over the faces in
 * [@p fac, @p nface), leaving one value per point of the `(nq0, nq2)`
 * volume plane, with the direction-0 index varying fastest. Under
 * @p END_PTS_COLLOCATED1 the delta property reduces this to reading the
 * boundary plane `j = f * (nq1 - 1)`.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the domain
 *                            endpoints.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   fac         Position of the first face within direction 1.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of @p ntbasis1: the number of traces
 *                      the direction has, which is not @p nface when the
 *                      call covers only part of the direction.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction; the glue
 *                      kernel forwards its own nq0, nq1 and nq2, which
 *                      came from the entry point's nm0, nm1 and nm2.
 * @param   ntbasis1    eInterp table \f$h_j(\pm 1)\f$ of direction 1.
 * @param   in          Volume field of this element group.
 * @param   out         Extracted planes, `nq0 * nq2` per face.
 */
template <bool END_PTS_COLLOCATED1, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractEndFacesN1Kernel(
    const unsigned fac, const unsigned nface, const unsigned tstride,
    const unsigned nq0, const unsigned nq1, const unsigned nq2,
    [[maybe_unused]] const typename simd_type::scalarType *ntbasis1,
    const simd_type *in, simd_type *out)
{
    if constexpr (END_PTS_COLLOCATED1) // copy end faces
    {
        unsigned nq0nq1 = nq0 * nq1;
        for (unsigned f = fac, cnt_fki = 0; f < nface; ++f)
        {
            unsigned offset = f * nq0 * (nq1 - 1);

            for (unsigned k = 0; k < nq2; ++k)
            {
                unsigned offset_ki = k * nq0nq1 + offset;
                for (unsigned i = 0; i < nq0; ++i, ++cnt_fki)
                {
                    out[cnt_fki] = in[i + offset_ki];
                }
            }
        }
    }
    else // interpolate in dir 1
    {
        // Two fixes against the collocated branch above. There is no
        // f * nq0 * (nq1 - 1) offset, for the same reason as dir 0. And the
        // q = 0 term needs the + i that the q >= 1 terms already carry,
        // otherwise every point on the face takes its leading contribution
        // from i = 0.
        unsigned nq0nq1 = nq0 * nq1;
        for (unsigned f = fac, cnt_fki = 0; f < nface; ++f)
        {
            for (unsigned k = 0; k < nq2; ++k)
            {
                unsigned offset_ki = k * nq0nq1;
                for (unsigned i = 0; i < nq0; ++i, ++cnt_fki)
                {
                    simd_type tmp = in[offset_ki + i] * ntbasis1[f];

                    for (unsigned q = 1; q < nq1; ++q)
                    {
                        tmp.fma(in[offset_ki + i + q * nq0],
                                ntbasis1[q * tstride + f]);
                    }
                    out[cnt_fki] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Stage one in three dimensions: the direction-2 twin of
 * PhysExtractEndFacesN0Kernel.
 *
 * Applies \f$v_{ij} = \sum_k u_{ijk}\, h_k(\pm 1)\f$ over the faces in
 * [@p fac, @p nface), leaving one value per point of the `(nq0, nq1)`
 * volume plane, with the direction-0 index varying fastest. Under
 * @p END_PTS_COLLOCATED2 the delta property reduces this to reading the
 * boundary plane `k = f * (nq2 - 1)`. The loop variable is named @c k
 * here although it is the face position, the direction-2 index being the
 * one summed over.
 *
 * @tparam END_PTS_COLLOCATED2  Direction-2 volume rule contains the domain
 *                            endpoints.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   fac         Position of the first face within direction 2.
 * @param   nface       One past the position of the last face.
 * @param   tstride     Row stride of @p ntbasis2: the number of traces
 *                      the direction has, which is not @p nface when the
 *                      call covers only part of the direction.
 * @param   nq0,nq1,nq2 Volume quadrature points per direction; the glue
 *                      kernel forwards its own nq0, nq1 and nq2, which
 *                      came from the entry point's nm0, nm1 and nm2.
 * @param   ntbasis2    eInterp table \f$h_k(\pm 1)\f$ of direction 2.
 * @param   in          Volume field of this element group.
 * @param   out         Extracted planes, `nq0 * nq1` per face.
 */
template <bool END_PTS_COLLOCATED2, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractEndFacesN2Kernel(
    const unsigned fac, const unsigned nface, const unsigned tstride,
    const unsigned nq0, const unsigned nq1, const unsigned nq2,
    [[maybe_unused]] const typename simd_type::scalarType *ntbasis2,
    const simd_type *in, simd_type *out)
{
    if constexpr (END_PTS_COLLOCATED2) // copy end faces
    {
        for (unsigned k = fac, cnt_kji = 0; k < nface; ++k)
        {
            unsigned offset = k * nq0 * nq1 * (nq2 - 1);
            for (unsigned j = 0, cnt_ji = 0; j < nq1; ++j)
            {
                for (unsigned i = 0; i < nq0; ++i, ++cnt_kji, ++cnt_ji)
                {
                    out[cnt_kji] = in[cnt_ji + offset];
                }
            }
        }
    }
    else // interplolate in dir 2
    {
        unsigned nq01 = nq0 * nq1;
        for (unsigned k = fac, cnt_kji = 0; k < nface; ++k)
        {
            for (unsigned j = 0, cnt_ji = 0; j < nq1; ++j)
            {
                for (unsigned i = 0; i < nq0; ++i, ++cnt_kji, ++cnt_ji)
                {
                    simd_type tmp = in[cnt_ji] * ntbasis2[k];

                    for (unsigned r = 1; r < nq2; ++r)
                    {
                        tmp.fma(in[cnt_ji + r * nq01],
                                ntbasis2[r * tstride + k]);
                    }
                    out[cnt_kji] = tmp;
                }
            }
        }
    }
}

/**
 * @brief Glue for the direction-0 normal faces: extract, then
 * interpolate.
 *
 * Runs PhysExtractEndFacesN0Kernel over the faces in [@p fac, @p nfac)
 * and, unless both tangential directions are collocated, follows it with
 * PhysInterpFaceKernel from the `(nq1, nq2)` volume plane onto the
 * face's `(nq00, nq01)` trace grid. When both are collocated the
 * extraction writes straight to @p out and neither workspace is
 * touched. The tangential directions of a direction-0 normal are
 * directions 1 and 2, which is why the interpolation is set up as
 * `(nq1, nq2) -> (nq00, nq01)` with the direction's two tangential tables.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the domain
 *                            endpoints; forwarded to stage one.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   fac             Position of the first face within direction
 *                          0; (0, 2) covers the pair, one face is either
 *                          (0, 1) or (1, 2).
 * @param   nfac            One past the position of the last face.
 * @param   tstride         Row stride of the @c ntbasis table, forwarded
 *                          to stage one: the number of traces the
 *                          direction has.
 * @param   nq0,nq1,nq2     Volume quadrature points per direction; the
 *                          entry point passes its nm0, nm1 and nm2.
 * @param   ntbasis0        eInterp table \f$h_i(\pm 1)\f$ of direction
 *                          0.
 * @param   nq00,nq01       Trace quadrature points of a direction-0
 *                          normal face, per tangential direction.
 * @param   tbasis00        eInterp table of tangential direction 0 (volume
 *                          direction 1).
 * @param   tbasis01        eInterp table of tangential direction 1 (volume
 *                          direction 2).
 * @param   wsp1            Scratch for the extracted planes,
 *                          `nq1 * nq2` per face.
 * @param   wsp2            Scratch for the tangential-0 interpolation pass.
 * @param   in              Volume field of this element group.
 * @param   out             Trace values, `nq00 * nq01` per face.
 * @param   isCollocated00  Tangential-0 trace points are the direction-1
 *                          volume points.
 * @param   isCollocated01  Tangential-1 trace points are the direction-2
 *                          volume points.
 */
template <bool END_PTS_COLLOCATED0, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractFaceN0Kernel(
    const unsigned fac, const unsigned nfac, const unsigned tstride,
    const unsigned nq0, const unsigned nq1, const unsigned nq2,
    const typename simd_type::scalarType *ntbasis0, const unsigned nq00,
    const unsigned nq01, const typename simd_type::scalarType *tbasis00,
    const typename simd_type::scalarType *tbasis01, simd_type *wsp1,
    simd_type *wsp2, const simd_type *in, simd_type *out,
    const bool isCollocated00, const bool isCollocated01)
{
    if (isCollocated00 && isCollocated01) // can just extract points directly
    {
        PhysExtractEndFacesN0Kernel<END_PTS_COLLOCATED0>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis0, in, out);
    }
    else
    {
        PhysExtractEndFacesN0Kernel<END_PTS_COLLOCATED0>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis0, in, wsp1);

        PhysInterpFaceKernel(nfac - fac, nq1, nq2, nq00, nq01, tbasis00,
                             tbasis01, wsp2, wsp1, out, isCollocated00,
                             isCollocated01);
    }
}

/**
 * @brief Glue for the direction-1 normal faces: the direction-1 twin of
 * PhysExtractFaceN0Kernel.
 *
 * The tangential directions of a direction-1 normal are directions 0 and
 * 2, so the interpolation runs `(nq0, nq2) -> (nq10, nq11)`. Everything
 * else, including the both-collocated shortcut, matches the direction-0
 * kernel.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the domain
 *                            endpoints; forwarded to stage one.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   fac             Position of the first face within direction
 *                          1. A tetrahedron has a single direction-1
 *                          face and is called with (0, 1).
 * @param   nfac            One past the position of the last face.
 * @param   tstride         Row stride of the @c ntbasis table, forwarded
 *                          to stage one: the number of traces the
 *                          direction has.
 * @param   nq0,nq1,nq2     Volume quadrature points per direction; the
 *                          entry point passes its nm0, nm1 and nm2.
 * @param   ntbasis1        eInterp table \f$h_j(\pm 1)\f$ of direction
 *                          1.
 * @param   nq10,nq11       Trace quadrature points of a direction-1
 *                          normal face, per tangential direction.
 * @param   tbasis10        eInterp table of tangential direction 0 (volume
 *                          direction 0).
 * @param   tbasis11        eInterp table of tangential direction 1 (volume
 *                          direction 2).
 * @param   wsp1            Scratch for the extracted planes,
 *                          `nq0 * nq2` per face.
 * @param   wsp2            Scratch for the tangential-0 interpolation pass.
 * @param   in              Volume field of this element group.
 * @param   out             Trace values, `nq10 * nq11` per face.
 * @param   isCollocated10  Tangential-0 trace points are the direction-0
 *                          volume points.
 * @param   isCollocated11  Tangential-1 trace points are the direction-2
 *                          volume points.
 */
template <bool END_PTS_COLLOCATED1, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractFaceN1Kernel(
    const unsigned fac, const unsigned nfac, const unsigned tstride,
    const unsigned nq0, const unsigned nq1, const unsigned nq2,
    const typename simd_type::scalarType *ntbasis1, const unsigned nq10,
    const unsigned nq11, const typename simd_type::scalarType *tbasis10,
    const typename simd_type::scalarType *tbasis11, simd_type *wsp1,
    simd_type *wsp2, const simd_type *in, simd_type *out,
    const bool isCollocated10, const bool isCollocated11)
{
    if (isCollocated10 && isCollocated11) // can just extract points directly
    {
        PhysExtractEndFacesN1Kernel<END_PTS_COLLOCATED1>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis1, in, out);
    }
    else
    {
        PhysExtractEndFacesN1Kernel<END_PTS_COLLOCATED1>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis1, in, wsp1);

        PhysInterpFaceKernel(nfac - fac, nq0, nq2, nq10, nq11, tbasis10,
                             tbasis11, wsp2, wsp1, out, isCollocated10,
                             isCollocated11);
    }
}

/**
 * @brief Glue for the direction-2 normal faces: the direction-2 twin of
 * PhysExtractFaceN0Kernel.
 *
 * The tangential directions of a direction-2 normal are directions 0 and
 * 1, so the interpolation runs `(nq0, nq1) -> (nq20, nq21)`. Everything
 * else, including the both-collocated shortcut, matches the direction-0
 * kernel.
 *
 * @tparam END_PTS_COLLOCATED2  Direction-2 volume rule contains the domain
 *                            endpoints; forwarded to stage one.
 * @tparam simd_type          SIMD vector type, one element per lane.
 *
 * @param   fac             Position of the first face within direction
 *                          2. Prism, pyramid and tetrahedron collapse
 *                          this direction to a single face and are
 *                          called with (0, 1).
 * @param   nfac            One past the position of the last face.
 * @param   tstride         Row stride of the @c ntbasis table, forwarded
 *                          to stage one: the number of traces the
 *                          direction has.
 * @param   nq0,nq1,nq2     Volume quadrature points per direction; the
 *                          entry point passes its nm0, nm1 and nm2.
 * @param   ntbasis2        eInterp table \f$h_k(\pm 1)\f$ of direction
 *                          2.
 * @param   nq20,nq21       Trace quadrature points of a direction-2
 *                          normal face, per tangential direction.
 * @param   tbasis20        eInterp table of tangential direction 0 (volume
 *                          direction 0).
 * @param   tbasis21        eInterp table of tangential direction 1 (volume
 *                          direction 1).
 * @param   wsp1            Scratch for the extracted planes,
 *                          `nq0 * nq1` per face.
 * @param   wsp2            Scratch for the tangential-0 interpolation pass.
 * @param   in              Volume field of this element group.
 * @param   out             Trace values, `nq20 * nq21` per face.
 * @param   isCollocated20  Tangential-0 trace points are the direction-0
 *                          volume points.
 * @param   isCollocated21  Tangential-1 trace points are the direction-1
 *                          volume points.
 */
template <bool END_PTS_COLLOCATED2, typename simd_type>
NEK_FORCE_INLINE static void PhysExtractFaceN2Kernel(
    const unsigned fac, const unsigned nfac, const unsigned tstride,
    const unsigned nq0, const unsigned nq1, const unsigned nq2,
    const typename simd_type::scalarType *ntbasis2, const unsigned nq20,
    const unsigned nq21, const typename simd_type::scalarType *tbasis20,
    const typename simd_type::scalarType *tbasis21, simd_type *wsp1,
    simd_type *wsp2, const simd_type *in, simd_type *out,
    const bool isCollocated20, const bool isCollocated21)
{
    if (isCollocated20 && isCollocated21) // can just extract points directly
    {
        PhysExtractEndFacesN2Kernel<END_PTS_COLLOCATED2>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis2, in, out);
    }
    else
    {
        PhysExtractEndFacesN2Kernel<END_PTS_COLLOCATED2>(
            fac, nfac, tstride, nq0, nq1, nq2, ntbasis2, in, wsp1);

        PhysInterpFaceKernel(nfac - fac, nq0, nq1, nq20, nq21, tbasis20,
                             tbasis21, wsp2, wsp1, out, isCollocated20,
                             isCollocated21);
    }
}

/**
 * @brief Three-dimensional entry point for a single named face: turn the
 * face id into a normal direction and a position, then call the glue.
 *
 * The mapping is the face-to-(direction, position) table of the shape.
 * For the hexahedron, prism and pyramid it is `4 -> (0, 0)`,
 * `2 -> (0, 1)`, `1 -> (1, 0)`, `3 -> (1, 1)`, `0 -> (2, 0)`, and for
 * the hexahedron alone `5 -> (2, 1)`; the prism and pyramid collapse
 * direction 2 and stop at face 4. The tetrahedron collapses directions 1
 * and 2 and keeps `3 -> (0, 0)`, `2 -> (0, 1)`, `1 -> (1, 0)`,
 * `0 -> (2, 0)`. Any other id, and any shape outside the four, raises a
 * fatal error.
 *
 * The position becomes the (@c fac, @c nfac) pair the glue takes, so a
 * lower face is passed as (0, 1) and an upper face as (1, 2). @p out is
 * written from index zero; the block operator adds the trace offset to
 * the pointer it hands in.
 *
 * The three runtime endpoint booleans select the compile-time arms of
 * the glue, which is why each case appears twice.
 *
 * @note A nodal tetrahedron or nodal prism reaches this through the
 * block operator's single-trace entry, which names `LibUtilities::Tet`
 * and `LibUtilities::Prism` explicitly for those cases, so the right arm
 * is taken. That is not true of the bulk entry point; see the warning on
 * PhysTraceExtractKernelLauncher.
 *
 * @note Every (@c fac, @c nfac) pair here reaches stage one as its
 * @c ntbasis row stride. A bound of one is correct only where the shape
 * really has a single trace in that direction, which makes the lower
 * face of every two-trace direction wrong on the general arm: the
 * direction-0 lower face of all four shapes, the direction-1 lower face
 * of the hexahedron, prism and pyramid, and the direction-2 lower face
 * of the hexahedron. See the notes on the extraction kernels.
 *
 * @tparam SHAPE_TYPE Tet, Prism, Pyr or Hex; anything else is a fatal
 *                    error.
 * @tparam simd_type  SIMD vector type, one element per lane.
 *
 * @param   face                Face id within the shape.
 * @param   nm0,nm1,nm2         Volume quadrature points per direction.
 * @param   nq00,nq01           Trace quadrature points of a direction-0
 *                              normal face, per tangential direction.
 * @param   nq10,nq11           The same for a direction-1 normal face.
 * @param   nq20,nq21           The same for a direction-2 normal face.
 * @param   ntbasis0            eInterp table \f$h_i(\pm 1)\f$ of
 *                              direction 0.
 * @param   ntbasis1            eInterp table \f$h_j(\pm 1)\f$ of
 *                              direction 1.
 * @param   ntbasis2            eInterp table \f$h_k(\pm 1)\f$ of
 *                              direction 2.
 * @param   tbasis00,tbasis01   eInterp tables of the two tangential
 *                              directions of a direction-0 normal face.
 * @param   tbasis10,tbasis11   The same for a direction-1 normal face.
 * @param   tbasis20,tbasis21   The same for a direction-2 normal face.
 * @param   sum_irq             Scratch for the extracted plane.
 * @param   sum_jir             Scratch for the tangential-0 interpolation
 *                              pass.
 * @param   in                  Volume field of this element group.
 * @param   out                 Trace values of this face, from index
 *                              zero.
 * @param   isCollocated00      Tangential flags of the direction-0 normal
 *                              faces.
 * @param   isCollocated01      See @p isCollocated00.
 * @param   isCollocated10      Tangential flags of the direction-1 normal
 *                              faces.
 * @param   isCollocated11      See @p isCollocated10.
 * @param   isCollocated20      Tangential flags of the direction-2 normal
 *                              faces.
 * @param   isCollocated21      See @p isCollocated20.
 * @param   endPtsCollocated0   Direction-0 volume rule contains the
 *                              domain endpoints.
 * @param   endPtsCollocated1   The same for direction 1.
 * @param   endPtsCollocated2   The same for direction 2.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename simd_type>
NEK_FORCE_INLINE static void PhysTraceFaceExtractKernel(
    const unsigned face, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const unsigned nq00, const unsigned nq01,
    const unsigned nq10, const unsigned nq11, const unsigned nq20,
    const unsigned nq21, const typename simd_type::scalarType *ntbasis0,
    const typename simd_type::scalarType *ntbasis1,
    const typename simd_type::scalarType *ntbasis2,
    const typename simd_type::scalarType *tbasis00,
    const typename simd_type::scalarType *tbasis01,
    const typename simd_type::scalarType *tbasis10,
    const typename simd_type::scalarType *tbasis11,
    const typename simd_type::scalarType *tbasis20,
    const typename simd_type::scalarType *tbasis21, simd_type *sum_irq,
    simd_type *sum_jir, const simd_type *in, simd_type *out,
    const bool isCollocated00 = false, const bool isCollocated01 = false,
    const bool isCollocated10 = false, const bool isCollocated11 = false,
    const bool isCollocated20 = false, const bool isCollocated21 = false,
    const bool endPtsCollocated0 = false, const bool endPtsCollocated1 = false,
    const bool endPtsCollocated2 = false)
{
    // One ntbasis row per trace of the direction. A call covering part of
    // a direction still indexes at that stride, so it cannot be taken
    // from the loop bound.
    constexpr unsigned tstride0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned tstride1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned tstride2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];

    if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        switch (face)
        {
            case 0:
                if (endPtsCollocated2)
                {
                    PhysExtractFaceN2Kernel<true>(
                        0, 1, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21,
                        tbasis20, tbasis21, sum_irq, sum_jir, in, out,
                        isCollocated20, isCollocated21);
                }
                else
                {
                    PhysExtractFaceN2Kernel<false>(
                        0, 1, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21,
                        tbasis20, tbasis21, sum_irq, sum_jir, in, out,
                        isCollocated20, isCollocated21);
                }
                break;
            case 1:
                if (endPtsCollocated1)
                {
                    PhysExtractFaceN1Kernel<true>(
                        0, 1, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                else
                {
                    PhysExtractFaceN1Kernel<false>(
                        0, 1, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                break;
            case 2:
                if (endPtsCollocated0)
                {
                    PhysExtractFaceN0Kernel<true>(
                        1, 2, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                else
                {
                    PhysExtractFaceN0Kernel<false>(
                        1, 2, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                break;
            case 3:
                if (endPtsCollocated0)
                {
                    PhysExtractFaceN0Kernel<true>(
                        0, 1, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                else
                {
                    PhysExtractFaceN0Kernel<false>(
                        0, 1, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Face not valid");
                break;
        }
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::Pyr)
    {
        switch (face)
        {
            case 0:
                if (endPtsCollocated2)
                {
                    PhysExtractFaceN2Kernel<true>(
                        0, 1, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21,
                        tbasis20, tbasis21, sum_irq, sum_jir, in, out,
                        isCollocated20, isCollocated21);
                }
                else
                {
                    PhysExtractFaceN2Kernel<false>(
                        0, 1, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21,
                        tbasis20, tbasis21, sum_irq, sum_jir, in, out,
                        isCollocated20, isCollocated21);
                }
                break;
            case 1:
                if (endPtsCollocated1)
                {
                    PhysExtractFaceN1Kernel<true>(
                        0, 1, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                else
                {
                    PhysExtractFaceN1Kernel<false>(
                        0, 1, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                break;
            case 2:
                if (endPtsCollocated0)
                {
                    PhysExtractFaceN0Kernel<true>(
                        1, 2, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                else
                {
                    PhysExtractFaceN0Kernel<false>(
                        1, 2, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                break;
            case 3:
                if (endPtsCollocated1)
                {
                    PhysExtractFaceN1Kernel<true>(
                        1, 2, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                else
                {
                    PhysExtractFaceN1Kernel<false>(
                        1, 2, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                break;
            case 4:
                if (endPtsCollocated0)
                {
                    PhysExtractFaceN0Kernel<true>(
                        0, 1, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                else
                {
                    PhysExtractFaceN0Kernel<false>(
                        0, 1, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Face not valid");
                break;
        }
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        switch (face)
        {
            case 0:
                if (endPtsCollocated2)
                {
                    PhysExtractFaceN2Kernel<true>(
                        0, 1, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21,
                        tbasis20, tbasis21, sum_irq, sum_jir, in, out,
                        isCollocated20, isCollocated21);
                }
                else
                {
                    PhysExtractFaceN2Kernel<false>(
                        0, 1, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21,
                        tbasis20, tbasis21, sum_irq, sum_jir, in, out,
                        isCollocated20, isCollocated21);
                }
                break;
            case 1:
                if (endPtsCollocated1)
                {
                    PhysExtractFaceN1Kernel<true>(
                        0, 1, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                else
                {
                    PhysExtractFaceN1Kernel<false>(
                        0, 1, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                break;
            case 2:
                if (endPtsCollocated0)
                {
                    PhysExtractFaceN0Kernel<true>(
                        1, 2, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                else
                {
                    PhysExtractFaceN0Kernel<false>(
                        1, 2, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                break;
            case 3:
                if (endPtsCollocated1)
                {
                    PhysExtractFaceN1Kernel<true>(
                        1, 2, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                else
                {
                    PhysExtractFaceN1Kernel<false>(
                        1, 2, tstride1, nm0, nm1, nm2, ntbasis1, nq10, nq11,
                        tbasis10, tbasis11, sum_irq, sum_jir, in, out,
                        isCollocated10, isCollocated11);
                }
                break;
            case 4:
                if (endPtsCollocated0)
                {
                    PhysExtractFaceN0Kernel<true>(
                        0, 1, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                else
                {
                    PhysExtractFaceN0Kernel<false>(
                        0, 1, tstride0, nm0, nm1, nm2, ntbasis0, nq00, nq01,
                        tbasis00, tbasis01, sum_irq, sum_jir, in, out,
                        isCollocated00, isCollocated01);
                }
                break;
            case 5:
                if (endPtsCollocated2)
                {
                    PhysExtractFaceN2Kernel<true>(
                        1, 2, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21,
                        tbasis20, tbasis21, sum_irq, sum_jir, in, out,
                        isCollocated20, isCollocated21);
                }
                else
                {
                    PhysExtractFaceN2Kernel<false>(
                        1, 2, tstride2, nm0, nm1, nm2, ntbasis2, nq20, nq21,
                        tbasis20, tbasis21, sum_irq, sum_jir, in, out,
                        isCollocated20, isCollocated21);
                }
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Face not valid");
                break;
        }
    }
    else
    {
        NEKERROR(ErrorUtil::efatal, "Shape not recognised");
    }
}

// -----------------------------------------------------------------------------
//  LAUNCHERS
// -----------------------------------------------------------------------------

/**
 * @brief One-dimensional bulk entry point: both endpoint values of
 * every element.
 *
 * The one-dimensional arm of the overload set OperatorND() calls,
 * overload resolution picking between the arms by the number of
 * arguments the index sequences expand to. Unlike the two- and
 * three-dimensional arms, which do the work themselves, this one is a
 * dispatcher: a segment has no in-trace direction, so the whole
 * operation is the two-point extraction PhysTraceExtractSegKernel
 * performs, and all this arm adds is the output stride of two and the
 * turn from a runtime @p endPtsCollocated0 into that kernel's
 * END_PTS_COLLOCATED template argument. APPEND is always false here:
 * the bulk route never accumulates.
 *
 * @tparam SHAPE_TYPE   Seg; unread, the path being the same for every
 *                      one-dimensional expansion.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter1D,
          typename simd_type>
NEK_FORCE_INLINE static void PhysTraceExtractKernelLauncher(
    const TTraceSizeParameter1D sizeParam1D,
    const typename simd_type::scalarType *ntbasis0, const simd_type *in,
    simd_type *out, const bool endPtsCollocated0)
{
    static_assert(IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter1D or "
                  "TemplatedTraceSizeParameter1D.");

    // A segment carries no tangential collocation flag; the constructor
    // records a single false one that the leaf marks unused anyway.
    if (endPtsCollocated0)
    {
        PhysTraceExtractSegKernel<false, true>(sizeParam1D.nm0(), 2, ntbasis0,
                                               in, out, false);
    }
    else
    {
        PhysTraceExtractSegKernel<false, false>(sizeParam1D.nm0(), 2, ntbasis0,
                                                in, out, false);
    }
}

/**
 * @brief Two-dimensional bulk entry point: extract every edge of the
 * element in packed order.
 *
 * The direction-0 pair goes first, as the pair call (0, 2) writing
 * `2 * nq00` values from index zero, then the direction-1 traces at
 * `out + tstride0 * nq00`: the pair (0, 2) for a quadrilateral and the single
 * lower edge (0, 1) for a triangle, whose direction 1 is collapsed. The
 * scratch @p wsp is reused by the two calls, which run one after the
 * other.
 *
 * On the collocated path the extraction writes @p nm1 and @p nm0 values
 * per edge rather than @p nq00 and @p nq10, which is consistent because
 * collocation is exactly the statement that those counts agree.
 *
 * @tparam SHAPE_TYPE Tri for the triangle arm, any other two-dimensional
 *                    shape for the quadrilateral arm.
 * @tparam simd_type  SIMD vector type, one element per lane.
 *
 * @param   sizeParam2D         Volume and trace point counts of the
 *                              shape, non-templated where the
 *                              generated switch found no compiled
 *                              combination and templated where it
 *                              did.
 * @param   ntbasis0,ntbasis1   eInterp tables \f$h(\pm 1)\f$ per
 *                              direction.
 * @param   tbasis00,tbasis10   eInterp tables onto the trace points of a
 *                              direction-0 and a direction-1 normal
 *                              edge.
 * @param   wsp                 Scratch for the extracted values, shared
 *                              by both direction calls.
 * @param   in                  Volume field of this element group.
 * @param   out                 Packed trace values of the element.
 * @param   isCollocated00      Direction-0 normal edge's trace points
 *                              are the direction-1 volume points.
 * @param   isCollocated10      Direction-1 normal edge's trace points
 *                              are the direction-0 volume points.
 * @param   endPtsCollocated0   Direction-0 volume rule contains the
 *                              domain endpoints.
 * @param   endPtsCollocated1   Direction-1 volume rule contains the
 *                              domain endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter2D,
          typename simd_type>
NEK_FORCE_INLINE static void PhysTraceExtractKernelLauncher(
    const TTraceSizeParameter2D sizeParam2D,
    const typename simd_type::scalarType *ntbasis0,
    const typename simd_type::scalarType *ntbasis1,
    const typename simd_type::scalarType *tbasis00,
    const typename simd_type::scalarType *tbasis10, simd_type *wsp,
    const simd_type *in, simd_type *out, const bool isCollocated00,
    const bool isCollocated10, const bool endPtsCollocated0,
    const bool endPtsCollocated1)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    // Shape size.
    const unsigned nm0  = sizeParam2D.nm0();
    const unsigned nm1  = sizeParam2D.nm1();
    const unsigned nq00 = sizeParam2D.nq00();
    const unsigned nq10 = sizeParam2D.nq10();

    // One ntbasis row per trace of the direction. A call covering part of
    // a direction still indexes at that stride, so it cannot be taken
    // from the loop bound.
    constexpr unsigned tstride0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned tstride1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

    // edges with dir0 normals (1 & 3 in quads)
    if (endPtsCollocated0)
    {
        PhysExtractEdgeN0Kernel<true>(0, tstride0, tstride0, nm0, nm1, ntbasis0,
                                      nq00, tbasis00, wsp, in, out,
                                      isCollocated00);
    }
    else
    {
        PhysExtractEdgeN0Kernel<false>(0, tstride0, tstride0, nm0, nm1,
                                       ntbasis0, nq00, tbasis00, wsp, in, out,
                                       isCollocated00);
    }

    // edges with dir 1 normals (edge 0 in tris, edges 0 & 2 in quads)
    if (endPtsCollocated1)
    {
        PhysExtractEdgeN1Kernel<true>(0, tstride1, tstride1, nm0, nm1, ntbasis1,
                                      nq10, tbasis10, wsp, in,
                                      out + tstride0 * nq00, isCollocated10);
    }
    else
    {
        PhysExtractEdgeN1Kernel<false>(0, tstride1, tstride1, nm0, nm1,
                                       ntbasis1, nq10, tbasis10, wsp, in,
                                       out + tstride0 * nq00, isCollocated10);
    }
}

/**
 * @brief Three-dimensional bulk entry point: extract every face of the
 * element in packed order.
 *
 * The direction-0 pair goes first, as the pair call (0, 2) writing
 * `2 * nq00 * nq01` values from index zero. The running @c offset then
 * advances by exactly what each direction wrote, so the direction-1
 * block follows -- the pair (0, 2) for a hexahedron, prism and pyramid,
 * the single face (0, 1) for a tetrahedron, whose direction 1 is
 * collapsed -- and then the direction-2 block, the pair (0, 2) for a
 * hexahedron and the single face (0, 1) for every other shape. That
 * reproduces the faces-per-direction counts 2/2/2, 2/2/1 and 2/1/1 of
 * hexahedron, prism or pyramid, and tetrahedron.
 *
 * The two workspaces are reused by the three direction calls, which run
 * one after the other.
 *
 * @tparam SHAPE_TYPE Tet for the single direction-1 face, Hex for the
 *                    direction-2 pair, any other three-dimensional shape
 *                    for the prism and pyramid combination.
 * @tparam simd_type  SIMD vector type, one element per lane.
 *
 * @param   sizeParam3D         Volume and trace point counts of the
 *                              shape, non-templated where the
 *                              generated switch found no compiled
 *                              combination and templated where it
 *                              did.
 * @param   ntbasis0            eInterp table \f$h_i(\pm 1)\f$ of
 *                              direction 0.
 * @param   ntbasis1            eInterp table \f$h_j(\pm 1)\f$ of
 *                              direction 1.
 * @param   ntbasis2            eInterp table \f$h_k(\pm 1)\f$ of
 *                              direction 2.
 * @param   tbasis00,tbasis01   eInterp tables of the two tangential
 *                              directions of a direction-0 normal face.
 * @param   tbasis10,tbasis11   The same for a direction-1 normal face.
 * @param   tbasis20,tbasis21   The same for a direction-2 normal face.
 * @param   sum_irq             Scratch for the extracted planes, shared
 *                              by all three direction calls.
 * @param   sum_jir             Scratch for the tangential-0 interpolation
 *                              pass, likewise shared.
 * @param   in                  Volume field of this element group.
 * @param   out                 Packed trace values of the element.
 * @param   isCollocated00      Tangential flags of the direction-0 normal
 *                              faces.
 * @param   isCollocated01      See @p isCollocated00.
 * @param   isCollocated10      Tangential flags of the direction-1 normal
 *                              faces.
 * @param   isCollocated11      See @p isCollocated10.
 * @param   isCollocated20      Tangential flags of the direction-2 normal
 *                              faces.
 * @param   isCollocated21      See @p isCollocated20.
 * @param   endPtsCollocated0   Direction-0 volume rule contains the
 *                              domain endpoints.
 * @param   endPtsCollocated1   The same for direction 1.
 * @param   endPtsCollocated2   The same for direction 2.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter3D,
          typename simd_type>
NEK_FORCE_INLINE static void PhysTraceExtractKernelLauncher(
    const TTraceSizeParameter3D sizeParam3D,
    const typename simd_type::scalarType *ntbasis0,
    const typename simd_type::scalarType *ntbasis1,
    const typename simd_type::scalarType *ntbasis2,
    const typename simd_type::scalarType *tbasis00,
    const typename simd_type::scalarType *tbasis01,
    const typename simd_type::scalarType *tbasis10,
    const typename simd_type::scalarType *tbasis11,
    const typename simd_type::scalarType *tbasis20,
    const typename simd_type::scalarType *tbasis21, simd_type *sum_irq,
    simd_type *sum_jir, const simd_type *in, simd_type *out,
    const bool isCollocated00, const bool isCollocated01,
    const bool isCollocated10, const bool isCollocated11,
    const bool isCollocated20, const bool isCollocated21,
    const bool endPtsCollocated0, const bool endPtsCollocated1,
    const bool endPtsCollocated2)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    // Shape size.
    const unsigned nm0  = sizeParam3D.nm0();
    const unsigned nm1  = sizeParam3D.nm1();
    const unsigned nm2  = sizeParam3D.nm2();
    const unsigned nq00 = sizeParam3D.nq00();
    const unsigned nq01 = sizeParam3D.nq01();
    const unsigned nq10 = sizeParam3D.nq10();
    const unsigned nq11 = sizeParam3D.nq11();
    const unsigned nq20 = sizeParam3D.nq20();
    const unsigned nq21 = sizeParam3D.nq21();

    // One ntbasis row per trace of the direction. A call covering part of
    // a direction still indexes at that stride, so it cannot be taken
    // from the loop bound.
    constexpr unsigned tstride0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned tstride1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned tstride2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];

    // faces with dir0 normals (4 & 2 in hex)
    if (endPtsCollocated0)
    {
        PhysExtractFaceN0Kernel<true>(0, tstride0, tstride0, nm0, nm1, nm2,
                                      ntbasis0, nq00, nq01, tbasis00, tbasis01,
                                      sum_irq, sum_jir, in, out, isCollocated00,
                                      isCollocated01);
    }
    else
    {
        PhysExtractFaceN0Kernel<false>(0, tstride0, tstride0, nm0, nm1, nm2,
                                       ntbasis0, nq00, nq01, tbasis00, tbasis01,
                                       sum_irq, sum_jir, in, out,
                                       isCollocated00, isCollocated01);
    }

    unsigned offset = tstride0 * nq00 * nq01;

    // faces with dir1 normals (1 & 3 in hex, 2 in tet)
    if (endPtsCollocated1)
    {
        PhysExtractFaceN1Kernel<true>(0, tstride1, tstride1, nm0, nm1, nm2,
                                      ntbasis1, nq10, nq11, tbasis10, tbasis11,
                                      sum_irq, sum_jir, in, out + offset,
                                      isCollocated10, isCollocated11);
    }
    else
    {
        PhysExtractFaceN1Kernel<false>(0, tstride1, tstride1, nm0, nm1, nm2,
                                       ntbasis1, nq10, nq11, tbasis10, tbasis11,
                                       sum_irq, sum_jir, in, out + offset,
                                       isCollocated10, isCollocated11);
    }
    offset += tstride1 * nq10 * nq11;

    // faces with dir2 normals (0 & 5 in hex, 0 in tet)
    if (endPtsCollocated2)
    {
        PhysExtractFaceN2Kernel<true>(0, tstride2, tstride2, nm0, nm1, nm2,
                                      ntbasis2, nq20, nq21, tbasis20, tbasis21,
                                      sum_irq, sum_jir, in, out + offset,
                                      isCollocated20, isCollocated21);
    }
    else
    {
        PhysExtractFaceN2Kernel<false>(0, tstride2, tstride2, nm0, nm1, nm2,
                                       ntbasis2, nq20, nq21, tbasis20, tbasis21,
                                       sum_irq, sum_jir, in, out + offset,
                                       isCollocated20, isCollocated21);
    }
}

} // namespace Nektar::MultiRegions::detail
