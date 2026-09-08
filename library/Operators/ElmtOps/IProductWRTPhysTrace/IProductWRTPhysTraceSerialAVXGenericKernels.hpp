///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysTraceSerialAVXGenericKernels.hpp
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
 * @file IProductWRTPhysTraceSerialAVXGenericKernels.hpp
 * @brief Serial and AVX kernels of the trace inner product: one
 * element per SIMD lane.
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
 * h_{p_b}(\xi_b)\f$, which splits every trace integral into two stages: a
 * sum-factorised inner product over the in-trace directions, then a
 * rank-one lift along the normal direction. Every kernel in this file is
 * one of those two stages, or glue routing a trace to them.
 *
 * @section ipwrttrace_avx_naming Naming
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
 * collapsed direction (direction 1 of a tetrahedron and of a triangle,
 * direction 2 of prism, pyramid and tetrahedron). @c tw are the trace
 * quadrature weights and @c tjac the trace Jacobian.
 *
 * @section ipwrttrace_avx_layout Layout contracts
 *
 * - SIMD interleaving. The block operator hands the kernels one element
 *   group at a time, so one @c simd_type entry carries the same quantity
 *   for @c simd_type::width elements at once: entry @em n of the group is
 *   `buf[n]` and lane @em l of it belongs to element @em l. @p in,
 *   @p out, the workspaces and @p tjac are interleaved that way, while
 *   @c nbasis, @c tbasis and @c tw are tables and hold the same value in
 *   every lane. This is the SIMD analogue of the warp interleaving of the
 *   device kernels, with the element-group loop of the block operator in
 *   place of the grid-stride loop. For the Serial execution space
 *   @c simd_type is `tinysimd::scalarT`, a width of one.
 * - Trace packing. The packed input, and the deformed Jacobian, hold the
 *   traces in pair order: the \f$N_0\f$ pair first, then the \f$N_1\f$
 *   pair or single, then \f$N_2\f$, trace-major within a pair.
 * - Regular Jacobian. One slot per trace in that same order, which gives
 *   the face-id to slot tables hex `{4,2,1,3,0,5}`, prism and pyramid
 *   `{4,2,1,3,0}` and tetrahedron `{3,2,1,0}`. Those tables belong to the
 *   per-trace route and live in the block operator, which offsets @p tjac
 *   before calling; the bulk entry points here walk the same slots in
 *   packed order themselves. Either way the leaf kernels see only
 *   `tjac[e - edg]` or `tjac[f - fac]`, the slot relative to the first
 *   trace of the call.
 * - Face to (direction, position). Hex, prism and pyramid `4->(0,0)`,
 *   `2->(0,1)`, `1->(1,0)`, `3->(1,1)`, `0->(2,0)`, and for the
 *   hexahedron alone `5->(2,1)`; tetrahedron `3->(0,0)`, `2->(0,1)`,
 *   `1->(1,0)`, `0->(2,0)`. In two dimensions the quadrilateral has
 *   `3->(0,0)`, `1->(0,1)`, `0->(1,0)`, `2->(1,1)` and the triangle
 *   `2->(0,0)`, `1->(0,1)`, `0->(1,0)`.
 * - The \f$N_a\f$ trace block spans the two volume quadrature counts
 *   other than \f$Q_a\f$, so the glue level routes purely by direction:
 *   \f$N_0\f$ uses `(nm1,nm2)`, \f$N_1\f$ uses `(nm0,nm2)` and
 *   \f$N_2\f$ uses `(nm0,nm1)`.
 *
 * @section ipwrttrace_avx_fast Fast paths
 *
 * Two flags select them, both decided at setup. @c endPtsCollocated says the
 * volume rule of the normal direction contains the domain endpoints, so
 * \f$h_p(\pm 1)\f$ is a Kronecker delta and the lift degenerates to
 * writing a single boundary plane. It reaches the leaf kernels as a
 * template parameter, which is why every call site that depends on it is
 * written out twice. @c isCollocated says the trace quadrature points of a
 * tangential direction coincide with the volume points of that direction, so
 * the interpolation table is the identity and the contraction degenerates
 * to a pointwise multiply by the trace weights and the Jacobian. It stays
 * a runtime boolean all the way down to the leaf.
 *
 * @section ipwrttrace_avx_layer Layering
 *
 * The file is banner-separated into four sections: the one-, two- and
 * three-dimensional kernels, each holding everything the launcher of
 * that dimension reaches, then the launchers themselves. Within a
 * dimension the order is bottom-up, leaves before the glue that calls
 * them and the entry points last. Across the sections the two- and
 * three-dimensional paths are two layers deep below their entry
 * points:
 * - entry points. The two- and three-dimensional arms of
 *   `IProductWRTPhysTraceKernelLauncher` apply every trace of one
 *   element group, walking the packed traces and advancing the input
 *   and Jacobian pointers as they go; the one-dimensional arm only
 *   dispatches onto `IProductWRTPhysTrace1DKernel`, the segment case,
 *   which has no layers below it.
 *   `IProductWRTPhysTrace{Edge,Face}Kernel` apply one named trace, whose
 *   data the block operator has already offset to;
 * - glue (`IPWRTPhysTraceEdgeN{0,1}Kernel`,
 *   `IPWRTPhysTraceFaceN{0,1,2}Kernel`), which pick the direction's
 *   tables and counts and call the two leaf stages in order;
 * - leaves, `IPWRTPhys{Edge,Face}Kernel` for the in-trace inner product
 *   and `AddEdgeN{0,1}ToVolKernel` or `AddFaceN{0,1,2}ToVolKernel` for
 *   the lift.
 *
 * The file declares no includes of its own and relies on its includer for
 * @c std::memset, @c NEKERROR, @c ASSERTL1, @c NEK_FORCE_INLINE and the
 * @c LibUtilities shape enumeration.
 *
 * @section ipwrttrace_avx_state State of the paths
 *
 * Not everything here is correct:
 * - The result of a nodal expansion is left in the modal space. The
 *   kernels integrate against the volume basis and stop there, while a
 *   nodal expansion's coefficients are the nodal ones that
 *   StdNodalTriExp::v_IProductWRTBase reaches by following the same
 *   integral with NodalToModalTranspose. Nothing here applies that
 *   transform, so every coefficient a @c NodalTri, @c NodalTet or
 *   @c NodalPrism produces is wrong. The forward PhysTraceExtract is
 *   unaffected, mapping physical values to physical values.
 * - `AddEdgeN{0,1}ToVolKernel` and `AddFaceN{0,1,2}ToVolKernel` take
 *   their loop bound as the @c nbasis row stride. That bound differs
 *   from the direction's trace count only for a window of [0, 1) on a
 *   two-trace direction: the bulk kernels always span the direction,
 *   and an upper-trace window of [1, 2) still passes a bound of two.
 *   The per-trace entry points do call with a window of one for the
 *   lower trace of a two-trace direction, so the general (non-@c
 *   endPtsCollocated) arm then reads the table with a stride of one. The
 *   device port gave its face kernels an explicit @c tstride parameter
 *   for exactly this reason.
 * - `AddFaceN1ToVolKernel` advances its input by `nm0 * (nm1 - 1)` per
 *   face in the general arm, where the trace-mode slabs it reads are
 *   `nm0 * nm2` apart; see its note.
 * - There is no per-trace segment path: the block operator raises a fatal
 *   error for a segment rather than calling in here.
 *
 * @see IProductWRTPhysTraceSerialAVXGeneric.hpp for the block operator
 * that owns the element-group loop, the workspace sizing and the shape
 * dispatch, IProductWRTPhysTraceDeviceGenericKernels.hpp for the warp
 * interleaved counterparts of the same decomposition, and the
 * PhysTraceExtract kernels for the adjoint operation, which applies the
 * same tables untransposed.
 */

#pragma once

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

// -----------------------------------------------------------------------------
//  ONE-DIMENSIONAL KERNELS
// -----------------------------------------------------------------------------

/**
 * @brief One group of segments: lift the two endpoint values onto the
 * volume grid.
 *
 * The traces of a segment are its two vertices, so there is no surface
 * measure, no trace quadrature and no Jacobian to apply: the whole
 * operator is the rank-one lift
 * \f$\Lambda_p \mathrel{+}= h_p(-1)\,\hat f_0 + h_p(+1)\,\hat f_1\f$.
 * Under @p END_PTS_COLLOCATED the volume rule contains both domain
 * endpoints, \f$h_p(\pm 1)\f$ is a Kronecker delta and the two values
 * drop straight into the first and last volume point. Otherwise the
 * interpolation table is applied in full; a segment has two traces in
 * direction 0, which is the stride of two into @p basisn0.
 *
 * Both arms accumulate onto @p out, so it is the zeroing under
 * `APPEND == false` that turns the non-appending instantiation into a
 * plain write.
 *
 * @tparam END_PTS_COLLOCATED   Volume rule contains the domain endpoints.
 * @tparam APPEND             Accumulate onto @p out instead of zeroing it
 *                            first.
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   nm0       Volume quadrature points of the segment.
 * @param   basisn0   eInterp table \f$h_p(\pm 1)\f$, indexed
 *                    `basisn0[2 * p + f]`.
 * @param   in        Packed trace values, the two vertices.
 * @param   out       Volume-shaped result of the element group.
 */
template <bool END_PTS_COLLOCATED, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysTrace1DKernel(
    const unsigned nm0, [[maybe_unused]] const simd_type *basisn0,
    const simd_type *in, simd_type *out)
{
    if constexpr (!APPEND)
    {
        // zero field
        std::memset((void *)out, 0, nm0 * sizeof(simd_type));
    }

    if constexpr (END_PTS_COLLOCATED)
    {
        out[0] += in[0];
        out[nm0 - 1] += in[1];
    }
    else
    {
        for (unsigned p = 0; p < nm0; ++p)
        {
            simd_type sum = 0.0;
            sum.fma(basisn0[2 * p], in[0]);
            sum.fma(basisn0[2 * p + 1], in[1]);

            out[p] += sum;
        }
    }
}

// -----------------------------------------------------------------------------
//  TWO-DIMENSIONAL KERNELS
// -----------------------------------------------------------------------------

/**
 * @brief Leaf kernel, stage one in two dimensions: integrate an edge, or
 * a pair of edges, against the hat functions of the in-edge direction.
 *
 * For every edge in [@p edg, @p nedge) and every volume point @em q of
 * the in-edge direction this forms
 * \f$F_q = \sum_i w_i\, h_q(\xi^{tr}_i)\, J\, \hat f_i\f$ over the
 * @p tnq trace quadrature points. Under @p isCollocated the trace points
 * are the volume points, the table is the identity and the contraction
 * degenerates to a pointwise multiply by the weight and the Jacobian;
 * the debug assertion states the precondition that makes that legal.
 *
 * @p out is the caller's trace-mode workspace, not the volume field: it
 * receives @p nm entries per edge, edge-major, starting at zero, and both
 * arms write it rather than accumulate onto it. For regular geometry the
 * Jacobian slot is `e - edg`, that is relative to the first edge of this
 * call, the caller having already offset @p tjac to the right trace.
 *
 * @tparam DEFORMED    Trace Jacobian varies point by point.
 * @tparam simd_type   SIMD vector type carrying one element per lane.
 *
 * @param   edg         Position of the first edge within its direction.
 * @param   nedge       One past the position of the last edge.
 * @param   nm          Volume quadrature points of the in-edge
 *                      direction, hence entries produced per edge.
 * @param   tnq         Trace quadrature points of the edge.
 * @param   tbasis      eInterp table \f$h_q(\xi^{tr}_i)\f$, indexed
 *                      `tbasis[q * tnq + i]`.
 * @param   tw          Trace quadrature weights.
 * @param   tjac        Trace Jacobian of this element group.
 * @param   in          Packed trace values of the edges covered.
 * @param   out         Trace-mode workspace.
 * @param   isCollocated  Trace points coincide with the volume points.
 */
template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysEdgeKernel(
    const unsigned edg, const unsigned nedge, const unsigned nm,
    const unsigned tnq, const simd_type *tbasis, const simd_type *tw,
    const simd_type *tjac, const simd_type *in, simd_type *out,
    const bool isCollocated)
{
    if (isCollocated)
    {
        ASSERTL1(nm == tnq, "Basis is not collocated");

        unsigned cnt = 0;
        for (unsigned e = edg; e < nedge; ++e)
        {
            for (unsigned q = 0; q < nm; ++q, ++cnt)
            {
                // integrate mode over edge
                simd_type jac_val;

                if constexpr (DEFORMED)
                {
                    jac_val = tjac[cnt];
                }
                else
                {
                    jac_val = tjac[e - edg];
                }

                out[cnt] = in[cnt] * jac_val * tw[q];
            }
        }
    }
    else // integrate
    {
        unsigned cnt     = 0;
        unsigned qOffset = 0;
        for (unsigned e = edg; e < nedge; ++e)
        {
            for (unsigned q = 0; q < nm; ++q)
            {
                simd_type sum_q  = 0.0;
                unsigned bOffset = q * tnq;

                for (unsigned i = 0; i < tnq; ++i)
                {
                    simd_type jac_val;

                    if constexpr (DEFORMED)
                    {
                        jac_val = tjac[qOffset + i];
                    }
                    else
                    {
                        jac_val = tjac[e - edg];
                    }

                    simd_type prod =
                        in[qOffset + i] * tbasis[bOffset + i] * jac_val;
                    sum_q.fma(prod, tw[i]);
                }
                out[cnt++] = sum_q;
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
 * @note The general arm indexes `nbasis[i * nedge + e]`, taking the loop
 * bound @p nedge as the table's row stride. That is wrong only for a
 * window of [0, 1) on a direction that has two traces; every other call
 * here happens to pass a bound equal to the direction's trace count.
 * IProductWRTPhysTraceEdgeKernel calls this with `edg = 0, nedge = 1`
 * for the lower edge of a two-trace direction, where the stride should
 * still be two; see the file note.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   edg     Position of the first edge within direction 0.
 * @param   nedge   One past the position of the last edge; also used as
 *                  the @p nbasis row stride.
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_i(\pm 1)\f$ of direction 0.
 * @param   in      Trace-mode workspace written by IPWRTPhysEdgeKernel.
 * @param   out     Volume field of this element group, accumulated onto.
 */
template <bool END_PTS_COLLOCATED0, typename simd_type>
NEK_FORCE_INLINE static void AddEdgeN0ToVolKernel(
    const unsigned edg, const unsigned nedge, const unsigned nm0,
    const unsigned nm1, [[maybe_unused]] const simd_type *nbasis,
    const simd_type *in, simd_type *out)
{
    if constexpr (END_PTS_COLLOCATED0)
    {
        for (unsigned e = edg, cnt = 0; e < nedge; ++e)
        {
            unsigned offset = e * (nm0 - 1);
            for (unsigned q = 0; q < nm1; ++q, ++cnt)
            {
                out[nm0 * q + offset] += in[cnt];
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
                    out[nm0 * q + i].fma(in[cnt], nbasis[i * nedge + e]);
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
 * endpoint fast path the whole edge lands in the row `q = e * (nm1 - 1)`
 * and the general arm accumulates
 * \f$\Lambda_{iq} \mathrel{+}= h_q(\pm 1)\, F_i\f$.
 *
 * @note The template parameter is spelled @c END_PTS_COLLOCATED0 although
 * it is the direction-1 flag that the callers pass; only the name is
 * inherited from the direction-0 kernel. The @p nbasis row stride caveat
 * of AddEdgeN0ToVolKernel applies here unchanged.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-1 volume rule contains the
 *                            domain endpoints, despite the name.
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   edg     Position of the first edge within direction 1.
 * @param   nedge   One past the position of the last edge; also used as
 *                  the @p nbasis row stride.
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   in      Trace-mode workspace written by IPWRTPhysEdgeKernel.
 * @param   out     Volume field of this element group, accumulated onto.
 */
template <bool END_PTS_COLLOCATED0, typename simd_type>
NEK_FORCE_INLINE static void AddEdgeN1ToVolKernel(
    const unsigned edg, const unsigned nedge, const unsigned nm0,
    const unsigned nm1, [[maybe_unused]] const simd_type *nbasis,
    const simd_type *in, simd_type *out)
{
    if constexpr (END_PTS_COLLOCATED0)
    {
        for (unsigned e = edg, cnt = 0; e < nedge; ++e)
        {
            unsigned offset = e * nm0 * (nm1 - 1);
            for (unsigned q = 0; q < nm0; ++q, ++cnt)
            {
                out[offset + q] += in[cnt];
            }
        }
    }
    else // project to interior
    {
        for (unsigned e = edg, cnt = 0; e < nedge; ++e)
        {
            for (unsigned q = 0; q < nm1; ++q)
            {
                simd_type val = nbasis[q * nedge + e];
                for (unsigned i = 0; i < nm0; ++i)
                {
                    out[nm0 * q + i].fma(in[cnt + i], val);
                }
            }
            cnt += nm0;
        }
    }
}

// kernels for edge 3 and 1
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
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   edg     Position of the first edge within direction 0.
 * @param   nedge   One past the position of the last edge.
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_i(\pm 1)\f$ of direction 0.
 * @param   tnq     Trace quadrature points of the edge.
 * @param   tbasis  eInterp table to the trace points of direction 0.
 * @param   tw      Trace quadrature weights of direction 0.
 * @param   tjac    Trace Jacobian of this element group.
 * @param   wsp     Trace-mode scratch, at least `nm1` entries per edge
 *                  covered.
 * @param   in      Packed trace values of the edges covered.
 * @param   out     Volume field of this element group, accumulated onto.
 * @param   isCollocated  Trace points coincide with the volume points.
 */
template <bool END_PTS_COLLOCATED0, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysTraceEdgeN0Kernel(
    const unsigned edg, const unsigned nedge, const unsigned nm0,
    const unsigned nm1, const simd_type *nbasis, const unsigned tnq,
    const simd_type *tbasis, const simd_type *tw, const simd_type *tjac,
    simd_type *wsp, const simd_type *in, simd_type *out,
    const bool isCollocated)
{
    IPWRTPhysEdgeKernel<DEFORMED>(edg, nedge, nm1, tnq, tbasis, tw, tjac, in,
                                  wsp, isCollocated);

    AddEdgeN0ToVolKernel<END_PTS_COLLOCATED0>(edg, nedge, nm0, nm1, nbasis, wsp,
                                              out);
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
 * @tparam END_PTS_COLLOCATED0  Direction-1 volume rule contains the
 *                            domain endpoints, despite the name.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   edg     Position of the first edge within direction 1.
 * @param   nedge   One past the position of the last edge.
 * @param   nm0     Volume quadrature points in direction 0.
 * @param   nm1     Volume quadrature points in direction 1.
 * @param   nbasis  eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   tnq     Trace quadrature points of the edge.
 * @param   tbasis  eInterp table to the trace points of direction 1.
 * @param   tw      Trace quadrature weights of direction 1.
 * @param   tjac    Trace Jacobian of this element group.
 * @param   wsp     Trace-mode scratch, at least `nm0` entries per edge
 *                  covered.
 * @param   in      Packed trace values of the edges covered.
 * @param   out     Volume field of this element group, accumulated onto.
 * @param   isCollocated  Trace points coincide with the volume points.
 */
template <bool END_PTS_COLLOCATED0, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysTraceEdgeN1Kernel(
    const unsigned edg, const unsigned nedge, const unsigned nm0,
    const unsigned nm1, const simd_type *nbasis, const unsigned tnq,
    const simd_type *tbasis, const simd_type *tw, const simd_type *tjac,
    simd_type *wsp, const simd_type *in, simd_type *out,
    const bool isCollocated)
{
    IPWRTPhysEdgeKernel<DEFORMED>(edg, nedge, nm0, tnq, tbasis, tw, tjac, in,
                                  wsp, isCollocated);

    AddEdgeN1ToVolKernel<END_PTS_COLLOCATED0>(edg, nedge, nm0, nm1, nbasis, wsp,
                                              out);
}

/**
 * @brief One named edge of one two-dimensional element group: turn the
 * edge id into a direction and a position, then integrate and lift.
 *
 * The single-trace entry point, used when the solver drives one trace at
 * a time rather than the whole boundary. @p in and @p tjac already point
 * at that edge's block, so the position within the direction is expressed
 * only through the [`edg`, `nedge`) window handed to the glue kernels.
 * The edge to (direction, position) map is `3->(0,0)`, `1->(0,1)`,
 * `0->(1,0)`, `2->(1,1)` for a quadrilateral, and `2->(0,0)`, `1->(0,1)`,
 * `0->(1,0)` for a triangle, whose direction 1 carries a single trace at
 * the collapsed end. An unrecognised @p edge raises a fatal error, which
 * the device twin cannot do.
 *
 * @note @c SHAPE_TYPE here is the shape the block operator names when it
 * calls, and it names the parent shape: a @c NodalTri arrives as @c Tri
 * and takes the triangle arm correctly. That is the opposite of the bulk
 * kernel, which is instantiated on the true shape; see the file note.
 *
 * @note The block operator only ever instantiates this with
 * `APPEND == true`, so nothing is zeroed here and the caller must zero
 * the volume field before applying its first trace.
 *
 * @note For the lower edge of a two-trace direction the window is
 * [0, 1), which the lift kernels also use as the @c nbasis row stride;
 * the general (non-@c endPtsCollocated) arm then reads that table with a stride
 * of one instead of two. See the file note.
 *
 * @tparam SHAPE_TYPE  Quad, or Tri standing in for any triangle.
 * @tparam DEFORMED    Trace Jacobian varies point by point.
 * @tparam APPEND      Accumulate onto @p out instead of zeroing it first.
 * @tparam simd_type   SIMD vector type carrying one element per lane.
 *
 * @param   edge            Nektar edge id of the trace to apply.
 * @param   nm0,nm1         Volume quadrature points per direction.
 * @param   nbasis0,nbasis1 eInterp tables \f$h_p(\pm 1)\f$ per direction.
 * @param   tnq00,tnq10     Trace quadrature points of the direction-0 and
 *                          direction-1 edges.
 * @param   tbasis0,tbasis1 eInterp tables to the trace points, per
 *                          direction.
 * @param   tw00,tw10       Trace quadrature weights per direction.
 * @param   tjac            Trace Jacobian of this element group, already
 *                          offset to this edge.
 * @param   wsp             Trace-mode scratch, `max(nm0,nm1)` entries.
 * @param   in              Packed trace values, already offset to this
 *                          edge.
 * @param   out             Volume field of this element group.
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
          typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysTraceEdgeKernel(
    const unsigned edge, const unsigned nm0, const unsigned nm1,
    const simd_type *nbasis0, const simd_type *nbasis1, const unsigned tnq00,
    const unsigned tnq10, const simd_type *tbasis0, const simd_type *tbasis1,
    const simd_type *tw00, const simd_type *tw10, const simd_type *tjac,
    simd_type *wsp, const simd_type *in, simd_type *out,
    const bool isCollocated0, const bool isCollocated1,
    const bool endPtsCollocated0, const bool endPtsCollocated1)
{
    if constexpr (!APPEND)
    {
        // zero field
        std::memset((void *)out, 0, nm0 * nm1 * sizeof(simd_type));
    }

    if constexpr (SHAPE_TYPE == Nektar::LibUtilities::Tri)
    {
        switch (edge)
        {
            case 0:
                if (endPtsCollocated1)
                {
                    IPWRTPhysTraceEdgeN1Kernel<true, DEFORMED>(
                        0, 1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10, tjac,
                        wsp, in, out, isCollocated1);
                }
                else
                {
                    IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
                        0, 1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10, tjac,
                        wsp, in, out, isCollocated1);
                }
                break;
            case 1:
                if (endPtsCollocated0)
                {

                    IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                        1, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac,
                        wsp, in, out, isCollocated0);
                }
                else
                {

                    IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                        1, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac,
                        wsp, in, out, isCollocated0);
                }
                break;
            case 2:
                if (endPtsCollocated0)
                {

                    IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                        0, 1, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac,
                        wsp, in, out, isCollocated0);
                }
                else
                {

                    IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                        0, 1, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac,
                        wsp, in, out, isCollocated0);
                }
                break;
            default:
                NEKERROR(Nektar::ErrorUtil::efatal, "Unrecognised edge input");
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
                        0, 1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10, tjac,
                        wsp, in, out, isCollocated1);
                }
                else
                {
                    IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
                        0, 1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10, tjac,
                        wsp, in, out, isCollocated1);
                }
                break;
            case 1:
                if (endPtsCollocated0)
                {

                    IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                        1, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac,
                        wsp, in, out, isCollocated0);
                }
                else
                {

                    IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                        1, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac,
                        wsp, in, out, isCollocated0);
                }
                break;
            case 2:
                if (endPtsCollocated1)
                {
                    IPWRTPhysTraceEdgeN1Kernel<true, DEFORMED>(
                        1, 2, nm0, nm1, nbasis1, tnq10, tbasis1, tw10, tjac,
                        wsp, in, out, isCollocated1);
                }
                else
                {
                    IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
                        1, 2, nm0, nm1, nbasis1, tnq10, tbasis1, tw10, tjac,
                        wsp, in, out, isCollocated1);
                }
                break;
            case 3:
                if (endPtsCollocated0)
                {

                    IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                        0, 1, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac,
                        wsp, in, out, isCollocated0);
                }
                else
                {

                    IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                        0, 1, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac,
                        wsp, in, out, isCollocated0);
                }
                break;
            default:
                NEKERROR(Nektar::ErrorUtil::efatal, "Unrecognised edge input");
                break;
        }
    }
}

// -----------------------------------------------------------------------------
//  THREE-DIMENSIONAL KERNELS
// -----------------------------------------------------------------------------

// kernels for fasces 4 and 2 in hex
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
 * @note The general arm indexes `nbasis[p * nface + f]`, taking the loop
 * bound @p nface as the table's row stride; see AddEdgeN0ToVolKernel and
 * the file note for when that is not the direction's trace count.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   fac     Position of the first face within direction 0.
 * @param   nface   One past the position of the last face; also used as
 *                  the @p nbasis row stride.
 * @param   nm0,nm1,nm2   Volume quadrature points per direction.
 * @param   nbasis  eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   in      Trace-mode workspace written by IPWRTPhysFaceKernel,
 *                  `nm1 * nm2` entries per face.
 * @param   out     Volume field of this element group, accumulated onto.
 */
template <bool END_PTS_COLLOCATED0, typename simd_type>
NEK_FORCE_INLINE static void AddFaceN0ToVolKernel(
    const unsigned fac, const unsigned nface, const unsigned nm0,
    const unsigned nm1, const unsigned nm2,
    [[maybe_unused]] const simd_type *nbasis, const simd_type *in,
    simd_type *out)
{
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
                    out[outOffset + q * nm0] += in[cnt];
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
                        out[ocnt].fma(in[cnt], nbasis[p * nface + f]);
                    }
                }
            }
        }
    }
}

// kernels for fasces 1 and 3 in hex
/**
 * @brief Leaf kernel, stage two in three dimensions: lift the results of
 * a direction-1 normal face into the volume field.
 *
 * The direction-1 twin of AddFaceN0ToVolKernel. The face carries
 * `nm0 * nm2` entries laid out face-local as `r * nm0 + p`, the endpoint
 * fast path writes the plane `q = f * (nm1 - 1)`, and the general arm
 * accumulates \f$\Lambda_{pqr} \mathrel{+}= h_q(\pm 1)\, F_{pr}\f$ over
 * every @em q.
 *
 * @note The general arm starts each face's input slab at
 * `f * nm0 * (nm1 - 1)`, the same expression the endpoint arm uses for
 * its @em output plane, whereas the slabs it reads are `nm0 * nm2` apart
 * and are counted from the first face of the call. It therefore reads the
 * wrong slab whenever the call covers more than the first face at
 * position 0: the upper face of the direction-1 pair on the bulk route,
 * and face 3 on the per-trace route. Only the general arm is affected, so
 * a direction-1 rule that contains the domain endpoints hides it. The
 * device port writes `(f - fac) * nm0 * nm2` here. Open defect, stated
 * rather than documented away.
 *
 * @note The @p nbasis row stride caveat of AddFaceN0ToVolKernel applies
 * here unchanged.
 *
 * @tparam END_PTS_COLLOCATED1  Direction-1 volume rule contains the
 *                            domain endpoints.
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   fac     Position of the first face within direction 1.
 * @param   nface   One past the position of the last face; also used as
 *                  the @p nbasis row stride.
 * @param   nm0,nm1,nm2   Volume quadrature points per direction.
 * @param   nbasis  eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   in      Trace-mode workspace written by IPWRTPhysFaceKernel,
 *                  `nm0 * nm2` entries per face.
 * @param   out     Volume field of this element group, accumulated onto.
 */
template <bool END_PTS_COLLOCATED1, typename simd_type>
NEK_FORCE_INLINE static void AddFaceN1ToVolKernel(
    const unsigned fac, const unsigned nface, const unsigned nm0,
    const unsigned nm1, const unsigned nm2,
    [[maybe_unused]] const simd_type *nbasis, const simd_type *in,
    simd_type *out)
{
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
                    out[outOffset + p] += in[cnt];
                }
            }
        }
    }
    else
    {
        for (unsigned f = fac; f < nface; ++f)
        {
            // Offset of this face's data in the input. The collocated branch
            // above uses f * nm0 * (nm1 - 1), but that is an offset into the
            // *output*, picking the j = 0 or j = nm1 - 1 row. Here the face
            // contributes nm0 * nm2 values, indexed by the directions
            // tangential to dir 1.
            unsigned offset = f * nm0 * nm2;
            for (unsigned r = 0, ocnt = 0; r < nm2; ++r)
            {
                unsigned cnt = r * nm0 + offset;
                for (unsigned q = 0; q < nm1; ++q)
                {
                    simd_type val = nbasis[q * nface + f];
                    for (unsigned p = 0; p < nm0; ++p, ++ocnt)
                    {
                        out[ocnt].fma(in[cnt + p], val);
                    }
                }
            }
        }
    }
}

// kernels for fasces 0 and 5 in hex
/**
 * @brief Leaf kernel, stage two in three dimensions: lift the results of
 * a direction-2 normal face into the volume field.
 *
 * The direction-2 twin of AddFaceN0ToVolKernel. The face carries
 * `nm0 * nm1` entries laid out face-local as `q * nm0 + p`, the endpoint
 * fast path writes the plane `r = f * (nm2 - 1)`, and the general arm
 * accumulates \f$\Lambda_{pqr} \mathrel{+}= h_r(\pm 1)\, F_{pq}\f$ over
 * every @em r, advancing the input slab by `nm0 * nm1` per face.
 *
 * @note The @p nbasis row stride caveat of AddFaceN0ToVolKernel applies
 * here unchanged.
 *
 * @tparam END_PTS_COLLOCATED2  Direction-2 volume rule contains the
 *                            domain endpoints.
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   fac     Position of the first face within direction 2.
 * @param   nface   One past the position of the last face; also used as
 *                  the @p nbasis row stride.
 * @param   nm0,nm1,nm2   Volume quadrature points per direction.
 * @param   nbasis  eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   in      Trace-mode workspace written by IPWRTPhysFaceKernel,
 *                  `nm0 * nm1` entries per face.
 * @param   out     Volume field of this element group, accumulated onto.
 */
template <bool END_PTS_COLLOCATED2, typename simd_type>
NEK_FORCE_INLINE static void AddFaceN2ToVolKernel(
    const unsigned fac, const unsigned nface, const unsigned nm0,
    const unsigned nm1, const unsigned nm2,
    [[maybe_unused]] const simd_type *nbasis, const simd_type *in,
    simd_type *out)
{
    if constexpr (END_PTS_COLLOCATED2)
    {
        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            unsigned offset = f * nm0 * nm1 * (nm2 - 1);
            for (unsigned q = 0; q < nm1; ++q)
            {
                for (unsigned p = 0; p < nm0; ++p, ++cnt)
                {
                    out[offset++] += in[cnt];
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
                simd_type val = nbasis[r * nface + f];

                for (unsigned q = 0, cnt = 0; q < nm1; ++q)
                {
                    for (unsigned p = 0; p < nm0; ++p, ++ocnt, ++cnt)
                    {
                        out[ocnt].fma(in[offset + cnt], val);
                    }
                }
            }
            offset += nm0 * nm1;
        }
    }
}

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
 * @p out distinct. The debug assertions state the preconditions the
 * collocated arms rely on.
 *
 * @note @p nm0 and @p nm1 are the in-face counts the glue level supplies,
 * `(nm1,nm2)` for an \f$N_0\f$ face, `(nm0,nm2)` for \f$N_1\f$ and
 * `(nm0,nm1)` for \f$N_2\f$, not the element's direction-0 and
 * direction-1 counts.
 *
 * @tparam DEFORMED    Trace Jacobian varies point by point.
 * @tparam simd_type   SIMD vector type carrying one element per lane.
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
 * @param   tjac         Trace Jacobian of this element group.
 * @param   wsp          Scratch for the fully general arm, `nm0 * tnq1`
 *                       entries.
 * @param   in           Packed trace values of the faces covered.
 * @param   out          Trace-mode workspace.
 * @param   isCollocated0  First in-face direction is collocated.
 * @param   isCollocated1  Second in-face direction is collocated.
 */
template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysFaceKernel(
    const unsigned fac, const unsigned nface, const unsigned nm0,
    const unsigned nm1, const unsigned tnq0, const unsigned tnq1,
    const simd_type *tbasis0, const simd_type *tbasis1, const simd_type *tw0,
    const simd_type *tw1, const simd_type *tjac, simd_type *wsp,
    const simd_type *in, simd_type *out, const bool isCollocated0,
    const bool isCollocated1)
{
    if (isCollocated0 && isCollocated1) // simply multipby by weights
    {
        ASSERTL1((nm0 == tnq0) && (nm1 == tnq1), "Bases are not collocated");

        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            for (unsigned j = 0; j < tnq1; ++j)
            {
                for (unsigned i = 0; i < tnq0; ++i, ++cnt)
                {
                    // integrate mode over edge
                    simd_type jac_val;

                    if constexpr (DEFORMED)
                    {
                        jac_val = tjac[cnt];
                    }
                    else
                    {
                        jac_val = tjac[f - fac];
                    }

                    out[cnt] = in[cnt] * jac_val * tw0[i] * tw1[j];
                }
            }
        }
    }
    else if (isCollocated0) // full inner product in dir 1
    {
        ASSERTL1((nm0 == tnq0), "Basis is not collocated");

        unsigned inOffset = 0, outOffset = 0;
        for (unsigned f = fac; f < nface; ++f)
        {
            for (unsigned i = 0; i < tnq0; ++i)
            {
                for (unsigned q = 0; q < nm1; ++q)
                {
                    unsigned qoffset = q * tnq1;
                    simd_type sum_q  = 0;
                    for (unsigned j = 0; j < tnq1; ++j)
                    {
                        unsigned joffset = j * tnq0 + inOffset;

                        // integrate mode over edge
                        simd_type jac_val;

                        if constexpr (DEFORMED)
                        {
                            jac_val = tjac[joffset + i];
                        }
                        else
                        {
                            jac_val = tjac[f - fac];
                        }

                        simd_type prod =
                            in[joffset + i] * jac_val * tbasis1[qoffset + j];
                        sum_q.fma(prod, tw1[j]);
                    }
                    out[outOffset + q * nm0 + i] = sum_q * tw0[i];
                }
            }
            inOffset += tnq0 * tnq1;
            outOffset += nm0 * nm1;
        }
    }
    else if (isCollocated1) // full inner product in dir 0
    {
        ASSERTL1((nm1 == tnq1), "Basis is not collocated");

        unsigned offset = 0;
        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            for (unsigned j = 0; j < tnq1; ++j)
            {
                unsigned joffset = j * tnq0 + offset;
                for (unsigned p = 0; p < nm0; ++p, ++cnt)
                {
                    unsigned poffset = p * tnq0;
                    simd_type sum_p  = 0;
                    for (unsigned i = 0; i < tnq0; ++i)
                    {
                        // integrate mode over edge
                        simd_type jac_val;

                        if constexpr (DEFORMED)
                        {
                            jac_val = tjac[joffset + i];
                        }
                        else
                        {
                            jac_val = tjac[f - fac];
                        }
                        simd_type prod =
                            in[joffset + i] * jac_val * tbasis0[poffset + i];
                        sum_p.fma(prod, tw0[i]);
                    }
                    out[cnt] = sum_p * tw1[j];
                }
            }
            offset += tnq0 * tnq1;
        }
    }
    else // full inner product
    {
        unsigned inOffset = 0, outOffset = 0;
        for (unsigned f = fac, cnt = 0; f < nface; ++f)
        {
            for (unsigned j = 0; j < tnq1; ++j)
            {
                unsigned joffset = j * tnq0 + inOffset;

                for (unsigned p = 0; p < nm0; ++p, ++cnt)
                {
                    unsigned poffset = p * tnq0;
                    simd_type sum_p  = 0;
                    for (unsigned i = 0; i < tnq0; ++i)
                    {
                        // integrate mode over edge
                        simd_type jac_val;

                        if constexpr (DEFORMED)
                        {
                            jac_val = tjac[joffset + i];
                        }
                        else
                        {
                            jac_val = tjac[f - fac];
                        }
                        simd_type prod =
                            in[joffset + i] * jac_val * tbasis0[poffset + i];
                        sum_p.fma(prod, tw0[i]);
                    }
                    wsp[j * nm0 + p] = sum_p;
                }
            }

            for (unsigned p = 0; p < nm0; ++p)
            {
                for (unsigned q = 0; q < nm1; ++q)
                {
                    unsigned qoffset = q * tnq1;
                    simd_type sum_q  = 0;
                    for (unsigned j = 0; j < tnq1; ++j)
                    {
                        simd_type prod =
                            wsp[j * nm0 + p] * tbasis1[qoffset + j];
                        sum_q.fma(prod, tw1[j]);
                    }
                    out[outOffset + q * nm0 + p] = sum_q;
                }
            }
            inOffset += tnq0 * tnq1;
            outOffset += nm0 * nm1;
        }
    }
}

/**
 * @brief Glue for a direction-0 normal face: integrate, then lift.
 *
 * Routes the direction's tables and counts into the two leaf stages. The
 * in-face directions of an \f$N_0\f$ face are 1 and 2, so the face
 * integration is asked for `(nm1, nm2)` entries per face and the lift
 * spreads them along direction 0. Note the workspace swap: @p wsp1 is
 * handed to the integration as its scratch and @p wsp receives the
 * trace-mode result.
 *
 * @tparam END_PTS_COLLOCATED0  Direction-0 volume rule contains the
 *                            domain endpoints.
 * @tparam DEFORMED           Trace Jacobian varies point by point.
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   fac         Position of the first face within direction 0.
 * @param   nface       One past the position of the last face.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis      eInterp table \f$h_p(\pm 1)\f$ of direction 0.
 * @param   tnq0,tnq1   Trace quadrature points of the two in-face
 *                      directions.
 * @param   tbasis0,tbasis1   eInterp tables to those trace points.
 * @param   tw0,tw1     Trace quadrature weights of the in-face
 *                      directions.
 * @param   tjac        Trace Jacobian of this element group.
 * @param   wsp         Trace-mode workspace, `nm1 * nm2` entries per face
 *                      covered.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of the faces covered.
 * @param   out         Volume field of this element group, accumulated
 *                      onto.
 * @param   isCollocated0    First in-face direction is collocated.
 * @param   isCollocated1    Second in-face direction is collocated.
 */
template <bool END_PTS_COLLOCATED0, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysTraceFaceN0Kernel(
    const unsigned fac, const unsigned nface, const unsigned nm0,
    const unsigned nm1, const unsigned nm2, const simd_type *nbasis,
    const unsigned tnq0, const unsigned tnq1, const simd_type *tbasis0,
    const simd_type *tbasis1, const simd_type *tw0, const simd_type *tw1,
    const simd_type *tjac, simd_type *wsp, simd_type *wsp1, const simd_type *in,
    simd_type *out, const bool isCollocated0 = false,
    const bool isCollocated1 = false)
{
    IPWRTPhysFaceKernel<DEFORMED>(fac, nface, nm1, nm2, tnq0, tnq1, tbasis0,
                                  tbasis1, tw0, tw1, tjac, wsp1, in, wsp,
                                  isCollocated0, isCollocated1);

    AddFaceN0ToVolKernel<END_PTS_COLLOCATED0>(fac, nface, nm0, nm1, nm2, nbasis,
                                              wsp, out);
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
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   fac         Position of the first face within direction 1.
 * @param   nface       One past the position of the last face.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis      eInterp table \f$h_q(\pm 1)\f$ of direction 1.
 * @param   tnq0,tnq1   Trace quadrature points of the two in-face
 *                      directions.
 * @param   tbasis0,tbasis1   eInterp tables to those trace points.
 * @param   tw0,tw1     Trace quadrature weights of the in-face
 *                      directions.
 * @param   tjac        Trace Jacobian of this element group.
 * @param   wsp         Trace-mode workspace, `nm0 * nm2` entries per face
 *                      covered.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of the faces covered.
 * @param   out         Volume field of this element group, accumulated
 *                      onto.
 * @param   isCollocated0    First in-face direction is collocated.
 * @param   isCollocated1    Second in-face direction is collocated.
 */
template <bool END_PTS_COLLOCATED1, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysTraceFaceN1Kernel(
    const unsigned fac, const unsigned nface, const unsigned nm0,
    const unsigned nm1, const unsigned nm2, const simd_type *nbasis,
    const unsigned tnq0, const unsigned tnq1, const simd_type *tbasis0,
    const simd_type *tbasis1, const simd_type *tw0, const simd_type *tw1,
    const simd_type *tjac, simd_type *wsp, simd_type *wsp1, const simd_type *in,
    simd_type *out, const bool isCollocated0 = false,
    const bool isCollocated1 = false)
{
    IPWRTPhysFaceKernel<DEFORMED>(fac, nface, nm0, nm2, tnq0, tnq1, tbasis0,
                                  tbasis1, tw0, tw1, tjac, wsp1, in, wsp,
                                  isCollocated0, isCollocated1);

    AddFaceN1ToVolKernel<END_PTS_COLLOCATED1>(fac, nface, nm0, nm1, nm2, nbasis,
                                              wsp, out);
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
 * @tparam simd_type          SIMD vector type carrying one element per
 *                            lane.
 *
 * @param   fac         Position of the first face within direction 2.
 * @param   nface       One past the position of the last face.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis      eInterp table \f$h_r(\pm 1)\f$ of direction 2.
 * @param   tnq0,tnq1   Trace quadrature points of the two in-face
 *                      directions.
 * @param   tbasis0,tbasis1   eInterp tables to those trace points.
 * @param   tw0,tw1     Trace quadrature weights of the in-face
 *                      directions.
 * @param   tjac        Trace Jacobian of this element group.
 * @param   wsp         Trace-mode workspace, `nm0 * nm1` entries per face
 *                      covered.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of the faces covered.
 * @param   out         Volume field of this element group, accumulated
 *                      onto.
 * @param   isCollocated0    First in-face direction is collocated.
 * @param   isCollocated1    Second in-face direction is collocated.
 */
template <bool END_PTS_COLLOCATED2, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysTraceFaceN2Kernel(
    const unsigned fac, const unsigned nface, const unsigned nm0,
    const unsigned nm1, const unsigned nm2, const simd_type *nbasis,
    const unsigned tnq0, const unsigned tnq1, const simd_type *tbasis0,
    const simd_type *tbasis1, const simd_type *tw0, const simd_type *tw1,
    const simd_type *tjac, simd_type *wsp, simd_type *wsp1, const simd_type *in,
    simd_type *out, const bool isCollocated0 = false,
    const bool isCollocated1 = false)
{
    IPWRTPhysFaceKernel<DEFORMED>(fac, nface, nm0, nm1, tnq0, tnq1, tbasis0,
                                  tbasis1, tw0, tw1, tjac, wsp1, in, wsp,
                                  isCollocated0, isCollocated1);

    AddFaceN2ToVolKernel<END_PTS_COLLOCATED2>(fac, nface, nm0, nm1, nm2, nbasis,
                                              wsp, out);
}

/**
 * @brief One named face of one three-dimensional element group: turn the
 * face id into a direction and a position, then integrate and lift.
 *
 * The single-trace entry point, used when the solver drives one trace at
 * a time rather than the whole boundary. @p in and @p tjac already point
 * at that face's block, so the position within the direction is expressed
 * only through the [`fac`, `nface`) window handed to the glue kernels.
 * The face to (direction, position) map is `4->(0,0)`, `2->(0,1)`,
 * `1->(1,0)`, `3->(1,1)`, `0->(2,0)` and, for the hexahedron alone,
 * `5->(2,1)`; a tetrahedron instead has `3->(0,0)`, `2->(0,1)`,
 * `1->(1,0)`, `0->(2,0)`, which is what the @c SHAPE_TYPE test inside
 * `case 3` selects.
 *
 * @note The switch has no default arm, so a face id outside the shape's
 * range falls through with no diagnostic, leaving @p out as the @c APPEND
 * gate left it: untouched under `APPEND == true`, zeroed under
 * `APPEND == false`. The two-dimensional twin raises a fatal error in the
 * same position.
 *
 * @note @c SHAPE_TYPE here is the shape the block operator names when it
 * calls, and it names the parent shape: a @c NodalTet arrives as @c Tet
 * and a @c NodalPrism as @c Prism, so both take the right arm. That is
 * the opposite of the bulk kernel, which is instantiated on the true
 * shape; see the file note.
 *
 * @note The block operator only ever instantiates this with
 * `APPEND == true`, so nothing is zeroed here and the caller must zero
 * the volume field before applying its first trace.
 *
 * @note For the lower face of a two-trace direction the window is
 * [0, 1), which the lift kernels also use as the @c nbasis row stride;
 * the general (non-@c endPtsCollocated) arm then reads that table with a stride
 * of one instead of two. See the file note.
 *
 * @tparam SHAPE_TYPE  Hex, Prism, Pyr or Tet; the block operator maps the
 *                     nodal shapes onto these.
 * @tparam DEFORMED    Trace Jacobian varies point by point.
 * @tparam APPEND      Accumulate onto @p out instead of zeroing it first.
 * @tparam simd_type   SIMD vector type carrying one element per lane.
 *
 * @param   face        Nektar face id of the trace to apply.
 * @param   nm0,nm1,nm2 Volume quadrature points per direction.
 * @param   nbasis0,nbasis1,nbasis2   eInterp tables \f$h_p(\pm 1)\f$ per
 *                      direction.
 * @param   tnq00,tnq01 Trace quadrature points of an \f$N_0\f$ face.
 * @param   tnq10,tnq11 Trace quadrature points of an \f$N_1\f$ face.
 * @param   tnq20,tnq21 Trace quadrature points of an \f$N_2\f$ face.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points, per normal
 *                      direction and tangential direction.
 * @param   tw00,tw01,tw10,tw11,tw20,tw21   Trace quadrature weights in
 *                      the same order.
 * @param   tjac        Trace Jacobian of this element group, already
 *                      offset to this face.
 * @param   wsp         Trace-mode workspace of one face.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values, already offset to this face.
 * @param   out         Volume field of this element group.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2   Volume
 *                      rule of that direction contains the domain
 *                      endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool APPEND,
          typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysTraceFaceKernel(
    const unsigned face, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const simd_type *nbasis0, const simd_type *nbasis1,
    const simd_type *nbasis2, const unsigned tnq00, const unsigned tnq01,
    const unsigned tnq10, const unsigned tnq11, const unsigned tnq20,
    const unsigned tnq21, const simd_type *tbasis00, const simd_type *tbasis01,
    const simd_type *tbasis10, const simd_type *tbasis11,
    const simd_type *tbasis20, const simd_type *tbasis21, const simd_type *tw00,
    const simd_type *tw01, const simd_type *tw10, const simd_type *tw11,
    const simd_type *tw20, const simd_type *tw21, const simd_type *tjac,
    simd_type *wsp, simd_type *wsp1, const simd_type *in, simd_type *out,
    const bool isCollocated00, const bool isCollocated01,
    const bool isCollocated10, const bool isCollocated11,
    const bool isCollocated20, const bool isCollocated21,
    const bool endPtsCollocated0, const bool endPtsCollocated1,
    const bool endPtsCollocated2)
{
    if constexpr (!APPEND)
    {
        // zero field
        std::memset((void *)out, 0, nm0 * nm1 * nm2 * sizeof(simd_type));
    }

    switch (face)
    {
        case 0:
            if (endPtsCollocated2)
            {
                IPWRTPhysTraceFaceN2Kernel<true, DEFORMED>(
                    0, 1, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20,
                    tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21);
            }
            else
            {
                IPWRTPhysTraceFaceN2Kernel<false, DEFORMED>(
                    0, 1, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20,
                    tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21);
            }
            break;
        case 1:
            if (endPtsCollocated1)
            {
                IPWRTPhysTraceFaceN1Kernel<true, DEFORMED>(
                    0, 1, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10,
                    tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11);
            }
            else
            {
                IPWRTPhysTraceFaceN1Kernel<false, DEFORMED>(
                    0, 1, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10,
                    tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                    isCollocated10, isCollocated11);
            }
            break;
        case 2:
            if (endPtsCollocated0)
            {
                IPWRTPhysTraceFaceN0Kernel<true, DEFORMED>(
                    1, 2, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                    tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01);
            }
            else
            {
                IPWRTPhysTraceFaceN0Kernel<false, DEFORMED>(
                    1, 2, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                    tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01);
            }

            break;
        case 3:
            if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                if (endPtsCollocated0)
                {
                    IPWRTPhysTraceFaceN0Kernel<true, DEFORMED>(
                        0, 1, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                        tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                        isCollocated00, isCollocated01);
                }
                else
                {
                    IPWRTPhysTraceFaceN0Kernel<false, DEFORMED>(
                        0, 1, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                        tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                        isCollocated00, isCollocated01);
                }
            }
            else
            {
                if (endPtsCollocated1)
                {
                    IPWRTPhysTraceFaceN1Kernel<true, DEFORMED>(
                        1, 2, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10,
                        tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                        isCollocated10, isCollocated11);
                }
                else
                {
                    IPWRTPhysTraceFaceN1Kernel<false, DEFORMED>(
                        1, 2, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10,
                        tbasis11, tw10, tw11, tjac, wsp, wsp1, in, out,
                        isCollocated10, isCollocated11);
                }
            }
            break;
        case 4:
            if (endPtsCollocated0)
            {
                IPWRTPhysTraceFaceN0Kernel<true, DEFORMED>(
                    0, 1, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                    tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01);
            }
            else
            {
                IPWRTPhysTraceFaceN0Kernel<false, DEFORMED>(
                    0, 1, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00,
                    tbasis01, tw00, tw01, tjac, wsp, wsp1, in, out,
                    isCollocated00, isCollocated01);
            }
            break;
        case 5:
            if (endPtsCollocated2)
            {
                IPWRTPhysTraceFaceN2Kernel<true, DEFORMED>(
                    1, 2, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20,
                    tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21);
            }
            else
            {
                IPWRTPhysTraceFaceN2Kernel<false, DEFORMED>(
                    1, 2, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20,
                    tbasis21, tw20, tw21, tjac, wsp, wsp1, in, out,
                    isCollocated20, isCollocated21);
            }
            break;
    }
}

// -----------------------------------------------------------------------------
//  LAUNCHERS
// -----------------------------------------------------------------------------

/**
 * @brief Launcher for one element group of segments.
 *
 * The one-dimensional arm of the overload set OperatorND() calls; the
 * higher-dimensional arms follow below, and overload resolution picks
 * between them by the number of arguments the index sequences expand
 * to. A segment's traces are points carrying no surface measure, so it
 * takes no tangential table, no weight, no workspace and no tangential
 * collocation flag, and the trace Jacobian it is handed is null and
 * unread.
 *
 * @tparam SHAPE_TYPE   Seg; unread, the path being the same for every
 *                      one-dimensional expansion.
 * @tparam DEFORMED     Unread: a segment trace carries no Jacobian.
 * @tparam APPEND       Accumulate onto the volume field instead of
 *                      zeroing it first.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool APPEND,
          typename TTraceSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysTraceKernelLauncher(
    const TTraceSizeParameter1D sizeParam1D, const simd_type *nbasis0,
    [[maybe_unused]] const simd_type *tjac, const simd_type *in, simd_type *out,
    const bool endPtsCollocated0)
{
    static_assert(IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter1D or "
                  "TemplatedTraceSizeParameter1D.");

    if (endPtsCollocated0)
    {
        IProductWRTPhysTrace1DKernel<true, APPEND>(sizeParam1D.nm0(), nbasis0,
                                                   in, out);
    }
    else
    {
        IProductWRTPhysTrace1DKernel<false, APPEND>(sizeParam1D.nm0(), nbasis0,
                                                    in, out);
    }
}

/**
 * @brief All edges of one two-dimensional element group: optionally zero
 * the volume field, then walk the packed traces.
 *
 * The bulk entry point. It applies the direction-0 pair first, then
 * advances @p in by the pair's `2 * tnq00` values and @p tjac by the same
 * number of points when deformed or by two trace slots when regular, and
 * applies the direction-1 traces: a pair, or a single edge when
 * @c SHAPE_TYPE is exactly @c Tri, whose direction-1 upper end is the
 * collapsed vertex. Quadrilateral and triangle run the identical leaf
 * kernels; the hypotenuse is simply the \f$\eta_0 = +1\f$ line of the
 * collapsed square and its edge Jacobian carries the physical metric of
 * the slant.
 *
 * The @c endPtsCollocated flags are runtime booleans, so each call site is
 * written out twice to instantiate both arms of the leaf template.
 *
 * @tparam SHAPE_TYPE  Quad, Tri or NodalTri.
 * @tparam DEFORMED    Trace Jacobian varies point by point.
 * @tparam APPEND      Accumulate onto @p out instead of zeroing it first.
 * @tparam simd_type   SIMD vector type carrying one element per lane.
 *
 * @param   sizeParam2D     Volume and trace point counts of the shape.
 * @param   nbasis0,nbasis1 eInterp tables \f$h_p(\pm 1)\f$ per direction.
 * @param   tbasis0,tbasis1 eInterp tables to the trace points, per
 *                          direction.
 * @param   tw00,tw10       Trace quadrature weights per direction.
 * @param   tjac            Trace Jacobian of this element group.
 * @param   wsp             Trace-mode scratch, `2 * max(nm0,nm1)`
 *                          entries.
 * @param   in              Packed trace values of this element group.
 * @param   out             Volume field of this element group.
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
          typename TTraceSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysTraceKernelLauncher(
    const TTraceSizeParameter2D sizeParam2D, const simd_type *nbasis0,
    const simd_type *nbasis1, const simd_type *tbasis0,
    const simd_type *tbasis1, const simd_type *tw00, const simd_type *tw10,
    const simd_type *tjac, simd_type *wsp, const simd_type *in, simd_type *out,
    const bool isCollocated0, const bool isCollocated1,
    const bool endPtsCollocated0, const bool endPtsCollocated1)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    // Shape size.
    const unsigned nm0   = sizeParam2D.nm0();
    const unsigned nm1   = sizeParam2D.nm1();
    const unsigned tnq00 = sizeParam2D.nq00();
    const unsigned tnq10 = sizeParam2D.nq10();

    if constexpr (!APPEND)
    {
        // zero field
        std::memset((void *)out, 0, nm0 * nm1 * sizeof(simd_type));
    }

    if (endPtsCollocated0)
    {

        IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(0, 2, nm0, nm1, nbasis0,
                                                   tnq00, tbasis0, tw00, tjac,
                                                   wsp, in, out, isCollocated0);
    }
    else
    {

        IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
            0, 2, nm0, nm1, nbasis0, tnq00, tbasis0, tw00, tjac, wsp, in, out,
            isCollocated0);
    }

    unsigned offset  = 2 * tnq00; // edge offset for following edge
    unsigned joffset = (DEFORMED) ? 2 * tnq00 : 2;

    // edge 0 in tris, edges 0 + 2 in quads
    constexpr unsigned ntrace1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

    if (endPtsCollocated1)
    {
        IPWRTPhysTraceEdgeN1Kernel<true, DEFORMED>(
            0, ntrace1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10, tjac + joffset,
            wsp, in + offset, out, isCollocated1);
    }
    else
    {
        IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
            0, ntrace1, nm0, nm1, nbasis1, tnq10, tbasis1, tw10, tjac + joffset,
            wsp, in + offset, out, isCollocated1);
    }
}

/**
 * @brief All faces of one three-dimensional element group: optionally
 * zero the volume field, then walk the packed traces a pair at a time.
 *
 * The bulk entry point, and on this execution space the only one for
 * three dimensions: every shape reaches it, including the nodal ones.
 * It applies the direction-0 pair first, then advances @p in by that
 * pair's `2 * tnq00 * tnq01` values and @p tjac by the same count when
 * deformed or by two trace slots when regular, and repeats for
 * direction 1 and direction 2. Which of those carry a pair and which a
 * single trace is what the two @c if @c constexpr tests decide: a
 * tetrahedron has one direction-1 face, and everything but a hexahedron
 * has one direction-2 face.
 *
 * Because the pairs are applied together, @p wsp holds two faces' worth
 * of trace modes, which is how the block operator sizes it.
 *
 * The @c endPtsCollocated flags are runtime booleans, so each call site is
 * written out twice to instantiate both arms of the leaf template.
 *
 * @tparam SHAPE_TYPE  Hex, Prism, NodalPrism, Pyr, Tet or NodalTet.
 * @tparam DEFORMED    Trace Jacobian varies point by point.
 * @tparam APPEND      Accumulate onto @p out instead of zeroing it first.
 * @tparam simd_type   SIMD vector type carrying one element per lane.
 *
 * @param   sizeParam3D Volume and trace point counts of the shape.
 * @param   nbasis0,nbasis1,nbasis2   eInterp tables \f$h_p(\pm 1)\f$ per
 *                      direction.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21
 *                      eInterp tables to the trace points, per normal
 *                      direction and tangential direction.
 * @param   tw00,tw01,tw10,tw11,tw20,tw21   Trace quadrature weights in
 *                      the same order.
 * @param   tjac        Trace Jacobian of this element group.
 * @param   wsp         Trace-mode workspace, two faces' worth.
 * @param   wsp1        Scratch for the fully general face contraction.
 * @param   in          Packed trace values of this element group.
 * @param   out         Volume field of this element group.
 * @param   isCollocated00,isCollocated01   Tangential collocation flags
 *                      of the direction-0 faces.
 * @param   isCollocated10,isCollocated11   The same for direction 1.
 * @param   isCollocated20,isCollocated21   The same for direction 2.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2   Volume
 *                      rule of that direction contains the domain
 *                      endpoints.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool APPEND,
          typename TTraceSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysTraceKernelLauncher(
    const TTraceSizeParameter3D sizeParam3D, const simd_type *nbasis0,
    const simd_type *nbasis1, const simd_type *nbasis2,
    const simd_type *tbasis00, const simd_type *tbasis01,
    const simd_type *tbasis10, const simd_type *tbasis11,
    const simd_type *tbasis20, const simd_type *tbasis21, const simd_type *tw00,
    const simd_type *tw01, const simd_type *tw10, const simd_type *tw11,
    const simd_type *tw20, const simd_type *tw21, const simd_type *tjac,
    simd_type *wsp, simd_type *wsp1, const simd_type *in, simd_type *out,
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
    const unsigned nm0   = sizeParam3D.nm0();
    const unsigned nm1   = sizeParam3D.nm1();
    const unsigned nm2   = sizeParam3D.nm2();
    const unsigned tnq00 = sizeParam3D.nq00();
    const unsigned tnq01 = sizeParam3D.nq01();
    const unsigned tnq10 = sizeParam3D.nq10();
    const unsigned tnq11 = sizeParam3D.nq11();
    const unsigned tnq20 = sizeParam3D.nq20();
    const unsigned tnq21 = sizeParam3D.nq21();

    if constexpr (!APPEND)
    {
        // zero field
        std::memset((void *)out, 0, nm0 * nm1 * nm2 * sizeof(simd_type));
    }

    if (endPtsCollocated0)
    {
        IPWRTPhysTraceFaceN0Kernel<true, DEFORMED>(
            0, 2, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00, tbasis01,
            tw00, tw01, tjac, wsp, wsp1, in, out, isCollocated00,
            isCollocated01);
    }
    else
    {
        IPWRTPhysTraceFaceN0Kernel<false, DEFORMED>(
            0, 2, nm0, nm1, nm2, nbasis0, tnq00, tnq01, tbasis00, tbasis01,
            tw00, tw01, tjac, wsp, wsp1, in, out, isCollocated00,
            isCollocated01);
    }

    unsigned offset  = 2 * tnq00 * tnq01;
    unsigned joffset = (DEFORMED) ? offset : 2;

    constexpr unsigned nface1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

    if (endPtsCollocated1)
    {
        IPWRTPhysTraceFaceN1Kernel<true, DEFORMED>(
            0, nface1, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10, tbasis11,
            tw10, tw11, tjac + joffset, wsp, wsp1, in + offset, out,
            isCollocated10, isCollocated11);
    }
    else
    {
        IPWRTPhysTraceFaceN1Kernel<false, DEFORMED>(
            0, nface1, nm0, nm1, nm2, nbasis1, tnq10, tnq11, tbasis10, tbasis11,
            tw10, tw11, tjac + joffset, wsp, wsp1, in + offset, out,
            isCollocated10, isCollocated11);
    }
    offset += nface1 * tnq10 * tnq11;
    joffset += (DEFORMED) ? nface1 * tnq10 * tnq11 : nface1;

    constexpr unsigned nface2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];

    if (endPtsCollocated2)
    {
        IPWRTPhysTraceFaceN2Kernel<true, DEFORMED>(
            0, nface2, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20, tbasis21,
            tw20, tw21, tjac + joffset, wsp, wsp1, in + offset, out,
            isCollocated20, isCollocated21);
    }
    else
    {
        IPWRTPhysTraceFaceN2Kernel<false, DEFORMED>(
            0, nface2, nm0, nm1, nm2, nbasis2, tnq20, tnq21, tbasis20, tbasis21,
            tw20, tw21, tjac + joffset, wsp, wsp1, in + offset, out,
            isCollocated20, isCollocated21);
    }
}

} // namespace Nektar::Operators::detail
