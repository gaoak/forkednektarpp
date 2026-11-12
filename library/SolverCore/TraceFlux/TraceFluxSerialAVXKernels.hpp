///////////////////////////////////////////////////////////////////////////////
//
// File: TraceFluxSerialAVXKernels.hpp
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
// Description: The in-place interior trace flux kernels for the Serial and
// AVX execution spaces, for any point flux.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file TraceFluxSerialAVXKernels.hpp
 * @brief The in-place interior trace flux kernels for the Serial and AVX
 * execution spaces: the flux of a collocated interior or partition block
 * read from and written to the element-local trace fields through the
 * trace point maps, for any point flux (see TraceFluxKernels.hpp).
 *
 * @note TraceFluxOpImpl.hpp includes this header and
 * TraceFluxDeviceKernels.hpp. Each defines
 * InteriorBlockInPlaceKernelLauncher() for its own execution spaces only, so
 * the two never coincide. The kernels keep nothing between calls.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <SolverCore/TraceFlux/TraceFluxKernels.hpp>

#include <type_traits>

namespace Nektar::SolverCore::detail
{

/**
 * @brief A vector of @p valid consecutive points from a packed block array,
 * the spare lanes repeating the last valid point rather than reading past
 * the trace, which the block may end at. A full vector need not be aligned:
 * the arrays are padded per block, not per trace.
 */
template <typename vec_t, unsigned int LANE_WIDTH, typename TData>
NEK_FORCE_INLINE static vec_t LoadPacked(const TData *ptr, const unsigned valid)
{
    if constexpr (LANE_WIDTH == 1)
    {
        return vec_t(ptr[0]);
    }
    else
    {
        vec_t vec;
        if (valid == LANE_WIDTH)
        {
            vec.load(ptr, tinysimd::is_not_aligned);
            return vec;
        }
        alignas(64) TData tmp[LANE_WIDTH];
        for (unsigned l = 0; l < LANE_WIDTH; ++l)
        {
            tmp[l] = ptr[l < valid ? l : valid - 1];
        }
        vec.load(tmp);
        return vec;
    }
}

/// Whether the lanes' addresses @p off are consecutive.
template <unsigned int LANE_WIDTH>
NEK_FORCE_INLINE static bool LanesContiguous(const size_t *off)
{
    for (unsigned l = 1; l < LANE_WIDTH; ++l)
    {
        if (off[l] != off[0] + l)
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief A vector of one component at the lanes' element-local addresses
 * @p off: one unaligned vector load when they are consecutive (@p contig),
 * the usual case on a trace in its own orientation.
 */
template <typename vec_t, unsigned int LANE_WIDTH, typename TData>
NEK_FORCE_INLINE static vec_t GatherLanes(const TData *base, const size_t *off,
                                          [[maybe_unused]] const bool contig)
{
    if constexpr (LANE_WIDTH == 1)
    {
        return vec_t(base[off[0]]);
    }
    else
    {
        vec_t vec;
        if (contig)
        {
            vec.load(base + off[0], tinysimd::is_not_aligned);
            return vec;
        }
        alignas(64) TData tmp[LANE_WIDTH];
        for (unsigned l = 0; l < LANE_WIDTH; ++l)
        {
            tmp[l] = base[off[l]];
        }
        vec.load(tmp);
        return vec;
    }
}

/// Write, or add with @p APPEND, `sign * vec` to the first @p valid lanes'
/// element-local addresses @p off.
template <bool APPEND, typename vec_t, unsigned int LANE_WIDTH, typename TData>
NEK_FORCE_INLINE static void ScatterLanes(TData *base, const size_t *off,
                                          const vec_t vec, const TData sign,
                                          [[maybe_unused]] const unsigned valid,
                                          [[maybe_unused]] const bool contig)
{
    if constexpr (LANE_WIDTH == 1)
    {
        if constexpr (APPEND)
        {
            base[off[0]] += sign * TData(vec);
        }
        else
        {
            base[off[0]] = sign * TData(vec);
        }
    }
    else
    {
        if (contig && valid == LANE_WIDTH)
        {
            vec_t scaled = vec * vec_t(sign);
            if constexpr (APPEND)
            {
                vec_t cur;
                cur.load(base + off[0], tinysimd::is_not_aligned);
                scaled = scaled + cur;
            }
            scaled.store(base + off[0], tinysimd::is_not_aligned);
            return;
        }
        alignas(64) TData tmp[LANE_WIDTH];
        vec.store(tmp);
        for (unsigned l = 0; l < valid; ++l)
        {
            if constexpr (APPEND)
            {
                base[off[l]] += sign * tmp[l];
            }
            else
            {
                base[off[l]] = sign * tmp[l];
            }
        }
    }
}

/**
 * @brief The interior flux of one block read and written in place.
 *
 * For a block whose two sides are both collocated with the global trace:
 * each group of global trace points reads its two states and second inputs
 * straight from the element-local trace fields through the per-point maps
 * (TraceBlockView::ptMapFwd), and its trace normals and penalty factors from
 * the warehouse's global-trace arrays, evaluates the point flux through the
 * normal in registers, and writes the flux to both element-local sides,
 * negated on the backward one, or under @p PARALLEL to the local side only.
 * A trace takes whole vectors of its points, then one partial vector whose
 * spare lanes repeat the last point for the reads and are skipped on the
 * stores, so a trace never shares a vector with the next. Each side's points
 * are read at the stride of the element block the trace lies in
 * (TraceBlockView::stride), as one vector where that stride is 1 and lane by
 * lane otherwise.
 *
 * A tiled form of this kernel was removed from !2890 in November 2026 and
 * is in its history: the vector ran across the traces of one element group
 * at one trace point, so that at the lane width the owning side loaded and
 * stored whole vectors and only the far side was gathered, each trace
 * computed on one side with the normal reversed on the backward one. It
 * saved 3-4 % of an AVX Navier-Stokes step on an Apple M4 (hex 16^3 P3,
 * both viscous trace forms; Serial, Device and partition blocks unaffected)
 * for some 500 lines, a per-block plan kept per field layout, and a
 * contract that every point flux be antisymmetric under exchanging the
 * sides and reversing the normal. Worth revisiting only if the trace flux
 * becomes a larger share of the step than it is.
 *
 * @tparam NDIM           Spatial dimension: components of the normal.
 * @tparam APPEND         Add to the flux rather than overwrite it.
 * @tparam PARALLEL       The block is a partition block: side 1 is the
 *                        neighbour's element-local trace data from the
 *                        receive buffers (@p in1, @p second1), at width 1,
 *                        and the flux is written to the local side only.
 * @tparam PointFlux      The point flux (see TraceFluxKernels.hpp).
 * @param  views          The block's two sides and global-trace offsets.
 * @param  pointFlux      The point flux.
 * @param  inPtr,in1      The state field, and side 1's (the same field
 *                        unless PARALLEL).
 * @param  secondPtr,second1  The second input, likewise; unread by a point
 *                        flux without one.
 * @param  fluxPtr        The flux field.
 * @param  normPtr,penPtr The global-trace normals and penalty factors, as
 *                        the data warehouse holds them; the penalty factors
 *                        unread by a point flux without a penalty.
 * @param  streamID       Unused on the host spaces.
 */
template <unsigned int NDIM, bool APPEND, bool PARALLEL, typename ExecSpace,
          typename PointFlux, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                               std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                           bool>
              Enable = true>
void InteriorBlockInPlaceKernelLauncher(
    const InteriorBlockViews &views, const PointFlux &pointFlux,
    const TData *inPtr, const TData *in1, const TData *secondPtr,
    const TData *second1, TData *fluxPtr, const TData *normPtr,
    const TData *penPtr, [[maybe_unused]] const unsigned int streamID)
{
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    static constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;
    constexpr unsigned nVar    = PointFlux::numVar;
    constexpr unsigned nSecond = nVar * PointFlux::secondPerVar;
    static_assert(IsPointFlux<PointFlux, NDIM>(),
                  "PointFlux must present the point-flux interface of "
                  "TraceFluxKernels.hpp");

    const auto &T0        = views.side0;
    const auto &T1        = views.side1;
    const auto &glo       = views.glo;
    const unsigned npts   = T0.ptsPerTrace;
    const size_t numTrace = T0.numTrace;

    for (size_t trace = 0; trace < numTrace; ++trace)
    {
        const size_t cs0 = T0.compSize[trace], cs1 = T1.compSize[trace];
        const size_t s0 = T0.offset_st[trace], s1 = T1.offset_st[trace];
        const size_t b0 = T0.offset[trace], b1 = T1.offset[trace];
        const TData *nrmTrace = normPtr + glo.gloTOffset[trace];
        const size_t nrmCS    = glo.compSize[trace];
        // Unread by a point flux without a penalty.
        [[maybe_unused]] const TData *penTrace =
            PointFlux::usesPenalty ? penPtr + glo.scalarOffset[trace] : nullptr;
        for (unsigned p0 = 0; p0 < npts; p0 += vec_width)
        {
            const unsigned valid =
                (p0 + vec_width <= npts) ? vec_width : unsigned(npts - p0);

            // Element-local addresses of the lanes' points on each side, for
            // the state and flux fields and for the second input.
            // Side 1 of a partition block is the receive buffer, at stride 1.
            const unsigned stride0 = T0.stride[trace];
            const unsigned stride1 = T1.stride[trace];
            size_t o0[vec_width], o1[vec_width], d0[vec_width], d1[vec_width];
            for (unsigned l = 0; l < vec_width; ++l)
            {
                const unsigned pl = (l < valid) ? p0 + l : p0 + valid - 1;
                const size_t q0   = T0.MapFwd(trace, pl) * stride0;
                const size_t q1   = T1.MapFwd(trace, pl) * stride1;
                o0[l]             = s0 * nVar + b0 + q0;
                o1[l]             = s1 * nVar + b1 + q1;
                d0[l]             = s0 * nSecond + b0 + q0;
                d1[l]             = s1 * nSecond + b1 + q1;
            }
            // Consecutive points are a stride apart, so only at stride 1 can
            // the lanes of a side be one vector.
            const bool c0  = stride0 == 1 && LanesContiguous<vec_width>(o0);
            const bool c1  = stride1 == 1 && LanesContiguous<vec_width>(o1);
            const bool cd0 = stride0 == 1 && LanesContiguous<vec_width>(d0);
            const bool cd1 = stride1 == 1 && LanesContiguous<vec_width>(d1);

            vec_t fwd[nVar], bwd[nVar], flux[nVar];
            // One element, unread, for a point flux without a second input.
            [[maybe_unused]] vec_t secondFwd[nSecond > 0 ? nSecond : 1],
                secondBwd[nSecond > 0 ? nSecond : 1];
            for (unsigned n = 0; n < nVar; ++n)
            {
                fwd[n] = GatherLanes<vec_t, vec_width>(inPtr + n * cs0, o0, c0);
                bwd[n] = GatherLanes<vec_t, vec_width>(in1 + n * cs1, o1, c1);
            }
            for (unsigned c = 0; c < nSecond; ++c)
            {
                secondFwd[c] =
                    GatherLanes<vec_t, vec_width>(secondPtr + c * cs0, d0, cd0);
                secondBwd[c] =
                    GatherLanes<vec_t, vec_width>(second1 + c * cs1, d1, cd1);
            }

            // The global-trace points of a trace are consecutive, so the
            // normals and penalty factors load as vectors.
            vec_t normal[NDIM], penalty = vec_t(0.0);
            for (unsigned d = 0; d < NDIM; ++d)
            {
                normal[d] = LoadPacked<vec_t, vec_width>(
                    nrmTrace + d * nrmCS + p0, valid);
            }
            if constexpr (PointFlux::usesPenalty)
            {
                penalty = LoadPacked<vec_t, vec_width>(penTrace + p0, valid);
            }

            pointFlux(fwd, bwd, secondFwd, secondBwd, normal, penalty, flux);

            for (unsigned n = 0; n < nVar; ++n)
            {
                ScatterLanes<APPEND, vec_t, vec_width>(
                    fluxPtr + n * cs0, o0, flux[n], TData(1.0), valid, c0);
                if constexpr (!PARALLEL)
                {
                    ScatterLanes<APPEND, vec_t, vec_width>(
                        fluxPtr + n * cs1, o1, flux[n], TData(-1.0), valid, c1);
                }
            }
        }
    }
}

} // namespace Nektar::SolverCore::detail
