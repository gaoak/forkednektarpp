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
 * the two never coincide; this one also specialises InteriorInPlaceBlock for
 * AVX and defines BuildInteriorInPlaceBlock(), which fills it; Serial and
 * Device keep nothing.
 */

#pragma once

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <SolverCore/TraceFlux/TraceFluxKernels.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace Nektar::SolverCore::detail
{

/// No index: a trace not yet given an owning group trace, or a group trace's
/// masked lane, in BuildInteriorInPlaceBlock(). At namespace scope because GCC
/// 11 and 12 crash on a local class whose member initializer names a local of
/// the enclosing function template.
inline constexpr size_t noIndex = std::numeric_limits<size_t>::max();

/**
 * @brief The traces of an interior block that its group-trace plan leaves to
 * the per-trace kernel (InteriorInPlaceBlock::rest): copies of their per-trace
 * metadata, in the block's order, which Views() presents as a block of
 * their own.
 */
struct InteriorTraceRest
{
    size_t numTrace = 0;
    /// Per side, then per trace: TraceBlockView's per-trace arrays; the
    /// orientations only where the block has them.
    std::array<std::vector<size_t>, 2> offsetSt, offset, compSize;
    std::array<std::vector<int>, 2> orient;
    std::array<std::vector<unsigned>, 2> stride;
    /// Per trace: GloTraceOffsetView's arrays.
    std::vector<size_t> gloTOffset, scalarOffset, gloCompSize;

    /// The traces as the views of a block, with the whole block's tables
    /// (the point maps and points per trace) from @p all, its views.
    InteriorBlockViews Views(const InteriorBlockViews &all) const
    {
        InteriorBlockViews v                  = all;
        std::array<TraceBlockView *, 2> sides = {&v.side0, &v.side1};
        for (unsigned s = 0; s < 2; ++s)
        {
            sides[s]->offset_st = offsetSt[s].data();
            sides[s]->offset    = offset[s].data();
            sides[s]->compSize  = compSize[s].data();
            sides[s]->orient   = orient[s].empty() ? nullptr : orient[s].data();
            sides[s]->stride   = stride[s].data();
            sides[s]->numTrace = numTrace;
        }
        v.glo.gloTOffset   = gloTOffset.data();
        v.glo.scalarOffset = scalarOffset.data();
        v.glo.compSize     = gloCompSize.data();
        return v;
    }
};

/**
 * @brief What the in-place kernels of one interior block keep between calls
 * on AVX: the plan of the across-group form, built by
 * BuildInteriorInPlaceBlock()
 * and run by InteriorBlockGroupTraceKernel(). Serial and Device keep nothing
 * (the primary template, TraceFluxKernels.hpp).
 *
 * A group trace is one element group's copy of one local trace: in the
 * interleaved element-local layout its lanes at one trace point are adjacent,
 * so the group trace's own side loads and stores as whole vectors. Each trace
 * is computed in the group trace of one of its sides (see
 * BuildInteriorInPlaceBlock()); lanes whose trace is computed elsewhere are
 * masked. Only group traces whose other sides lie in few memory regions and
 * whose lanes are well filled are kept; the traces of the others are listed in
 * @p rest for the per-trace kernel, so a block is taken across the group in
 * part where its numbering aligns some group traces and not others.
 */
template <typename TData> struct InteriorInPlaceBlock<NektarSpaces::AVX, TData>
{
    /// The group traces kept; 0 where the block takes the per-trace form
    /// throughout, when @p rest is unset and the block's own views serve.
    size_t numGroupTrace = 0;
    unsigned npts        = 0;
    /// The traces not owned by a kept group trace.
    InteriorTraceRest rest;
    /// Per group trace: the owning side's element block
    /// (TraceBlockView::offset_st) and the point offset of lane 0 at trace
    /// point 0, which together with a field's component count address its lane
    /// 0 (FieldOffset()); the owning side's component stride; the other
    /// side's element block and component stride, one block for all lanes
    /// (BuildInteriorInPlaceBlock()); the lanes that own a trace, bit l for
    /// lane l.
    std::vector<size_t> ownOffsetSt, ownOffsetPt, ownCompSize, nbrOffsetSt,
        nbrCompSize;
    std::vector<unsigned> laneMask;
    /// Per (group trace, point, lane): the other side's point offset; a
    /// masked lane's is a copy of a live lane's, so its reads stay in bounds.
    std::vector<size_t> nbrOffsetPt;
    /// Per (group trace, lane): where the lane's trace sits in the warehouse's
    /// global-trace normals (offset and component stride) and penalty
    /// factors, and the sign its normal takes on the side computed.
    std::vector<size_t> nrmBase, nrmCompSize, penBase;
    std::vector<TData> sign;
    /// Per (group trace, point, lane): the global-trace point the lane's trace
    /// point lands on, which indexes both from the bases above.
    std::vector<unsigned> gloPt;

    /// Where component 0 of a trace point sits in a field of @p numComp
    /// components, from the point's element block @p offsetSt and point
    /// offset @p offsetPt: the element-local trace layout
    /// (LocPointToGloPointBlock()).
    static constexpr size_t FieldOffset(const size_t offsetSt,
                                        const size_t offsetPt,
                                        const unsigned numComp)
    {
        return offsetSt * numComp + offsetPt;
    }
};

/**
 * @brief Build the group-trace plan of the interior block @p views in one
 * field layout, when that layout is first seen
 * (TraceFluxOpImpl::BuildLayout()); see
 * InteriorInPlaceBlock<NektarSpaces::AVX, TData>. Partition blocks have no
 * plan: the across-group form covers interior blocks only.
 *
 * Each trace can sit in the group trace of either side: its forward side with
 * the trace normal, or its backward side with the normal reversed (the point
 * flux is antisymmetric under exchanging the sides and the normal, so the
 * flux written to each side is the same to round-off). Two assignments are
 * built and the one that puts more traces into well-filled group traces kept:
 * every trace on its forward side, which reproduces the point-vector form
 * bit for bit; and a greedy one in rounds on the graph whose edges are the
 * traces: a maximal independent set of group traces, each member taking all its
 * traces (so it is full), then among the group traces left those that can still
 * fill well from the traces between unkept group traces, the remaining traces
 * their forward side. A group trace is kept when the
 * other sides of its traces lie in at most two other group traces, so that
 * the gathers of the other side walk few memory regions (one row on a
 * lexicographically numbered hexahedral mesh; up to eight regions on an
 * unstructured tetrahedral mesh, where the across-group form measured slower
 * than the per-trace kernel with every lane live), and when at least three
 * quarters of its lanes are live, since a masked lane costs a full lane of
 * flux arithmetic. The traces of the group traces dropped go to
 * InteriorInPlaceBlock::rest; a block without a kept group trace has no plan
 * and takes the per-trace kernel on its own views.
 *
 * The plan holds offsets only: the trace normals and penalty factors are
 * read from the data warehouse's global-trace arrays through them, and the
 * element-local fields through each point's element block and point offset
 * (InteriorInPlaceBlock::FieldOffset()), so one plan serves fields of any
 * component count.
 *
 * @param  views    The block's two sides, in the layout of the fields the
 *                  plan will address (TraceBlockView::stride per trace), and
 *                  where its traces sit in the global trace.
 * @param  plan     The plan, overwritten; no group traces unless both sides of
 *                  every trace are at the lane width.
 */
template <typename TData>
void BuildInteriorInPlaceBlock(
    const InteriorBlockViews &views,
    InteriorInPlaceBlock<NektarSpaces::AVX, TData> &plan)
{
    const TraceBlockView &T0      = views.side0;
    const TraceBlockView &T1      = views.side1;
    const GloTraceOffsetView &glo = views.glo;
    constexpr unsigned laneWidth  = tinysimd::simd<TData>::width;
    const unsigned npts           = T0.ptsPerTrace;
    const size_t numTrace         = T0.numTrace;

    plan      = InteriorInPlaceBlock<NektarSpaces::AVX, TData>{};
    plan.npts = npts;

    // A group trace's own side loads and stores whole vectors only where its
    // element block is interleaved at the lane width, and either side of a
    // trace may own it, so the across-group form needs both sides of every
    // trace at the lane width; a block mixing widths takes the per-trace form
    // throughout.
    for (size_t t = 0; t < numTrace; ++t)
    {
        if (T0.stride[t] != laneWidth || T1.stride[t] != laneWidth)
        {
            return;
        }
    }

    /// Where a trace's side sits: its group trace, by its element block and the
    /// point offset of the group trace's lane 0, and its lane in it.
    using GroupTraceKey = std::pair<size_t, size_t>;
    struct Side
    {
        GroupTraceKey gt;
        unsigned lane;
    };
    const auto sideOf = [](const TraceBlockView &T, const size_t t) {
        // An element block holds whole element groups, so its point offsets
        // start at a lane boundary.
        ASSERTL1(T.offset_st[t] % laneWidth == 0,
                 "Element blocks of the interleaved layout are lane-aligned");
        const size_t pt = T.offset[t];
        return Side{{T.offset_st[t], pt - pt % laneWidth},
                    unsigned(pt % laneWidth)};
    };
    std::vector<Side> side0(numTrace), side1(numTrace);
    std::map<GroupTraceKey, size_t> gtIdx;
    for (size_t t = 0; t < numTrace; ++t)
    {
        side0[t] = sideOf(T0, t);
        side1[t] = sideOf(T1, t);
        gtIdx.emplace(side0[t].gt, 0);
        gtIdx.emplace(side1[t].gt, 0);
    }
    {
        size_t idx = 0;
        for (auto &gtEntry : gtIdx)
        {
            gtEntry.second = idx++;
        }
    }
    const size_t nGroupTraceAll = gtIdx.size();
    std::vector<std::vector<size_t>> incidentTraces(
        nGroupTraceAll); // group trace -> traces
    std::vector<size_t> gtFwd(numTrace), gtBwd(numTrace);
    std::vector<unsigned char> selfLoop(nGroupTraceAll, 0);
    for (size_t t = 0; t < numTrace; ++t)
    {
        gtFwd[t] = gtIdx[side0[t].gt];
        gtBwd[t] = gtIdx[side1[t].gt];
        incidentTraces[gtFwd[t]].push_back(t);
        if (gtBwd[t] != gtFwd[t])
        {
            incidentTraces[gtBwd[t]].push_back(t);
        }
        else
        {
            selfLoop[gtFwd[t]] = 1;
        }
    }
    // What the across-group form costs is set by how many separate memory
    // regions the other side's gathers walk, not by the lanes' fill: on a
    // lexicographically numbered hexahedral mesh a group trace's other sides
    // lie in one other group trace (two for the direction along the
    // groups), on an unstructured tetrahedral mesh in up to eight. Measured
    // on AVX-512 (8 lanes) the across-group form at eight regions costs 1.5x
    // the per-trace kernel per trace with every lane live (tet 3439 P3), at
    // one or two 0.55x (hex 16^3 P3). So a group trace is a candidate only
    // when the other sides of all its traces lie in at most two other group
    // traces, in one element block, so that the kernel has one component
    // stride for them.
    constexpr unsigned maxRegions = 2;
    std::vector<unsigned char> fewRegions(nGroupTraceAll, 1);
    for (size_t i = 0; i < nGroupTraceAll; ++i)
    {
        if (selfLoop[i])
        {
            fewRegions[i] = 0;
            continue;
        }
        std::array<size_t, maxRegions> other;
        unsigned nOther = 0;
        // The other side's element block, from the first trace.
        const auto otherBlock = [&](const size_t t) {
            return (gtFwd[t] == i ? T1 : T0).offset_st[t];
        };
        const size_t block = otherBlock(incidentTraces[i].front());
        for (const size_t t : incidentTraces[i])
        {
            const size_t o = gtFwd[t] == i ? gtBwd[t] : gtFwd[t];
            if (otherBlock(t) != block)
            {
                fewRegions[i] = 0;
                break;
            }
            unsigned k = 0;
            while (k < nOther && other[k] != o)
            {
                ++k;
            }
            if (k == nOther)
            {
                if (nOther == maxRegions)
                {
                    fewRegions[i] = 0;
                    break;
                }
                other[nOther++] = o;
            }
        }
    }
    // Live lanes a group trace needs to be kept: three quarters of them.
    constexpr unsigned minLive = (3 * laneWidth + 3) / 4;
    // The greedy assignment, in rounds. Round 1: a maximal independent set
    // of group traces in index order, each member taking all its traces, so it
    // is full. Later rounds: among the group traces not kept, those that can
    // still reach minLive from the traces neither of whose group traces is
    // kept, fullest first, each taking all of those. The traces left go to
    // their forward side. On tet 3439 P3 at 8 lanes the later rounds lift the
    // traces in kept group traces from 56 % to 74 %, the group traces they add
    // at six or seven lanes, since a few of each one's lanes are owned by
    // round-1 group traces.
    std::vector<size_t> owner(numTrace, noIndex);
    std::vector<unsigned char> kept(nGroupTraceAll, 0);
    std::vector<unsigned> attainable(nGroupTraceAll);
    std::vector<size_t> order;
    for (unsigned round = 0;; ++round)
    {
        std::fill(attainable.begin(), attainable.end(), 0u);
        for (size_t t = 0; t < numTrace; ++t)
        {
            if (owner[t] == noIndex && gtFwd[t] != gtBwd[t])
            {
                ++attainable[gtFwd[t]];
                ++attainable[gtBwd[t]];
            }
        }
        order.clear();
        for (size_t i = 0; i < nGroupTraceAll; ++i)
        {
            if (!kept[i] && fewRegions[i] && attainable[i] >= minLive)
            {
                order.push_back(i);
            }
        }
        if (order.empty())
        {
            break;
        }
        if (round > 0)
        {
            std::stable_sort(order.begin(), order.end(),
                             [&](const size_t a, const size_t b) {
                                 return attainable[a] > attainable[b];
                             });
        }
        std::vector<unsigned char> keptNow(nGroupTraceAll, 0),
            blocked(nGroupTraceAll, 0);
        for (const size_t i : order)
        {
            if (blocked[i])
            {
                continue;
            }
            keptNow[i] = kept[i] = 1;
            for (const size_t t : incidentTraces[i])
            {
                if (owner[t] == noIndex)
                {
                    blocked[gtFwd[t] == i ? gtBwd[t] : gtFwd[t]] = 1;
                }
            }
        }
        for (size_t t = 0; t < numTrace; ++t)
        {
            if (owner[t] == noIndex && keptNow[gtFwd[t]] != keptNow[gtBwd[t]])
            {
                owner[t] = keptNow[gtFwd[t]] ? gtFwd[t] : gtBwd[t];
            }
        }
    }
    // Whether each trace is computed on its backward side.
    std::vector<unsigned char> onBwdMis(numTrace), onBwdFwd(numTrace, 0);
    for (size_t t = 0; t < numTrace; ++t)
    {
        onBwdMis[t] = owner[t] == gtBwd[t] && gtFwd[t] != gtBwd[t];
    }
    // The traces an assignment puts into kept group traces.
    auto keptTraces = [&](const std::vector<unsigned char> &onBwd) {
        std::vector<unsigned> live(nGroupTraceAll, 0);
        for (size_t t = 0; t < numTrace; ++t)
        {
            ++live[onBwd[t] ? gtBwd[t] : gtFwd[t]];
        }
        size_t num = 0;
        for (size_t i = 0; i < nGroupTraceAll; ++i)
        {
            num += fewRegions[i] && live[i] >= minLive ? live[i] : 0;
        }
        return num;
    };
    const std::vector<unsigned char> &onBwd =
        keptTraces(onBwdMis) > keptTraces(onBwdFwd) ? onBwdMis : onBwdFwd;

    /// The trace a lane computes and on which side; noIndex when masked.
    struct Lane
    {
        size_t trace = noIndex;
        bool bwd     = false;
    };
    std::map<GroupTraceKey, std::array<Lane, laneWidth>> groupTraces;
    for (size_t t = 0; t < numTrace; ++t)
    {
        const Side &sd              = onBwd[t] ? side1[t] : side0[t];
        groupTraces[sd.gt][sd.lane] = {t, bool(onBwd[t])};
    }

    // Keep the well-filled group traces with few streams; the traces of the
    // others are left to the per-trace kernel, in the block's order.
    std::vector<size_t> restTrace;
    for (auto it = groupTraces.begin(); it != groupTraces.end();)
    {
        unsigned live = 0;
        for (const Lane &ln : it->second)
        {
            live += ln.trace != noIndex;
        }
        if (live >= minLive && fewRegions[gtIdx[it->first]])
        {
            ++it;
            continue;
        }
        for (const Lane &ln : it->second)
        {
            if (ln.trace != noIndex)
            {
                restTrace.push_back(ln.trace);
            }
        }
        it = groupTraces.erase(it);
    }
    if (groupTraces.empty())
    {
        // Per-trace throughout, on the block's own views.
        return;
    }
    std::sort(restTrace.begin(), restTrace.end());
    {
        InteriorTraceRest &rest       = plan.rest;
        const TraceBlockView *sides[] = {&T0, &T1};
        rest.numTrace                 = restTrace.size();
        for (unsigned s = 0; s < 2; ++s)
        {
            const TraceBlockView &T = *sides[s];
            for (const size_t t : restTrace)
            {
                rest.offsetSt[s].push_back(T.offset_st[t]);
                rest.offset[s].push_back(T.offset[t]);
                rest.compSize[s].push_back(T.compSize[t]);
                rest.stride[s].push_back(T.stride[t]);
                if (T.orient)
                {
                    rest.orient[s].push_back(T.orient[t]);
                }
            }
        }
        for (const size_t t : restTrace)
        {
            rest.gloTOffset.push_back(glo.gloTOffset[t]);
            rest.scalarOffset.push_back(glo.scalarOffset[t]);
            rest.gloCompSize.push_back(glo.compSize[t]);
        }
    }

    plan.numGroupTrace = groupTraces.size();
    plan.ownOffsetSt.assign(plan.numGroupTrace, 0);
    plan.ownOffsetPt.assign(plan.numGroupTrace, 0);
    plan.ownCompSize.assign(plan.numGroupTrace, 0);
    plan.nbrOffsetSt.assign(plan.numGroupTrace, 0);
    plan.nbrCompSize.assign(plan.numGroupTrace, 0);
    plan.laneMask.assign(plan.numGroupTrace, 0u);
    plan.nbrOffsetPt.assign(plan.numGroupTrace * npts * laneWidth, 0);
    plan.nrmBase.assign(plan.numGroupTrace * laneWidth, 0);
    plan.nrmCompSize.assign(plan.numGroupTrace * laneWidth, 0);
    plan.penBase.assign(plan.numGroupTrace * laneWidth, 0);
    plan.sign.assign(plan.numGroupTrace * laneWidth, TData(1));
    plan.gloPt.assign(plan.numGroupTrace * npts * laneWidth, 0);
    std::vector<unsigned> inv(npts);

    size_t gt = 0;
    for (const auto &gtEntry : groupTraces)
    {
        const auto &lanes    = gtEntry.second;
        plan.ownOffsetSt[gt] = gtEntry.first.first;
        plan.ownOffsetPt[gt] = gtEntry.first.second;
        const size_t lane0   = gt * laneWidth;
        // The live lanes first; the masked lanes then copy a live lane's
        // other-side offsets, normal and penalty, so that their reads stay
        // in bounds and finite, and are never stored (laneMask).
        unsigned firstLive = laneWidth;
        for (unsigned l = 0; l < laneWidth; ++l)
        {
            if (lanes[l].trace == noIndex)
            {
                continue;
            }
            const size_t t  = lanes[l].trace;
            const bool bwd  = lanes[l].bwd;
            const auto &Own = bwd ? T1 : T0;
            const auto &Nbr = bwd ? T0 : T1;
            if (firstLive == laneWidth)
            {
                firstLive            = l;
                plan.ownCompSize[gt] = Own.compSize[t];
                plan.nbrOffsetSt[gt] = Nbr.offset_st[t];
                plan.nbrCompSize[gt] = Nbr.compSize[t];
            }
            ASSERTL1(Own.compSize[t] == plan.ownCompSize[gt] &&
                         Nbr.offset_st[t] == plan.nbrOffsetSt[gt] &&
                         Nbr.compSize[t] == plan.nbrCompSize[gt],
                     "A group trace's lanes lie in one element block on "
                     "each side.");
            plan.laneMask[gt] |= 1u << l;
            plan.nrmBase[lane0 + l]     = glo.gloTOffset[t];
            plan.nrmCompSize[lane0 + l] = glo.compSize[t];
            plan.penBase[lane0 + l]     = glo.scalarOffset[t];
            plan.sign[lane0 + l]        = bwd ? TData(-1) : TData(1);
            // Global point of each own-side local point.
            for (unsigned gloPoint = 0; gloPoint < npts; ++gloPoint)
            {
                inv[Own.MapFwd(t, gloPoint)] = gloPoint;
            }
            for (unsigned q = 0; q < npts; ++q)
            {
                const unsigned gloPoint = inv[q];
                const unsigned qn       = Nbr.MapFwd(t, gloPoint);
                plan.nbrOffsetPt[(gt * npts + q) * laneWidth + l] =
                    Nbr.offset[t] + qn * Nbr.stride[t];
                plan.gloPt[(gt * npts + q) * laneWidth + l] = gloPoint;
            }
        }
        for (unsigned l = 0; l < laneWidth; ++l)
        {
            if (lanes[l].trace != noIndex)
            {
                continue;
            }
            plan.nrmBase[lane0 + l]     = plan.nrmBase[lane0 + firstLive];
            plan.nrmCompSize[lane0 + l] = plan.nrmCompSize[lane0 + firstLive];
            plan.penBase[lane0 + l]     = plan.penBase[lane0 + firstLive];
            for (unsigned q = 0; q < npts; ++q)
            {
                const size_t at          = (gt * npts + q) * laneWidth;
                plan.nbrOffsetPt[at + l] = plan.nbrOffsetPt[at + firstLive];
                plan.gloPt[at + l]       = plan.gloPt[at + firstLive];
            }
        }
        ++gt;
    }
}

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

/// Write, or add with @p APPEND, `sign * vec` to the element-local addresses
/// @p off of the lanes set in @p mask (bit l for lane l), as one unaligned
/// vector where they are consecutive (@p contig) and all set, lane by lane
/// otherwise.
template <bool APPEND, typename vec_t, unsigned int LANE_WIDTH, typename TData>
NEK_FORCE_INLINE static void ScatterLanes(TData *base, const size_t *off,
                                          const vec_t vec, const TData sign,
                                          [[maybe_unused]] const unsigned mask,
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
        constexpr unsigned all = (1u << LANE_WIDTH) - 1u;
        if (contig && mask == all)
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
        // Up to the highest lane set, so a prefix mask (a trace's partial
        // vector) costs its valid lanes only.
        for (unsigned m = mask, l = 0; m != 0u; m >>= 1, ++l)
        {
            if (!(m & 1u))
            {
                continue;
            }
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
 * @brief Where the lanes of one vector step of the in-place interior flux
 * read their trace normals and penalty factors: along a trace, whose
 * global-trace points are consecutive, so they load packed.
 */
template <typename TData> struct AlongTraceNormals
{
    /// The trace's normals at the step's first point, and their component
    /// stride.
    const TData *nrm = nullptr;
    size_t nrmCS     = 0;
    /// The trace's penalty factors at the step's first point; unread by a
    /// point flux without a penalty.
    const TData *pen = nullptr;
    /// The lanes that are points of the trace; the rest repeat the last.
    unsigned valid = 0;
};

/**
 * @brief Where the lanes of one vector step read their trace normals and
 * penalty factors: across a group trace, every lane a different trace, so
 * they are gathered from the warehouse's global-trace arrays through the
 * plan's per-lane offsets, the normal taking the sign of the side computed.
 */
template <typename TData> struct AcrossGroupNormals
{
    const TData *normPtr = nullptr, *penPtr = nullptr;
    /// Per lane: InteriorInPlaceBlock::nrmBase, nrmCompSize, penBase, sign.
    const size_t *nrmBase = nullptr, *nrmCS = nullptr, *penBase = nullptr;
    const TData *sign = nullptr;
    /// Per lane: InteriorInPlaceBlock::gloPt at the step's point.
    const unsigned *gloPt = nullptr;
};

template <typename vec_t, unsigned int LANE_WIDTH, unsigned int NDIM,
          bool USES_PENALTY, typename TData>
NEK_FORCE_INLINE static void LoadNormals(const AlongTraceNormals<TData> &src,
                                         vec_t *normal, vec_t &penalty)
{
    for (unsigned d = 0; d < NDIM; ++d)
    {
        normal[d] =
            LoadPacked<vec_t, LANE_WIDTH>(src.nrm + d * src.nrmCS, src.valid);
    }
    if constexpr (USES_PENALTY)
    {
        penalty = LoadPacked<vec_t, LANE_WIDTH>(src.pen, src.valid);
    }
}

template <typename vec_t, unsigned int LANE_WIDTH, unsigned int NDIM,
          bool USES_PENALTY, typename TData>
NEK_FORCE_INLINE static void LoadNormals(const AcrossGroupNormals<TData> &src,
                                         vec_t *normal, vec_t &penalty)
{
    alignas(64) TData tmp[LANE_WIDTH];
    alignas(64) TData tmpNrm[NDIM][LANE_WIDTH];
    for (unsigned l = 0; l < LANE_WIDTH; ++l)
    {
        const TData *nrm = src.normPtr + src.nrmBase[l] + src.gloPt[l];
        for (unsigned d = 0; d < NDIM; ++d)
        {
            tmpNrm[d][l] = src.sign[l] * nrm[d * src.nrmCS[l]];
        }
        if constexpr (USES_PENALTY)
        {
            tmp[l] = src.penPtr[src.penBase[l] + src.gloPt[l]];
        }
    }
    for (unsigned d = 0; d < NDIM; ++d)
    {
        normal[d].load(tmpNrm[d]);
    }
    if constexpr (USES_PENALTY)
    {
        penalty.load(tmp);
    }
}

/**
 * @brief One vector step of the in-place interior flux: the lanes' states
 * and second inputs gathered from both sides, the point flux evaluated
 * through the normal in registers, the flux written to both sides, negated
 * on side 1, or to side 0 only under @p PARTITION.
 *
 * The two kernels differ only in what a lane is. Along a trace
 * (InteriorBlockInPlaceKernelLauncher()) the lanes are consecutive points
 * of one trace; across a group trace (InteriorBlockGroupTraceKernel()) they
 * are the elements of a group at one point, side 0 the owning side. Both
 * hand this the lanes' element-local addresses and whether they are
 * consecutive, the component strides, the lanes to store (@p mask, bit l for
 * lane l; the others read a repeated point and are never written), and
 * where the normals come from (@p normals, AlongTraceNormals or
 * AcrossGroupNormals).
 *
 * @param  o0,o1   Per lane, component 0's address on each side in the
 *                 state and flux layout; @p d0, @p d1 likewise in the
 *                 second-input layout.
 * @param  cs0,cs1 Component strides of the two sides.
 * @param  c0,c1,cd0,cd1  Whether each address set is consecutive.
 */
template <unsigned int NDIM, bool APPEND, bool PARTITION, typename vec_t,
          unsigned int LANE_WIDTH, typename PointFlux, typename Normals,
          typename TData>
NEK_FORCE_INLINE static void InteriorLanesFlux(
    const PointFlux &pointFlux, const size_t *o0, const size_t *o1,
    const size_t *d0, const size_t *d1, const size_t cs0, const size_t cs1,
    const bool c0, const bool c1, const bool cd0, const bool cd1,
    const unsigned mask, const Normals &normals, const TData *inPtr,
    const TData *in1, const TData *secondPtr, const TData *second1,
    TData *fluxPtr)
{
    constexpr unsigned nVar    = PointFlux::numVar;
    constexpr unsigned nSecond = nVar * PointFlux::secondPerVar;

    vec_t fwd[nVar], bwd[nVar], flux[nVar];
    // One element, unread, for a point flux without a second input.
    [[maybe_unused]] vec_t secondFwd[nSecond > 0 ? nSecond : 1],
        secondBwd[nSecond > 0 ? nSecond : 1];
    for (unsigned n = 0; n < nVar; ++n)
    {
        fwd[n] = GatherLanes<vec_t, LANE_WIDTH>(inPtr + n * cs0, o0, c0);
        bwd[n] = GatherLanes<vec_t, LANE_WIDTH>(in1 + n * cs1, o1, c1);
    }
    for (unsigned c = 0; c < nSecond; ++c)
    {
        secondFwd[c] =
            GatherLanes<vec_t, LANE_WIDTH>(secondPtr + c * cs0, d0, cd0);
        secondBwd[c] =
            GatherLanes<vec_t, LANE_WIDTH>(second1 + c * cs1, d1, cd1);
    }

    vec_t normal[NDIM], penalty = vec_t(0.0);
    LoadNormals<vec_t, LANE_WIDTH, NDIM, PointFlux::usesPenalty>(
        normals, normal, penalty);

    pointFlux(fwd, bwd, secondFwd, secondBwd, normal, penalty, flux);

    for (unsigned n = 0; n < nVar; ++n)
    {
        ScatterLanes<APPEND, vec_t, LANE_WIDTH>(fluxPtr + n * cs0, o0, flux[n],
                                                TData(1.0), mask, c0);
        if constexpr (!PARTITION)
        {
            ScatterLanes<APPEND, vec_t, LANE_WIDTH>(
                fluxPtr + n * cs1, o1, flux[n], TData(-1.0), mask, c1);
        }
    }
}

/**
 * @brief The interior flux of one block, group trace by group trace.
 *
 * The same vector step as InteriorBlockInPlaceKernelLauncher()
 * (InteriorLanesFlux()), with the vector running across the traces of a
 * group trace at one trace point instead of along the points of one trace:
 * the owning side's lanes are consecutive, so it loads and stores as whole
 * vectors, and the other side's are gathered unless the group trace's other
 * sides form a row too, when they load as vectors as well. The normals and
 * penalty factors are gathered from the warehouse's global-trace arrays
 * through the plan's offsets (AcrossGroupNormals). AVX, every field at the
 * lane width.
 *
 * @tparam NDIM      Spatial dimension: components of the normal.
 * @tparam APPEND    Add to the flux rather than overwrite it.
 * @tparam PointFlux The point flux (see TraceFluxKernels.hpp).
 * @param  plan      The block's group-trace plan (BuildInteriorInPlaceBlock()).
 * @param  pointFlux The point flux.
 * @param  inPtr,secondPtr,fluxPtr  The element-local state, second input
 *                   and flux fields, at their bases.
 * @param  normPtr,penPtr  The global-trace normals and penalty factors.
 */
template <unsigned int NDIM, bool APPEND, typename PointFlux, typename TData>
void InteriorBlockGroupTraceKernel(
    const InteriorInPlaceBlock<NektarSpaces::AVX, TData> &plan,
    const PointFlux &pointFlux, const TData *inPtr, const TData *secondPtr,
    TData *fluxPtr, const TData *normPtr, const TData *penPtr)
{
    using vec_t                  = tinysimd::simd<TData>;
    constexpr unsigned laneWidth = vec_t::width;
    constexpr unsigned nVar      = PointFlux::numVar;
    constexpr unsigned nSecond   = nVar * PointFlux::secondPerVar;
    static_assert(IsPointFlux<PointFlux, NDIM>(),
                  "PointFlux must present the point-flux interface of "
                  "TraceFluxKernels.hpp");
    const unsigned npts = plan.npts;

    size_t o0[laneWidth], o1[laneWidth], d0[laneWidth], d1[laneWidth];
    for (size_t k = 0; k < plan.numGroupTrace; ++k)
    {
        const size_t lane0 = k * laneWidth;
        const size_t ownSt = plan.ownOffsetSt[k], ownPt = plan.ownOffsetPt[k];
        const size_t nbrSt     = plan.nbrOffsetSt[k];
        const size_t ownState  = plan.FieldOffset(ownSt, ownPt, nVar);
        const size_t ownSecond = plan.FieldOffset(ownSt, ownPt, nSecond);
        const size_t cs0 = plan.ownCompSize[k], cs1 = plan.nbrCompSize[k];
        const unsigned mask = plan.laneMask[k];
        AcrossGroupNormals<TData> normals;
        normals.normPtr = normPtr;
        normals.penPtr  = penPtr;
        normals.nrmBase = plan.nrmBase.data() + lane0;
        normals.nrmCS   = plan.nrmCompSize.data() + lane0;
        normals.penBase = plan.penBase.data() + lane0;
        normals.sign    = plan.sign.data() + lane0;
        for (unsigned q = 0; q < npts; ++q)
        {
            const size_t *nbrOffsetPt =
                plan.nbrOffsetPt.data() + (k * npts + q) * laneWidth;
            normals.gloPt = plan.gloPt.data() + (k * npts + q) * laneWidth;
            for (unsigned l = 0; l < laneWidth; ++l)
            {
                o0[l] = ownState + q * laneWidth + l;
                d0[l] = ownSecond + q * laneWidth + l;
                o1[l] = plan.FieldOffset(nbrSt, nbrOffsetPt[l], nVar);
                d1[l] = plan.FieldOffset(nbrSt, nbrOffsetPt[l], nSecond);
            }
            const bool c1  = LanesContiguous<laneWidth>(o1);
            const bool cd1 = LanesContiguous<laneWidth>(d1);
            InteriorLanesFlux<NDIM, APPEND, false, vec_t, laneWidth>(
                pointFlux, o0, o1, d0, d1, cs0, cs1, true, c1, true, cd1, mask,
                normals, inPtr, inPtr, secondPtr, secondPtr, fluxPtr);
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
 * negated on the backward one, or under @p PARTITION to the local side only
 * (the vector step: InteriorLanesFlux()).
 * A trace takes whole vectors of its points, then one partial vector whose
 * spare lanes repeat the last point for the reads and are skipped on the
 * stores, so a trace never shares a vector with the next. Each side's points
 * are read at the stride of the element block the trace lies in
 * (TraceBlockView::stride), as one vector where that stride is 1 and lane by
 * lane otherwise. On AVX the well-filled group traces of an interior block's
 * group-trace plan are sent to InteriorBlockGroupTraceKernel() by the caller
 * (TraceFluxOpImpl::TakeBlockInPlace()), which then calls this for the
 * block's remaining traces (InteriorInPlaceBlock::rest) as a block of their
 * own; the dispatch is not made here because with it in this function the
 * per-trace loop below ran 2.7 % more cycles for the same instructions on
 * AVX-512 (tet 3439 P3, GCC 14).
 *
 * @tparam NDIM           Spatial dimension: components of the normal.
 * @tparam APPEND         Add to the flux rather than overwrite it.
 * @tparam PARTITION      The block is a partition block: side 1 is the
 *                        neighbour's element-local trace data from the
 *                        receive buffers (@p in1, @p second1), at width 1,
 *                        and the flux is written to the local side only.
 * @tparam PointFlux      The point flux (see TraceFluxKernels.hpp).
 * @param  views          The block's two sides and global-trace offsets.
 * @param  pointFlux      The point flux.
 * @param  inPtr,in1      The state field, and side 1's (the same field
 *                        unless PARTITION).
 * @param  secondPtr,second1  The second input, likewise; unread by a point
 *                        flux without one.
 * @param  fluxPtr        The flux field.
 * @param  normPtr,penPtr The global-trace normals and penalty factors, as
 *                        the data warehouse holds them; the penalty factors
 *                        unread by a point flux without a penalty.
 * @param  streamID       Unused on the host spaces.
 */
template <unsigned int NDIM, bool APPEND, bool PARTITION, typename ExecSpace,
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
    constexpr unsigned allLanes = (1u << vec_width) - 1u;

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
        AlongTraceNormals<TData> normals;
        normals.nrmCS = glo.compSize[trace];
        // Unread by a point flux without a penalty.
        const TData *nrmTrace = normPtr + glo.gloTOffset[trace];
        const TData *penTrace =
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

            // The global-trace points of a trace are consecutive, so the
            // normals and penalty factors load as vectors.
            normals.nrm   = nrmTrace + p0;
            normals.pen   = penTrace + p0;
            normals.valid = valid;
            const unsigned mask =
                valid == vec_width ? allLanes : (1u << valid) - 1u;
            InteriorLanesFlux<NDIM, APPEND, PARTITION, vec_t, vec_width>(
                pointFlux, o0, o1, d0, d1, cs0, cs1, c0, c1, cd0, cd1, mask,
                normals, inPtr, in1, secondPtr, second1, fluxPtr);
        }
    }
}

} // namespace Nektar::SolverCore::detail
