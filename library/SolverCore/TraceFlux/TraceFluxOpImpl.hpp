///////////////////////////////////////////////////////////////////////////////
//
// File: TraceFluxOpImpl.hpp
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
// Description: Trace Flux routine to set up numbering for derived classes
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <tuple>

#include "LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp"
#include "LibUtilities/BasicUtils/Field/Block.hpp"
#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"
#include "MultiRegions/DataWarehouse/TraceDataWarehouse.hpp"
#include "SolverCore/TraceFlux/TraceFluxKernels.hpp"
#include "SolverCore/TraceFlux/TraceFluxOp.hpp"

#include "LibUtilities/Communication/EntityResolver.hpp"
#include "MultiRegions/AssemblyMap/AssemblyComm.h"
#include "MultiRegions/AssemblyMap/AssemblyMapDG.h"

#include <array>
#include <numeric>
#include <tuple>

namespace Nektar::SolverCore::detail
{

/**
 * @brief Trace numbering and gather/scatter machinery shared by all trace-flux
 * operators.
 *
 * This class does not evaluate a numerical flux and does not override any of
 * the `v_` methods of TraceFluxOp. What it provides is everything a concrete
 * trace-flux operator needs *around* the Riemann solve: the constructor builds
 * the trace connectivity once, and the protected helpers below move data
 * between the element-local trace layout and the packed "global trace" layout
 * the flux kernels consume. Concrete operators, for example
 * `ScalarTraceFluxOpImpl` and `AdvTraceFluxCFEOpImpl` in the solvers, derive
 * from this class and supply `v_Apply`.
 *
 * ### Fwd/Bwd naming
 *
 * Throughout this class `T0` denotes the forward (left) side of a trace and
 * `T1` the backward (right) side; #m_intT0 and #m_intT1 hold the two sides of
 * each interior trace, and #m_bndT0 the interior side of a Dirichlet
 * boundary trace, whose exterior data comes from `TraceFluxOp::m_BndCondOp`
 * instead.
 *
 * ### Blocking
 *
 * Traces are grouped into *blocks*, and the vectors below are indexed
 * `[block][trace-within-block]`. A block collects traces that share the same
 * quadrature point counts, point types and interpolation requirements on both
 * sides, so that one set of interpolation matrices and one set of loop bounds
 * serves the whole block. HaveSimilarTraceInfo() decides that grouping. The
 * per-block parameters that the helpers read (#m_npT0, #m_npT1, #m_npT,
 * #m_npTot, #m_T0Collocated, #m_T1Collocated) are *not* per-block storage:
 * they are scratch fields that SetInteriorParams() or SetBoundaryParams()
 * must be called to refresh before working on a given block.
 *
 * ### Interpolation
 *
 * Where the two sides of a trace carry different point distributions, that is
 * variable polynomial order, data is interpolated to a common trace before
 * the flux is evaluated and projected back afterwards. `interpFwd` carries
 * local-to-global matrices and `interpBwd` global-to-local; a null pointer in
 * either means that direction is collocated and needs only reorientation. One
 * side of every interior trace is required to be collocated with the global
 * trace; see the assertions in SetInteriorParams().
 *
 * @tparam ExecSpace Execution space the operator runs in. Selects the memory
 *                   space used for the interpolation matrices and workspace,
 *                   and the SIMD width used to pad trace blocks.
 * @tparam TData     Floating-point representation used by the field data.
 * @tparam TBase     Operator interface the implementation presents, so that a
 *                   solver's named operator class can sit between this helper
 *                   and TraceFluxOp; it derives from TraceFluxOp<TData> and
 *                   is constructed from (expansionList, components).
 *
 * @see TraceFluxOp for the public operator interface.
 * @see TraceFluxKernels.hpp for the reorientation and interpolation kernels
 *      the helpers below dispatch into.
 */
template <typename ExecSpace, typename TData,
          typename TBase = TraceFluxOp<TData>>
class TraceFluxOpImpl : public TBase
{
    using MemSpace = typename ExecSpace::memory_space;

protected:
    /**
     * @brief Round @p n up to a whole number of vector lanes.
     *
     * The AVX kernels reinterpret_cast the workspace sub-buffers and the
     * per-block trace arrays to tinysimd::simd<TData>, a type declaring
     * 32-byte alignment, so the compiler emits aligned vector moves against
     * them. The region base is over-aligned, but the running offsets are
     * element counts and need not be a whole number of lanes, so without this
     * a sub-buffer can start mid-vector and fault. Serial and Device have a
     * lane width of one and pad by nothing.
     */
    static constexpr size_t PadToVectorWidth(const size_t n)
    {
        constexpr size_t w = std::is_same_v<ExecSpace, NektarSpaces::AVX>
                                 ? tinysimd::simd<TData>::width
                                 : 1u;
        return ((n + w - 1) / w) * w;
    }

public:
    /**
     * @brief Build the trace connectivity for @p expansionList.
     *
     * All of the trace numbering work happens here, once, so that the flux
     * evaluation itself is a loop over precomputed offsets. In outline:
     *
     * 1. Walk every element trace. The first time a trace id is seen its
     *    details are parked in a local map; the second time, the two sides are
     *    paired up and appended to a block of #m_intT0 / #m_intT1. Periodic
     *    edges and faces are pre-mapped so that the two halves of a periodic
     *    pair are recognised as the same trace, with the orientation of the
     *    second side adjusted to match the first.
     * 2. Whatever is still unpaired after that walk is by definition a
     *    boundary trace, and is used to build #m_bndT0.
     * 3. Per-block trace sizes are padded up to the SIMD width for the AVX
     *    execution space.
     * 4. The trace normals are gathered into `TraceFluxOp::m_traceNormals`,
     *    and #m_intGloTraceOffset / #m_bndGloTraceOffset are rewritten from
     *    global trace offsets into block-component ordering to match.
     *
     * @param expansionList - Expansion list the operator acts on. Supplies the
     *                        trace map, periodic entities and trace normals.
     * @param components    - Names of the field components.
     */
    TraceFluxOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : TBase(std::move(expansionList), components)
    {
        unsigned int coordDim = expansionList->GetCoordim(0);

        // obtain list of connecting elements and traces
        auto trace       = expansionList->GetTrace();
        auto elmtToTrace = expansionList->GetTraceMap()->GetElmtToTrace();
        auto dim         = expansionList->GetExp(0)->GetShapeDimension();
        m_traceDim       = dim - 1;

        std::map<size_t, TraceInfo> TraceData;
        std::map<size_t, std::map<size_t, size_t>> PtsOffset;
        size_t compSizeSum = 0;

        std::vector<std::pair<TraceInfo, TraceInfo>> SaveTraceData;

        auto blocks = MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
            expansionList);
        auto traceblocks =
            MultiRegions::GetLocTraceBlockAttributes<TData, FieldState::Phys>(
                expansionList);

        // Set up a list map of periodic local ids
        std::map<size_t, std::pair<size_t, StdRegions::Orientation>>
            PeriodicGIDs;
        MultiRegions::PeriodicMap periodicVerts, periodicEdges, periodicFaces;
        expansionList->GetPeriodicEntities(periodicVerts, periodicEdges,
                                           periodicFaces);

        if (periodicVerts.size() || periodicEdges.size() ||
            periodicFaces.size())
        {
            // make a list of local to global edges
            std::map<size_t, size_t> Gid2Eid;
            for (unsigned t = 0; t < trace->GetNumElmts(); ++t)
            {
                Gid2Eid[trace->GetExp(t)->GetGeom()->GetGlobalID()] = t;
            }
            switch (dim)
            {
                case 1:
                    RegisterPeriodicPairs(periodicVerts, Gid2Eid, PeriodicGIDs);
                    break;
                case 2:
                    RegisterPeriodicPairs(periodicEdges, Gid2Eid, PeriodicGIDs);
                    break;
                case 3:
                    RegisterPeriodicPairs(periodicFaces, Gid2Eid, PeriodicGIDs);
                    break;
            }
        }

        unsigned eid = 0;
        for (unsigned blk = 0; blk < blocks.size(); ++blk)
        {
            auto elmt      = expansionList->GetExp(eid);
            auto ntrace    = elmt->GetNtraces();
            auto shapetype = elmt->DetShapeType();

            unsigned offset = 0;

            std::vector<unsigned> traceOffset(ntrace);
            unsigned tptsTot = 0;
            // calculate offset of element trace size for each trace in legacy
            // order
            for (unsigned d = 0; d < dim; ++d)
            {
                auto numt = LibUtilities::ShapeTypeNumTraceInDir[shapetype][d];
                for (unsigned i = 0; i < numt; ++i)
                {
                    auto nt =
                        LibUtilities::ShapeTypeTraceIDInDir[shapetype][d][i];
                    traceOffset[nt] = tptsTot;
                    unsigned npts   = 1;
                    for (unsigned d1 = 0; d1 < m_traceDim; ++d1)
                    {
                        auto bkey = elmt->GetTraceBasisKey(nt, d1);
                        npts *= bkey.GetNumPoints();
                    }
                    tptsTot += npts;
                }
            }

            for (unsigned e = 0; e < blocks[blk].GetNumElements(); ++e, ++eid)
            {
                elmt = expansionList->GetExp(eid);
                std::map<size_t, size_t> tPtsOffset;

                for (unsigned t = 0; t < ntrace; ++t)
                {
                    tPtsOffset[t] = offset + traceOffset[t];

                    auto tid = elmtToTrace[eid][t]->GetElmtId();
                    // check to see if a periodic edge/face exists and
                    // if the other trace has already been setup then
                    // reset tid to global id of periodic faces/edge

                    StdRegions::Orientation t1Orient = elmt->GetTraceOrient(t);
                    if (PeriodicGIDs.count(tid))
                    {
                        auto other = PeriodicGIDs[tid].first;
                        if (TraceData.count(other))
                        {
                            // The pair is about to share 'other' as its
                            // global trace, but t1Orient relates this
                            // element's local trace to its *own* global
                            // periodic trace. The two periodic traces need
                            // not be aligned with each other, and
                            // PeriodicGIDs[other].second is that relative
                            // orientation, so compose the two to land on
                            // the trace being adopted.
                            //
                            // It is inverted first. DisContField stores the
                            // periodic orientation as
                            // GetFaceOrientation(this trace, partner),
                            // relating the two global traces in the
                            // opposite sense to the one ComposeOrient()
                            // works in, so it has to be turned round before
                            // it can be composed with t1Orient.
                            t1Orient = ComposeOrient(
                                t1Orient,
                                InvertOrient(PeriodicGIDs[other].second));

                            tid = other;
                        }
                    }

                    // have existing data so can set up new values
                    if (TraceData.count(tid))
                    {
                        bool SaveTraceInfo = false;

                        // first entry
                        TraceInfo tinfo0 = TraceData[tid];

                        auto b = HaveSimilarTraceInfo(SaveTraceData, tinfo0,
                                                      elmt, t);

                        // need to initialise new block of traces
                        if (b == m_intT0.size())
                        {
                            TraceDetails intT0, intT1;

                            m_intT0.push_back(intT0);
                            m_intT1.push_back(intT1);

                            std::vector<size_t> offset, compsize;
                            m_intGloTraceOffset.push_back(offset);
                            m_intGloTraceCompSize.push_back(compsize);

                            SaveTraceInfo = true;
                        }

                        // set up pointer data offset for start of block
                        m_intT0[b].offset_st.push_back(tinfo0.compSizeSum);
                        m_intT1[b].offset_st.push_back(compSizeSum);

                        // set up pointer data offset within block
                        m_intT0[b].offset.push_back(
                            PtsOffset[tinfo0.eid][tinfo0.t]);
                        m_intT1[b].offset.push_back(tPtsOffset[t]);

                        // store component size for offseting data in blcok
                        m_intT0[b].compSize.push_back(tinfo0.compSize);
                        m_intT1[b].compSize.push_back(
                            traceblocks[blk].CompSize());

                        // set up Global Trace offset (for normals etc)
                        m_intGloTraceOffset[b].push_back(
                            trace->GetPhys_Offset(tid));

                        // set up tinfo1 using existing element/trace
                        TraceInfo tinfo1;
                        tinfo1.eid         = eid;
                        tinfo1.t           = t;
                        tinfo1.compSize    = traceblocks[blk].CompSize();
                        tinfo1.compSizeSum = compSizeSum;

                        for (unsigned d = 0; d < m_traceDim; ++d)
                        {
                            auto bkey       = elmt->GetTraceBasisKey(t, d);
                            tinfo1.n[d]     = bkey.GetNumPoints();
                            tinfo1.ptype[d] = bkey.GetPointsType();
                        }

                        // share connecting point information
                        tinfo0.nc = tinfo1.n;
                        tinfo1.nc = tinfo0.n;

                        // set up orientation and interpolation as required
                        if (m_traceDim)
                        {
                            auto telmt = elmtToTrace[tinfo0.eid][tinfo0.t];

                            // set up tinfo for "global" trace to test
                            // against
                            TraceInfo tinfoglo;
                            for (unsigned d = 0; d < m_traceDim; ++d)
                            {
                                tinfoglo.n[d]     = telmt->GetNumPoints(d);
                                tinfoglo.ptype[d] = telmt->GetPointsType(d);
                            }
                            tinfoglo.orient =
                                (m_traceDim == 1)
                                    ? StdRegions::eForwards
                                    : StdRegions::eDir1FwdDir1_Dir2FwdDir2;

                            tinfo1.orient = t1Orient;

                            // populate information comparing tinfo0 with
                            // tinfoglo
                            PopulateTraceVecs(telmt, tinfo0, tinfoglo,
                                              m_intT0[b]);

                            // populate information comparing tinfo1 with
                            // tinfoglo
                            PopulateTraceVecs(telmt, tinfo1, tinfoglo,
                                              m_intT1[b]);
                        }
                        else
                        {
                            // segment case with point trace
                            m_intT0[0].npts.push_back(1);
                            m_intT1[0].npts.push_back(1);
                        }

                        // save copy of this Trace Info for later comparisons
                        if (SaveTraceInfo)
                        {
                            SaveTraceData.push_back(
                                std::make_pair(tinfo0, tinfo1));
                        }

                        // Remove details so what remains is boundary info
                        TraceData.erase(tid);
                    }
                    else // fill in data for later use
                    {
                        TraceInfo tinfo;
                        tinfo.eid         = eid;
                        tinfo.t           = t;
                        tinfo.compSize    = traceblocks[blk].CompSize();
                        tinfo.compSizeSum = compSizeSum;

                        if (m_traceDim) // only require for traceDim >= 1
                        {
                            // store entry
                            for (unsigned d = 0; d < m_traceDim; ++d)
                            {
                                auto bkey      = elmt->GetTraceBasisKey(t, d);
                                tinfo.n[d]     = bkey.GetNumPoints();
                                tinfo.ptype[d] = bkey.GetPointsType();
                            }
                            tinfo.orient = elmt->GetTraceOrient(t);
                        }

                        TraceData[tid] = tinfo;
                    }
                }
                PtsOffset[eid] = tPtsOffset;
                offset += tptsTot; // skip forward this element block
            }
            // generate a summation of composite sizes
            compSizeSum += traceblocks[blk].CompSize();
        }

        // set up padded number of traces time number of points
        for (unsigned b = 0; b < m_intT0.size(); ++b)
        {
            switch (m_traceDim)
            {
                case 0:
                    SetInteriorParams<0>(b);
                    break;
                case 1:
                    SetInteriorParams<1>(b);
                    break;
                case 2:
                    SetInteriorParams<2>(b);
                    break;
            }

            m_intT0[b].m_nTraceXnPtsPad =
                PadToVectorWidth(m_npTot * m_intT0[b].offset.size());

            AccumulateWorkSpaceBound(m_intT0[b].offset.size());
        }

        // Everything still in TraceData is unpaired, and is either a physical
        // boundary of the mesh or a cut through it made by the partitioner.
        // Claim the latter now, so that what reaches the boundary pass below
        // is what that pass has always been given.
        SetUpParallelTraces(expansionList, elmtToTrace, TraceData, PtsOffset);

        // clear trace info saving so can restart process with BCs
        SaveTraceData.clear();

        // Visit the boundary traces in the order BndCondPhysOp stores their
        // values, rather than the ascending global trace id that iterating
        // TraceData gives. Block membership is decided by
        // HaveSimilarBndTraceInfo() and does not depend on visit order, so the
        // shape and quadrature grouping the Riemann solve relies on is
        // untouched; what changes is that within a block the reads in
        // GetDirBCTrace() become an ordered walk of the boundary storage
        // instead of a scatter across it. Every per-trace array below is filled
        // by appending inside the loop, so all of them stay index aligned.
        std::vector<std::pair<size_t, TraceInfo>> bndTraces;
        if (this->m_BndCondOp)
        {
            // Resolve each trace's storage location once - the comparator must
            // not pay a virtual call and two map lookups per comparison.
            std::vector<std::tuple<unsigned, size_t, size_t, TraceInfo>> keyed;
            keyed.reserve(TraceData.size());

            for (const auto &tinfo : TraceData)
            {
                const auto &tinfo0 = tinfo.second;
                auto telmt         = elmtToTrace[tinfo0.eid][tinfo0.t];

                unsigned blk      = 0;
                size_t offset     = 0;
                size_t compOffset = 0;
                this->m_BndCondOp->GetTraceLocation(
                    telmt->GetGeom()->GetGlobalID(), blk, offset, compOffset);

                // global trace id last, so the order stays deterministic if two
                // traces ever resolve to the same place
                keyed.emplace_back(blk, offset, tinfo.first, tinfo0);
            }

            std::sort(keyed.begin(), keyed.end(),
                      [](const auto &a, const auto &b) {
                          return std::tie(std::get<0>(a), std::get<1>(a),
                                          std::get<2>(a)) <
                                 std::tie(std::get<0>(b), std::get<1>(b),
                                          std::get<2>(b));
                      });

            bndTraces.reserve(keyed.size());
            for (auto &k : keyed)
            {
                bndTraces.emplace_back(std::get<2>(k), std::get<3>(k));
            }
        }
        else
        {
            bndTraces.assign(TraceData.begin(), TraceData.end());
        }

        // set up Dir Bnd condition info
        for (auto &tinfo : bndTraces)
        {
            bool SaveTraceInfo = false;

            // first entry
            TraceInfo tinfo0 = tinfo.second;

            auto telmt = elmtToTrace[tinfo0.eid][tinfo0.t];

            auto b = HaveSimilarBndTraceInfo(SaveTraceData, tinfo0, telmt);

            // need to initialise new block of traces
            if (b == m_bndT0.size())
            {
                TraceDetails dirBndT0;

                m_bndT0.push_back(dirBndT0);

                std::vector<size_t> traceId, offset, compsize;
                m_bndTraceIdT1.push_back(traceId);
                m_bndGloTraceOffset.push_back(offset);
                m_bndGloTraceCompSize.push_back(compsize);

                std::vector<unsigned> T1npts;
                m_bndNptsT1.push_back(T1npts);

                SaveTraceInfo = true;
            }

            // store global trace id of T1 for boundary data extraction
            m_bndTraceIdT1[b].push_back(telmt->GetGeom()->GetGlobalID());

            // Store component offset -related boundary face will come from
            // DirBCOp
            m_bndT0[b].compSize.push_back(tinfo0.compSize);

            // set up pointer data offset - will use DirBCOp to get other
            // data
            m_bndT0[b].offset_st.push_back(tinfo0.compSizeSum);
            m_bndT0[b].offset.push_back(PtsOffset[tinfo0.eid][tinfo0.t]);

            // set up normal offset
            m_bndGloTraceOffset[b].push_back(
                trace->GetPhys_Offset(tinfo.first));

            if (m_traceDim) // orientation and interpolation
            {
                // set up tinfo for "global" trace to test against
                TraceInfo tinfoglo;
                for (unsigned d = 0; d < m_traceDim; ++d)
                {
                    auto npts         = telmt->GetNumPoints(d);
                    tinfoglo.n[d]     = npts;
                    tinfoglo.ptype[d] = telmt->GetPointsType(d);

                    // store trace/bnd data, once per block
                    if (SaveTraceInfo)
                    {
                        m_bndNptsT1[b].push_back(npts);
                    }

                    // setup tinfo0 with neighbouring pts
                    tinfo0.nc[d] = npts;
                }

                // Asssuming trace is co-aligned with global trace
                tinfoglo.orient = (m_traceDim == 1)
                                      ? StdRegions::eForwards
                                      : StdRegions::eDir1FwdDir1_Dir2FwdDir2;

                PopulateTraceVecs(telmt, tinfo0, tinfoglo, m_bndT0[b]);

                // save copy of this Trace Info for later comparisons
                if (SaveTraceInfo)
                {
                    // since global trace is same as boundary trace user here
                    SaveTraceData.push_back(std::make_pair(tinfo0, tinfoglo));
                }
            }
        }

        // set up padded number of traces time number of points
        for (unsigned b = 0; b < m_bndT0.size(); ++b)
        {
            switch (m_traceDim)
            {
                case 0:
                    SetBoundaryParams<0>(b);
                    break;
                case 1:
                    SetBoundaryParams<1>(b);
                    break;
                case 2:
                    SetBoundaryParams<2>(b);
                    break;
            }

            m_bndT0[b].m_nTraceXnPtsPad =
                PadToVectorWidth(m_npTot * m_bndT0[b].offset.size());

            AccumulateWorkSpaceBound(m_bndT0[b].offset.size());
        }

        // Create global blocks.
        auto glo_blocks_trace =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(trace, 1);

        // Create fields.
        unsigned int numHomoModes = 1;
        this->m_traceNormals = LibUtilities::Field<TData, FieldState::Phys>(
            "traceNormals", glo_blocks_trace, coordDim, numHomoModes);

        // Fill them from the mesh. Nothing can have overridden the normals
        // yet - SetTraceNormals() needs an operator to call it on - so this
        // always loads here; the guard inside matters only if the default is
        // ever reloaded later.
        LoadDefaultTraceNormals(glo_blocks_trace, trace->GetTotPoints(),
                                coordDim);

        // set up compsize for each trace data and adjust m_gloTOffset
        // to be in block-component  order
        std::map<size_t, unsigned> pt2blk;
        std::map<size_t, size_t> pt2blkpt;
        std::map<size_t, size_t> pt2blkptScalar;
        unsigned blkOffset       = 0;
        unsigned blkOffsetScalar = 0;
        unsigned blkid           = 0;
        eid                      = 0;
        for (auto blk : glo_blocks_trace)
        {
            unsigned offset       = blkOffset;
            unsigned offsetScalar = blkOffsetScalar;
            for (size_t i = 0; i < blk.GetNumElements(); ++i, ++eid)
            {
                // get offset of this trace expansion  in global trace which is
                // held below.
                auto cnt = trace->GetPhys_Offset(eid);
                // Store the block id for this offset
                pt2blk[cnt] = blkid;
                // Store the offset in global block trace format
                pt2blkpt[cnt]       = offset;
                pt2blkptScalar[cnt] = offsetScalar;
                // increment the offset
                const auto nPts = trace->GetExp(eid)->GetTotPoints();
                offset += nPts;
                offsetScalar += nPts;
            }
            // Both arrays span every block, so both offsets accumulate over
            // the blocks before this one - but at different rates. The normals
            // carry coordDim components per block; single-component trace data
            // - the interior penalty factor, the trace length - carries one.
            // Indexing the latter with the former's offset overshoots by a
            // factor of coordDim, which stays in range for the first block and
            // reads past it for the rest.
            blkOffset += coordDim * blk.CompSize();
            blkOffsetScalar += blk.CompSize();
            ++blkid;
        }

        m_gloTraceScalarSize = 0;
        for (const auto &gblk : glo_blocks_trace)
        {
            m_gloTraceScalarSize += gblk.CompSize();
        }

        RemapGloTraceOffsets(glo_blocks_trace, pt2blk, pt2blkpt, pt2blkptScalar,
                             m_intGloTraceOffset, m_intGloTraceScalarOffset,
                             m_intGloTraceCompSize);

        RemapGloTraceOffsets(glo_blocks_trace, pt2blk, pt2blkpt, pt2blkptScalar,
                             m_bndGloTraceOffset, m_bndGloTraceScalarOffset,
                             m_bndGloTraceCompSize);

        RemapGloTraceOffsets(glo_blocks_trace, pt2blk, pt2blkpt, pt2blkptScalar,
                             m_parGloTraceOffset, m_parGloTraceScalarOffset,
                             m_parGloTraceCompSize);

        // The offsets above are final now, so mirror them into the execution
        // space for the derived operators' gathers.
        PackGloTraceOffsets(m_intGloTraceOffset, m_intGloTraceScalarOffset,
                            m_intGloTraceCompSize, m_intGloTraceDev);
        PackGloTraceOffsets(m_bndGloTraceOffset, m_bndGloTraceScalarOffset,
                            m_bndGloTraceCompSize, m_bndGloTraceDev);
        PackGloTraceOffsets(m_parGloTraceOffset, m_parGloTraceScalarOffset,
                            m_parGloTraceCompSize, m_parGloTraceDev);

        // Mirror the per-trace interior metadata into the execution space so
        // the gather kernels can index it. Last, once the block details above
        // are final.
        m_intT0Dev.reserve(m_intT0.size());
        m_intT1Dev.reserve(m_intT1.size());
        for (unsigned b = 0; b < m_intT0.size(); ++b)
        {
            m_intT0Dev.emplace_back();
            m_intT1Dev.emplace_back();

            PackTraceBlock(m_intT0[b], m_intT0Dev[b]);
            PackTraceBlock(m_intT1[b], m_intT1Dev[b]);
        }

        // Same for the interior side of the Dirichlet boundary blocks, read
        // by GetLocalBndTrace().
        m_bndT0Dev.reserve(m_bndT0.size());
        for (unsigned b = 0; b < m_bndT0.size(); ++b)
        {
            m_bndT0Dev.emplace_back();
            PackTraceBlock(m_bndT0[b], m_bndT0Dev[b]);
        }

        // Both sides of the parallel blocks. The remote side addresses a
        // receive buffer in points rather than values, so it does not depend
        // on what is later exchanged through it.
        m_parT0Dev.reserve(m_parT0.size());
        m_parT1Dev.resize(m_parT0.size());
        for (unsigned b = 0; b < m_parT0.size(); ++b)
        {
            m_parT0Dev.emplace_back();
            PackTraceBlock(m_parT0[b], m_parT0Dev[b]);
            PackTraceBlock(m_parT1[b], m_parT1Dev[b]);
        }

        PackParallelSendLocations();

        PackBndBCLocations();
    }

    /**
     * @brief Record which local traces are periodic partners of which.
     *
     * @param periodic     Periodic map for the entity dimension in play -
     *                     vertices, edges or faces.
     * @param Gid2Eid      Global mesh id to local trace expansion id, holding
     *                     only the traces this rank owns.
     * @param PeriodicGIDs Filled with local trace id to (partner local trace
     *                     id, relative orientation).
     *
     * A partition may cut a periodic pair, leaving one half on another rank.
     * Such a pair has no local partner to name and so goes unrecorded here;
     * the trace then survives the pairing loop unpaired, and
     * SetUpParallelTraces() picks it up along with every other trace whose
     * neighbour is remote. Left unguarded, the lookup that finds no partner
     * would insert one: `Gid2Eid[remote id]` is a default-constructed zero,
     * which names local trace 0 - a real trace, elsewhere in the mesh, that
     * would then be paired against silently.
     */
    void RegisterPeriodicPairs(
        const MultiRegions::PeriodicMap &periodic,
        const std::map<size_t, size_t> &Gid2Eid,
        std::map<size_t, std::pair<size_t, StdRegions::Orientation>>
            &PeriodicGIDs)
    {
        for (const auto &p : periodic)
        {
            auto self = Gid2Eid.find(p.first);
            if (self == Gid2Eid.end())
            {
                continue;
            }

            auto partner = Gid2Eid.find(p.second[0].id);
            if (partner == Gid2Eid.end())
            {
                // Partner is on another rank; remember the pair by the ids
                // both ranks can name - and the relative orientation of the
                // two trace geometries, which the exchange must fold into the
                // payload exactly as the local pairing composes it above.
                m_periodicRemote[self->second] = {
                    static_cast<size_t>(p.first),
                    static_cast<size_t>(p.second[0].id), p.second[0].orient};
                continue;
            }

            PeriodicGIDs[self->second] =
                std::make_pair(partner->second, p.second[0].orient);
        }
    }

    /**
     * @brief Rewrite one family of global trace offsets into block-component
     * order, and record the block sizes that go with them.
     *
     * The constructor collects offsets as `ExpList::GetPhys_Offset()` reports
     * them, a plain running count over the trace expansions. The fields they
     * address - the normals, and the single-component trace data - are held
     * per block with a padded component stride, so every offset has to be
     * re-expressed in that layout before it can be used, and two of them are
     * needed: one at coordDim components per block and one at a single
     * component. Getting those two confused is what the separate
     * @p scalarOffset output exists to prevent; see #m_intGloTraceScalarOffset.
     *
     * @param blocks         - Block attributes of the global trace.
     * @param pt2blk         - Trace offset to the block containing it.
     * @param pt2blkpt       - Trace offset to its position in block order.
     * @param pt2blkptScalar - The same, for single-component trace data.
     * @param offset         - Offsets to rewrite, in place.
     * @param scalarOffset   - Filled with the single-component offsets.
     * @param compSize       - Filled with each trace's block component size.
     */
    void RemapGloTraceOffsets(
        const std::vector<LibUtilities::BlockAttributes<FieldState::Phys>>
            &blocks,
        std::map<size_t, unsigned> &pt2blk, std::map<size_t, size_t> &pt2blkpt,
        std::map<size_t, size_t> &pt2blkptScalar,
        std::vector<std::vector<size_t>> &offset,
        std::vector<std::vector<size_t>> &scalarOffset,
        std::vector<std::vector<size_t>> &compSize)
    {
        scalarOffset.resize(offset.size());
        compSize.resize(offset.size());

        for (unsigned b = 0; b < offset.size(); ++b)
        {
            const auto numblock = offset[b].size();
            scalarOffset[b].reserve(numblock);
            compSize[b].reserve(numblock);

            for (unsigned t = 0; t < numblock; ++t)
            {
                auto off = offset[b][t];
                ASSERTL1(pt2blk.count(off), "Did not find offset in map");
                auto blk = pt2blk[off];
                // reset normal offset to respect block ordering
                scalarOffset[b].push_back(pt2blkptScalar[off]);
                offset[b][t] = pt2blkpt[off];
                // store composite size for this offset
                compSize[b].push_back(blocks[blk].CompSize());
            }
        }
    }

    /**
     * @brief Resolve where each Dirichlet trace's boundary data lives.
     *
     * TraceFluxOp::m_BndCondOp maps a global trace id to a block, an offset and
     * a component stride through two std::maps behind a virtual call. That
     * mapping is fixed once the operator is built, so resolve it here and keep
     * only the integers, device-side. The base pointers are deliberately not
     * cached: they are refetched per apply in GetDirBCTrace(), so that the
     * memory manager still sees every access and can synchronise data that
     * changes with time.
     */
    NEK_FORCE_INLINE void PackBndBCLocations()
    {
        if (!this->m_BndCondOp)
        {
            return;
        }

        m_bndCondLoc.clear();
        m_bndCondLoc.reserve(m_bndTraceIdT1.size());

        for (unsigned b = 0; b < m_bndTraceIdT1.size(); ++b)
        {
            const auto numTrace = m_bndTraceIdT1[b].size();

            std::vector<unsigned> blk(numTrace);
            std::vector<size_t> offset(numTrace), compOffset(numTrace);

            for (size_t t = 0; t < numTrace; ++t)
            {
                this->m_BndCondOp->GetTraceLocation(
                    m_bndTraceIdT1[b][t], blk[t], offset[t], compOffset[t]);
            }

            m_bndCondLoc.emplace_back();
            auto &d = m_bndCondLoc.back();
            d.blk   = LibUtilities::MemoryRegion<unsigned>::template FromVector<
                  MemSpace>(blk);
            d.offset = LibUtilities::MemoryRegion<size_t>::template FromVector<
                MemSpace>(offset);
            d.compOffset = LibUtilities::MemoryRegion<
                size_t>::template FromVector<MemSpace>(compOffset);

            // Does the whole block come from one storage block at one stride?
            // The per-trace arrays are built either way: they cost little and
            // keeping them means a later consumer of m_bndCondLoc cannot trip
            // over an array that this block happened not to populate.
            d.singleBlock =
                numTrace > 0 &&
                std::equal(blk.begin() + 1, blk.end(), blk.begin()) &&
                std::equal(compOffset.begin() + 1, compOffset.end(),
                           compOffset.begin());
            if (d.singleBlock)
            {
                d.singleBlockId = blk[0];
                d.compStride    = compOffset[0];
            }
        }

        m_bndCondBase = LibUtilities::MemoryRegion<const TData *>(
            this->m_BndCondOp->GetNumBlocks());
        m_bndCondBaseW = LibUtilities::MemoryRegion<TData *>(
            this->m_BndCondOp->GetNumBlocks());

        // Condition type of every (storage block, component) pair, so that the
        // gather kernels can tell which components they own. Dirichlet
        // components take an exterior state, Neumann ones an exterior gradient,
        // and anything else is left with no jump across the trace. The table is
        // per storage block rather than per trace, so it stays small enough to
        // sit in cache and is indexed through the blk array already on device.
        const unsigned numStore = this->m_BndCondOp->GetNumBlocks();
        m_numBCComp             = this->m_BndCondOp->GetNumComponents();

        std::vector<unsigned> bcType(numStore * m_numBCComp,
                                     SpatialDomains::eNotDefined);
        for (unsigned k = 0; k < numStore; ++k)
        {
            for (unsigned c = 0; c < m_numBCComp; ++c)
            {
                bcType[k * m_numBCComp + c] = static_cast<unsigned>(
                    this->m_BndCondOp->GetBndCondType(k, c));
            }
        }
        m_bcTypeMask =
            LibUtilities::MemoryRegion<unsigned>::template FromVector<MemSpace>(
                bcType);
    }

    /**
     * @brief Allocate and carve up the scratch workspace.
     *
     * Sizes a single MemoryRegion from the largest padded trace block found in
     * any of #m_intT0, #m_bndT0 or #m_parT0, plus #m_maxInterpWsp for the
     * interpolation scratch, then hands out the sub-buffers #m_wsp, #m_loc,
     * #m_gloT0, #m_gloT1, #m_flux and #m_norms as pointers into it.
     * The allocation is skipped when the component count is unchanged since
     * the last call, so this is cheap to call on every apply.
     *
     * @param numComp - Number of components carried per trace point.
     *
     * @note The workspace is allocated write-only in the operator's memory
     *       space, on the assumption that it is never read back on the host.
     */
    // set up wspaces
    void SetWorkSpace(unsigned const numComp)
    {
        if (m_numComp != numComp)
        {
            size_t maxNTraceNPts = 0;
            for (unsigned b = 0; b < m_intT0.size(); ++b)
            {
                maxNTraceNPts =
                    std::max(maxNTraceNPts, m_intT0[b].m_nTraceXnPtsPad);
            }
            for (unsigned b = 0; b < m_bndT0.size(); ++b)
            {
                maxNTraceNPts =
                    std::max(maxNTraceNPts, m_bndT0[b].m_nTraceXnPtsPad);
            }
            for (unsigned b = 0; b < m_parT0.size(); ++b)
            {
                maxNTraceNPts =
                    std::max(maxNTraceNPts, m_parT0[b].m_nTraceXnPtsPad);
            }

            // padded workspace size
            auto maxTraceSizeNumComp = maxNTraceNPts * numComp;
            auto maxTraceSizeDim = maxNTraceNPts * numComp * (m_traceDim + 1);

            // Scratch for the gather and scatter kernels, which run one thread
            // per (trace, component) pair and give each its own slice; see
            // AccumulateWorkSpaceBound() for the bound. m_wsp is only touched
            // by the 2D interpolation, so below two trace dimensions it takes
            // no space of its own and aliases m_loc, as it always has.
            auto perThreadSize = m_maxTraceXnPts * numComp;
            auto interpWspSize = (m_traceDim == 2) ? perThreadSize : 0;

            // Every gather and scatter is now a block launcher, so this is the
            // only shape m_loc is asked for.
            auto locSize = perThreadSize;

            // Every sub-buffer below is handed to the AVX kernels, which
            // reinterpret_cast it to tinysimd::simd<TData> - a type declaring
            // 32-byte alignment, so the compiler emits aligned vector moves
            // (vmovapd) against it. The region base is over-aligned, but the
            // running offsets are element counts and need not be a whole
            // number of lanes, so a sub-buffer can start mid-vector and fault.
            // Round each offset up to a full vector. Serial pads by 1, so
            // this costs nothing when no vectorisation is in play.
            const size_t interpWspPad = PadToVectorWidth(interpWspSize);
            const size_t locPad       = PadToVectorWidth(locSize);
            const size_t numCompPad   = PadToVectorWidth(maxTraceSizeNumComp);
            const size_t dimPad       = PadToVectorWidth(maxTraceSizeDim);

            auto wspsize = interpWspPad + locPad + 3 * numCompPad + dimPad;

            m_workSpace = LibUtilities::MemoryRegion<TData>(wspsize);
            m_workSpace.template Initialize<MemSpace>(0.0);

            // believe we only need this locally on device or host so
            // making it WriteOnly space to avoid unnecessary copies?
            m_wsp   = m_workSpace.template GetPtr<MemSpace, WriteOnly>();
            m_loc   = m_wsp + interpWspPad;
            m_gloT0 = m_loc + locPad;
            m_gloT1 = m_gloT0 + numCompPad;
            m_flux  = m_gloT1 + numCompPad;
            m_norms = m_flux + numCompPad;

            m_numComp = numComp;
        }
    }

protected:
    /**
     * @brief Load the mesh's trace normals into TraceFluxOp::m_traceNormals.
     *
     * The values come from the data warehouse under
     * MultiRegions::GlobalTraceNormalKey, one entry per trace block, held
     * component-major with a stride of the block's padded CompSize().
     * Field::CopyArray() wants the whole trace component-major and unpadded
     * instead, so the two layouts are bridged here: for each block, each
     * component's real points are placed at that block's running offset within
     * its component.
     *
     * Does nothing once SetTraceNormals() has supplied normals from outside;
     * an override always wins over the mesh.
     *
     * @param blocks      - Block attributes of the global trace.
     * @param traceTotPts - Total real points on the trace.
     * @param coordDim    - Number of coordinate components.
     */
    void LoadDefaultTraceNormals(
        const std::vector<LibUtilities::BlockAttributes<FieldState::Phys>>
            &blocks,
        const size_t traceTotPts, const unsigned int coordDim)
    {
        if (this->m_traceNormalsOverridden)
        {
            return;
        }

        Array<OneD, double> normals(coordDim * traceTotPts, 0.0);

        size_t traceOffset = 0;
        for (unsigned int blk = 0; blk < blocks.size(); ++blk)
        {
            const auto *src =
                this->m_dataWarehouse
                    ->template GetData<NektarSpaces::HostSpace>(
                        MultiRegions::GlobalTraceNormalKey<double>(blk));

            const auto compSize = blocks[blk].CompSize();
            const size_t nRealPts =
                blocks[blk].GetNumElements() * blocks[blk].GetNumData();

            for (unsigned int d = 0; d < coordDim; ++d)
            {
                for (size_t i = 0; i < nRealPts; ++i)
                {
                    normals[d * traceTotPts + traceOffset + i] =
                        src[d * compSize + i];
                }
            }

            traceOffset += nRealPts;
        }

        this->m_traceNormals.template CopyArray<MemSpace>(normals);

        // Harmless from the constructor, where no derived override is
        // reachable yet and any normals-derived cache still starts stale.
        this->v_OnTraceNormalsChanged();
    }

    /**
     * @brief One side of every trace in a single block.
     *
     * Built by the constructor and read by the gather/scatter helpers. Except
     * for @p npts, @p interpFwd, @p interpBwd and @p m_nTraceXnPtsPad, which
     * hold one entry per trace *dimension* and describe the block as a whole,
     * every vector holds one entry per trace in the block, in the order the
     * traces were appended.
     */
    // collect related trace data together for convenience
    struct TraceDetails
    {
        /// Position of trace data in trace field, start of block
        std::vector<size_t> offset_st;
        /// Position of trace data in trace field within block
        std::vector<size_t> offset;
        // composite size of data
        std::vector<size_t> compSize;
        /// Num points on trace - only require on set
        std::vector<unsigned> npts;
        /// interpolation basis to global trace
        /// or null if collocated - one set per block
        std::vector<const TData *> interpFwd;
        std::vector<const TData *> interpBwd;
        /// orientation of trace 0 to global trace - one set per block
        std::vector<StdRegions::Orientation> orient;
        /// Points per trace times number of traces in this block, rounded up
        /// to the execution space's vector width. Used as the component stride
        /// in the packed global-trace workspace.
        size_t m_nTraceXnPtsPad;
    };

    /**
     * @brief Execution-space copies of one TraceDetails' per-trace vectors.
     *
     * The gather kernels index these by trace, so they cannot be the host
     * `std::vector`s of TraceDetails. Built once by PackTraceBlock() and read
     * through the TraceBlockView that GetTraceBlockView()
     * hands to a kernel. The remaining TraceDetails fields are per-block rather
     * than per-trace and are passed as kernel arguments instead.
     */
    struct TraceDeviceData
    {
        LibUtilities::MemoryRegion<size_t> offset_st;
        LibUtilities::MemoryRegion<size_t> offset;
        LibUtilities::MemoryRegion<size_t> compSize;
        /// Empty when #m_traceDim is 0: point traces carry no orientation.
        LibUtilities::MemoryRegion<int> orient;
        size_t numTrace = 0;
    };

    /**
     * @brief Execution-space copies of one block's offsets into the global
     * trace.
     *
     * TraceDeviceData mirrors the element-local side of a block; this mirrors
     * the other end, where the mesh-wide global-trace fields live - the trace
     * normals, the advection velocity, the interior-penalty factor. The
     * derived operators gather those into their block-local workspaces, and a
     * kernel doing so cannot read the host `std::vector<std::vector<size_t>>`
     * the offsets are held in, so each block's slice is copied across.
     *
     * Filled once, by PackGloTraceOffsets() at the end of the constructor: the
     * offsets are rewritten into block-component ordering by
     * RemapGloTraceOffsets() and not touched again.
     *
     * @see GloTraceOffsetView, the read-only face of this.
     */
    struct GloTraceOffsetDeviceData
    {
        LibUtilities::MemoryRegion<size_t> gloTOffset;
        LibUtilities::MemoryRegion<size_t> scalarOffset;
        LibUtilities::MemoryRegion<size_t> compSize;
    };

    /**
     * @brief Compose two trace orientations.
     *
     * Returns the orientation obtained by applying @p first and then
     * @p second. Needed for periodic traces, where a local trace has to be
     * related to the global trace of its *partner* rather than its own.
     *
     * The composition is carried out in the sense of the index arithmetic
     * in ReOrientFaceKernel(): that routine gathers over its output, so
     * with `Forwards` set it computes, for a point of the global trace,
     * which local point feeds it. Data still travels from local to global
     * there; it is only the arithmetic that runs the other way. Combining
     * orientations requires picking one of the two senses and being
     * consistent, and this is the one that matches the kernels.
     *
     * The eight quadrilateral orientations are the symmetries of a square,
     * and the enum lays them out as a three-bit code counting from
     * eDir1FwdDir1_Dir2FwdDir2: bit 0 reverses trace direction 1, bit 1
     * reverses direction 0, and bit 2 transposes the two. Composition is
     * then a small amount of bit arithmetic rather than an enumeration of
     * all 64 pairs. Writing an orientation as the map
     *
     *     transposed == 0:  p = rev0 ? 1-a : a,   q = rev1 ? 1-b : b
     *     transposed == 1:  p = rev0 ? 1-b : b,   q = rev1 ? 1-a : a
     *
     * with @c (a,b) the global and @c (p,q) the local normalised
     * coordinates, substituting one into the other gives the four cases
     * below. The reversal bits combine by exclusive-or, and which of them
     * applies to which direction swaps whenever a transpose is involved.
     *
     * Edges carry the separate eForwards/eBackwards pair, where composition
     * is just whether an odd number of reversals is applied. Point traces
     * have no orientation and are returned unchanged.
     *
     * @param second - Orientation applied second.
     * @param first  - Orientation applied first.
     */
    /// Inverse of a trace orientation. Every orientation but the two 90
    /// degree rotations (eDir1FwdDir2_Dir2BwdDir1 and
    /// eDir1BwdDir2_Dir2FwdDir1, which invert to each other) is its own
    /// inverse.
    NEK_FORCE_INLINE static StdRegions::Orientation InvertOrient(
        const StdRegions::Orientation o)
    {
        if (o == StdRegions::eDir1FwdDir2_Dir2BwdDir1)
        {
            return StdRegions::eDir1BwdDir2_Dir2FwdDir1;
        }
        if (o == StdRegions::eDir1BwdDir2_Dir2FwdDir1)
        {
            return StdRegions::eDir1FwdDir2_Dir2BwdDir1;
        }
        return o;
    }

    NEK_FORCE_INLINE static StdRegions::Orientation ComposeOrient(
        const StdRegions::Orientation second,
        const StdRegions::Orientation first)
    {
        // Edge traces
        if (first == StdRegions::eForwards || first == StdRegions::eBackwards)
        {
            const bool flip = (first == StdRegions::eBackwards) ^
                              (second == StdRegions::eBackwards);
            return flip ? StdRegions::eBackwards : StdRegions::eForwards;
        }

        // Anything that is not one of the eight face orientations - point
        // traces in particular - has nothing to compose.
        if (first < StdRegions::eDir1FwdDir1_Dir2FwdDir2 ||
            second < StdRegions::eDir1FwdDir1_Dir2FwdDir2)
        {
            return second;
        }

        const unsigned k1 = first - StdRegions::eDir1FwdDir1_Dir2FwdDir2;
        const unsigned k2 = second - StdRegions::eDir1FwdDir1_Dir2FwdDir2;

        const unsigned t1 = (k1 >> 2) & 1, a0 = (k1 >> 1) & 1, a1 = k1 & 1;
        const unsigned t2 = (k2 >> 2) & 1, b0 = (k2 >> 1) & 1, b1 = k2 & 1;

        unsigned t, r0, r1;
        if (!t1 && !t2)
        {
            t = 0, r0 = a0 ^ b0, r1 = a1 ^ b1;
        }
        else if (!t1 && t2)
        {
            t = 1, r0 = a1 ^ b0, r1 = a0 ^ b1;
        }
        else if (t1 && !t2)
        {
            t = 1, r0 = a0 ^ b0, r1 = a1 ^ b1;
        }
        else
        {
            t = 0, r0 = a1 ^ b0, r1 = a0 ^ b1;
        }

        return StdRegions::Orientation(StdRegions::eDir1FwdDir1_Dir2FwdDir2 +
                                       ((t << 2) | (r0 << 1) | r1));
    }

    /// Largest value #m_traceDim can take, plus one: a face of a 3D element
    /// spans two directions. Sizes the fixed-length per-direction arrays of
    /// TraceInfo.
    static constexpr unsigned s_maxTraceDim = 3;

    /// Dimension of a trace: element dimension minus one, so 0 for a segment
    /// mesh, 1 for edges and 2 for faces. Selects the TRACEDIM branch used
    /// throughout the helpers below.
    unsigned m_traceDim;
    // Left/Fwd interior Trace details
    std::vector<TraceDetails> m_intT0;
    // Right/Bwd interior Trace details
    std::vector<TraceDetails> m_intT1;
    // Execution-space mirrors of the per-trace fields of the two above, one
    // entry per block, read by GetInteriorTraces()
    std::vector<TraceDeviceData> m_intT0Dev;
    std::vector<TraceDeviceData> m_intT1Dev;
    // offset of normals from global trace -
    std::vector<std::vector<size_t>> m_intGloTraceOffset;
    /// Offset of single-component trace data - the interior penalty factor,
    /// the trace length. Like #m_intGloTraceOffset it spans every block, but
    /// with one component per block rather than coordDim, so the two are not
    /// interchangeable.
    std::vector<std::vector<size_t>> m_intGloTraceScalarOffset;
    // Global Trace Component size  (for Normals etc.)
    std::vector<std::vector<size_t>> m_intGloTraceCompSize;
    /// Execution-space mirror of the three above, one entry per block, read by
    /// the derived operators' normals and penalty-factor gathers.
    std::vector<GloTraceOffsetDeviceData> m_intGloTraceDev;

    /**
     * @name Traces cut by the partitioner
     *
     * A parallel trace is an *interior* trace whose backward side arrives by
     * message rather than by gather. It is not a boundary trace: its exterior
     * state is a genuine neighbouring element that happens to live on another
     * rank, so it takes the interior averaging throughout. Routing one through
     * the boundary path would reflect a perfectly good neighbour state about
     * itself and produce a plausible, wrong answer.
     *
     * The layout deliberately matches the interior one, so the gather,
     * scatter and flux kernels are the same code. #m_parT0 addresses the local
     * trace field exactly as #m_intT0 does. #m_parT1 addresses a receive buffer
     * instead: `offset_st` is zero, `offset` is the entry's start in that
     * buffer, and `compSize` is the neighbour's point count for the trace,
     * which under variable order need not equal this rank's. Those last two
     * depend on the component count and so are filled by
     * SetUpParallelComm() rather than by the constructor.
     */
    ///@{
    /// Local side of every trace whose neighbour is on another rank.
    std::vector<TraceDetails> m_parT0;
    /// Remote side of the same traces, as laid out in a receive buffer.
    std::vector<TraceDetails> m_parT1;
    std::vector<TraceDeviceData> m_parT0Dev;
    std::vector<TraceDeviceData> m_parT1Dev;
    /// Offsets of the parallel traces in the global trace, for normals and
    /// other whole-trace fields; see #m_intGloTraceOffset.
    std::vector<std::vector<size_t>> m_parGloTraceOffset;
    std::vector<std::vector<size_t>> m_parGloTraceScalarOffset;
    std::vector<std::vector<size_t>> m_parGloTraceCompSize;
    /// Execution-space mirror of the three above; see #m_intGloTraceDev.
    std::vector<GloTraceOffsetDeviceData> m_parGloTraceDev;

    /// Exchange entry each parallel trace belongs to, indexed [block][trace].
    /// The blocks group traces by shape and order; the exchange orders them by
    /// neighbouring rank and then by an id both ranks agree on. The two
    /// orderings are unrelated, and this is what joins them.
    std::vector<std::vector<size_t>> m_parEntry;

    /// Local trace ids of the periodic traces whose partner is on another
    /// rank, mapped to the pair of global mesh ids naming the pair. Filled by
    /// RegisterPeriodicPairs(); read by SetUpParallelTraces(), which keys such
    /// a trace on the pair rather than on itself, both halves being distinct
    /// entities that no single id names.
    struct PeriodicRemote
    {
        /// Global mesh ids of the two halves of the pair.
        size_t gidSelf;
        size_t gidPartner;
        /// GetFaceOrientation(this trace, partner), as DisContField stores
        /// it - the same sense the local pairing loop inverts before
        /// composing.
        StdRegions::Orientation orient;
    };
    std::map<size_t, PeriodicRemote> m_periodicRemote;

    /// Entry indices exchanged with each neighbouring rank, in the order both
    /// ends agree on. Handed to MultiRegions::AssemblyComm.
    std::map<int, std::vector<size_t>> m_parSharedEntries;
    /// Quadrature points per component that each entry sends, and receives.
    /// The two differ wherever the neighbouring element carries a different
    /// polynomial order. Scaled by the component count in SetUpParallelComm().
    std::vector<size_t> m_parSendPts;
    std::vector<size_t> m_parRecvPts;
    /// Where each entry's payload is read from in the local trace field, in
    /// exchange order: the same (offset_st, offset, compSize) triple the
    /// gather kernels use, flattened so the pack loop needs no block index.
    std::vector<size_t> m_parSendOffsetSt;
    std::vector<size_t> m_parSendOffset;
    std::vector<size_t> m_parSendCompSize;

    /// Communicator the exchange runs over; null in serial.
    LibUtilities::CommSharedPtr m_rowComm;

    /// Start of each entry in the buffers, in points; see the note where they
    /// are filled. Multiplied by the component count at the point of use, so
    /// one set serves every channel.
    std::vector<size_t> m_parSendPtsOffset;
    std::vector<size_t> m_parRecvPtsOffset;

    /**
     * @brief One exchange's worth of buffers and requests.
     *
     * There is more than one because the diffusion path has more than one
     * trace to send: the state, and its gradient, which carries `nDim` times
     * as many components. Both are in flight at once - the whole point of
     * starting them at different moments is that each overlaps the local work
     * that follows it - so they cannot share a buffer, and a single exchange
     * rebuilt whenever the component count changed would serve neither.
     *
     * What they *do* share is #m_parT0 and #m_parT1: the trace details are
     * written in points and scaled by the component count in the kernels, so
     * they describe a buffer of any width.
     */
    struct ParallelChannel
    {
        std::unique_ptr<MultiRegions::AssemblyComm<TData>> comm;
        LibUtilities::MemoryRegion<TData> sendBuffer;
        LibUtilities::MemoryRegion<TData> recvBuffer;
        /// Component count this channel was built for; rebuilt if it changes.
        unsigned numComp = 0;
        /// Set between Begin and End, so a missing End is caught rather than
        /// silently reading a stale buffer.
        bool inFlight = false;
    };

    /// Channel 0 carries the state trace, channel 1 the gradient trace; see
    /// TraceFluxOp::BeginParallelExchange().
    static constexpr unsigned s_numParChannels = 2;
    std::array<ParallelChannel, s_numParChannels> m_parChannel;
    ///@}

    // Left/Fwd boundary Trace details
    std::vector<TraceDetails> m_bndT0;
    // Execution-space mirror of the per-trace fields of the above, one entry
    // per block, read by GetLocalBndTrace()
    std::vector<TraceDeviceData> m_bndT0Dev;

    // Dirichlet Boundary Details - put into struct
    /// Global trace id of each boundary trace, used to fetch the exterior
    /// state from TraceFluxOp::m_BndCondOp.
    std::vector<std::vector<size_t>> m_bndTraceIdT1;

    /// Device-resident location of each Dirichlet trace's boundary data,
    /// indexed by block then trace. Built once by PackBndBCLocations().
    struct BndCondLocation
    {
        LibUtilities::MemoryRegion<unsigned> blk;
        LibUtilities::MemoryRegion<size_t> offset;
        LibUtilities::MemoryRegion<size_t> compOffset;

        /// Set when every trace in the block reads from the same boundary
        /// storage block with the same component stride, so the block draws on
        /// a single boundary region. This is the common case once the traces
        /// are ordered by that storage, and it lets GetDirBCTrace() hoist the
        /// base pointer out of the kernel: it then reads neither #blk nor
        /// #compOffset, and so avoids a load that depends on another load.
        bool singleBlock = false;
        /// Storage block every trace reads from, when #singleBlock.
        unsigned singleBlockId = 0;
        /// Stride between components of that block, when #singleBlock.
        size_t compStride = 0;
    };
    std::vector<BndCondLocation> m_bndCondLoc;

    /// Base pointer of each boundary-data block, refreshed on every apply.
    LibUtilities::MemoryRegion<const TData *> m_bndCondBase;
    /// Writable counterpart, used only by GatherBndInteriorState().
    LibUtilities::MemoryRegion<TData *> m_bndCondBaseW;

    /// SpatialDomains::BoundaryConditionType of every (storage block,
    /// component) pair, indexed [storageBlk * m_numBCComp + comp]. Built once
    /// by PackBndBCLocations(); read by GetDirBCTrace() and GetNeuBCTrace() to
    /// decide which components each of them owns.
    LibUtilities::MemoryRegion<unsigned> m_bcTypeMask;
    unsigned m_numBCComp = 0;
    /// Points per direction on the boundary trace itself. One set per block,
    /// not per trace: the boundary side is taken to define the global trace.
    std::vector<std::vector<unsigned>> m_bndNptsT1;
    std::vector<std::vector<size_t>> m_bndGloTraceOffset;
    /// Scalar-layout counterpart of the above; see
    /// #m_intGloTraceScalarOffset.
    std::vector<std::vector<size_t>> m_bndGloTraceScalarOffset;
    /// Total length of a single-component trace array, i.e. the sum of every
    /// block's CompSize(). Bounds a read made with a scalar offset.
    size_t m_gloTraceScalarSize = 0;
    // Global Trace Component size  (for Normals and Adv Vel)
    std::vector<std::vector<size_t>> m_bndGloTraceCompSize;
    /// Execution-space mirror of the three above; see #m_intGloTraceDev.
    std::vector<GloTraceOffsetDeviceData> m_bndGloTraceDev;

    // Could add Parallel interfaces here

    // workspace details
    /// Component count the workspace was last sized for; SetWorkSpace() is a
    /// no-op while this is unchanged.
    unsigned m_numComp = 0;
    /// Largest, over all blocks, of the traces in a block times the points on
    /// the largest single trace. Sizes the per-thread slices of #m_loc and
    /// #m_wsp, which the gather kernels index by flattened (trace, component);
    /// see AccumulateWorkSpaceBound().
    size_t m_maxTraceXnPts = 0;

    /// Backing allocation for all of the scratch pointers below.
    LibUtilities::MemoryRegion<TData> m_workSpace;
    /// Scratch used by the two-dimensional interpolation kernels.
    TData *m_wsp;
    /// Scratch holding one trace in element-local layout.
    TData *m_loc;
    /// Forward-side trace values, packed on the global trace.
    TData *m_gloT0;
    /// Backward-side trace values, packed on the global trace.
    TData *m_gloT1;
    /// Numerical flux produced from #m_gloT0 and #m_gloT1.
    TData *m_flux;
    /// Trace normals gathered into the same packed layout as #m_flux.
    TData *m_norms;

    // space for individual trace processing
    unsigned m_npT0[2];     // local points on trace 0
    unsigned m_npT1[2];     // local points on trace 1
    unsigned m_npT[2];      // global trace points
    unsigned m_npTot;       // total number of points in trace
    bool m_T0Collocated[2]; // Trace 0 is collocated
    bool m_T1Collocated[2]; // trace 1 is collocated

    /// Whether the *backward* map is a no-op, which is a different question
    /// from whether the forward one is. A trace already carrying the global
    /// trace's points needs no interpolation onto it, but if its neighbour
    /// carries fewer modes the flux coming back must still be filtered to the
    /// modes the two sides share - `interpFwd` null, `interpBwd` not. Gating
    /// the scatter on the forward flags skips that filter, leaving each side
    /// of the trace with a different flux and the scheme no longer
    /// conservative.
    bool m_T0CollocatedBwd[2];
    bool m_T1CollocatedBwd[2];

    /**
     * @brief Scratch description of a single element trace, used only while
     * the constructor is building the blocks.
     *
     * Two of these, one per side, are compared to decide whether a trace can
     * join an existing block and whether interpolation is needed between the
     * sides.
     */
    /**
     * @brief One side of one trace, as the constructor discovers it.
     *
     * Trivially copyable, and deliberately so: a trace straddling a partition
     * boundary is described by shipping this struct to the neighbouring rank,
     * through LibUtilities::SharedPayloadResolver, which transports payloads
     * as raw bytes. That is why the per-direction arrays are fixed-size rather
     * than `std::vector`, and why every member is value-initialised - a
     * padding-free, fully written struct is what makes the byte copy
     * meaningful on the far side.
     *
     * Directions beyond #m_traceDim are left at zero on every instance, so
     * whole-array comparisons of @p n and @p ptype are equivalent to comparing
     * only the directions in use.
     *
     * @note @p eid, @p t, @p compSize and @p compSizeSum address the *owning*
     *       rank's storage and mean nothing on any other. A received
     *       TraceInfo is read for @p n, @p nc, @p ptype and @p orient alone.
     */
    struct TraceInfo
    {
        /// Element id owning this trace.
        size_t eid = 0;
        /// Local trace number within that element.
        unsigned t = 0;
        /// Component stride of the block the element belongs to.
        size_t compSize = 0;
        /// Running sum of component sizes over preceding blocks.
        size_t compSizeSum = 0;
        /// Order of this trace, per trace direction.
        std::array<unsigned, s_maxTraceDim> n{};
        /// Order of the connected trace, per trace direction.
        std::array<unsigned, s_maxTraceDim> nc{};
        /// Quadrature point type per trace direction.
        std::array<LibUtilities::PointsType, s_maxTraceDim> ptype{};
        /// Orientation of this trace relative to the global trace.
        StdRegions::Orientation orient = StdRegions::eNoOrientation;
        /// Orientation of this trace relative to the *partner's* trace
        /// geometry - what the far side of an exchange must use, since it
        /// consumes these values against its own trace. For a trace the
        /// partitioner cut the two geometries are one and the same and this
        /// equals @p orient; for a periodic pair straddling ranks they are
        /// congruent but distinct, and this carries the periodic relative
        /// orientation pre-composed, exactly as the local pairing loop
        /// composes it for a pair it can see whole.
        StdRegions::Orientation porient = StdRegions::eNoOrientation;
        /// The *global* trace this side maps onto, as the owning rank built
        /// it. Filled only for a trace being offered to the resolver, where it
        /// is the one field of this struct whose whole purpose is to be
        /// compared against another rank's copy; see
        /// SetUpParallelTraces().
        std::array<unsigned, s_maxTraceDim> gn{};
        std::array<LibUtilities::PointsType, s_maxTraceDim> gptype{};
    };

    static_assert(std::is_trivially_copyable_v<TraceInfo>,
                  "TraceInfo is shipped between ranks as raw bytes");

    /**
     * @brief Find the interior block an already-paired trace belongs to.
     *
     * A trace joins an existing block only if *both* sides agree with that
     * block's saved representative on point count and point type in every
     * trace direction, so that the block's single set of interpolation
     * matrices remains valid.
     *
     * @param SaveTraceData - One representative pair per existing block.
     * @param tinfo0        - Forward side of the trace being placed.
     * @param elmt          - Element owning the backward side.
     * @param t             - Local trace number of the backward side.
     *
     * @return Index of the matching block, or `m_intT0.size()` if none matches
     *         and a new block must be started. With `m_traceDim == 0` there is
     *         nothing to distinguish and 0 is always returned.
     */
    // check to see if already have similar interior trace interpolation
    NEK_FORCE_INLINE unsigned HaveSimilarTraceInfo(
        std::vector<std::pair<TraceInfo, TraceInfo>> &SaveTraceData,
        TraceInfo &tinfo0, LocalRegions::ExpansionSharedPtr &elmt, unsigned t)
    {

        if (m_traceDim == 0)
        {
            return 0; // always same
        }

        unsigned b = 0;

        for (; b < m_intT0.size(); ++b)
        {
            bool IsSimilar = true;

            // check tinfo0
            for (unsigned d = 0; d < m_traceDim; ++d)
            {
                if ((tinfo0.n[d] != SaveTraceData[b].first.n[d]) ||
                    (tinfo0.ptype[d] != SaveTraceData[b].first.ptype[d]))
                {
                    IsSimilar = false;
                    break;
                }
            }
            // check second trace  if not exited found not similar
            if (IsSimilar)
            {
                for (unsigned d = 0; d < m_traceDim; ++d)
                {
                    auto bkey = elmt->GetTraceBasisKey(t, d);
                    if (SaveTraceData[b].second.n[d] != bkey.GetNumPoints() ||
                        SaveTraceData[b].second.ptype[d] !=
                            bkey.GetPointsType())
                    {
                        IsSimilar = false;
                        break;
                    }
                }
            }

            if (IsSimilar)
            {
                // exit and return index
                return b;
            }
        }
        // no previous similar faces so return next b value
        return b;
    }

    /**
     * @brief Find the Dirichlet boundary block a trace belongs to.
     *
     * As HaveSimilarTraceInfo(), but the second side compared against is the
     * boundary trace expansion itself rather than a neighbouring element.
     *
     * @param SaveTraceData - One representative pair per existing block.
     * @param tinfo0        - Interior side of the boundary trace.
     * @param telmt         - The boundary trace expansion.
     *
     * @return Index of the matching block, or `m_bndT0.size()` if a new
     *         block must be started.
     */
    // check to see if already have similar interior trace interpolation
    NEK_FORCE_INLINE unsigned HaveSimilarBndTraceInfo(
        std::vector<std::pair<TraceInfo, TraceInfo>> &SaveTraceData,
        TraceInfo &tinfo0, LocalRegions::ExpansionSharedPtr &telmt)
    {

        if (m_traceDim == 0)
        {
            return 0; // always same
        }

        unsigned b = 0;

        for (; b < m_bndT0.size(); ++b)
        {
            bool IsSimilar = true;

            // check tinfo0
            for (unsigned d = 0; d < m_traceDim; ++d)
            {
                if ((tinfo0.n[d] != SaveTraceData[b].first.n[d]) ||
                    (tinfo0.ptype[d] != SaveTraceData[b].first.ptype[d]))
                {
                    IsSimilar = false;
                    break;
                }
            }
            // check second part if not exited already
            if (IsSimilar)
            {
                for (unsigned d = 0; d < m_traceDim; ++d)
                {
                    auto bkey = telmt->GetBasis(d);
                    if (SaveTraceData[b].second.n[d] != bkey->GetNumPoints() ||
                        SaveTraceData[b].second.ptype[d] !=
                            bkey->GetPointsType())
                    {
                        IsSimilar = false;
                        break;
                    }
                }
            }
            if (IsSimilar)
            {
                // exit and return index
                return b;
            }
        }
        // no previous similar faces so return next b value
        return b;
    }

    /// One trace whose neighbouring element is on another rank, while the
    /// exchange that will feed it is still being worked out.
    struct ParallelTrace
    {
        /// Local trace expansion id.
        size_t tid;
        /// Id both ranks name this trace by.
        int64_t key;
        /// Rank holding the other side.
        int rank;
        /// This rank's side of the trace.
        TraceInfo local;
        /// The neighbour's side, as it arrived.
        TraceInfo remote;
    };

    /// Quadrature points on one side of a trace, across all trace directions.
    NEK_FORCE_INLINE size_t NumTracePoints(const TraceInfo &tinfo) const
    {
        size_t npts = 1;
        for (unsigned d = 0; d < m_traceDim; ++d)
        {
            npts *= tinfo.n[d];
        }
        return npts;
    }

    /// Order parallel traces by neighbouring rank, then by the id both ranks
    /// agree on - which is what makes the two ends of an exchange line up.
    NEK_FORCE_INLINE static bool ByRankThenKey(const ParallelTrace &a,
                                               const ParallelTrace &b)
    {
        return std::tie(a.rank, a.key) < std::tie(b.rank, b.key);
    }

    /**
     * @brief Claim the unpaired traces whose neighbour is on another rank.
     *
     * Called with @p TraceData holding every trace the pairing loop could not
     * complete. On a partitioned mesh those are of two kinds, and nothing
     * local distinguishes them: a trace on a physical boundary of the mesh,
     * and a trace the partitioner cut, whose neighbouring element is intact
     * but elsewhere. This asks the other ranks which is which.
     *
     * Every trace this rank's boundary conditions do not claim is registered
     * with a rendezvous resolver under an id both ranks name it by, carrying
     * this rank's TraceInfo as payload. Two
     * exchanges later every rank holds its neighbour's TraceInfo for every
     * shared trace - enough to describe the far side of the trace completely,
     * and so to build #m_parT0 / #m_parT1 in the same layout the interior
     * blocks use. A candidate that comes back shared with nobody is a physical
     * boundary after all, and is left in @p TraceData for the boundary pass.
     *
     * Nothing happens on one rank, where no trace can be remote.
     *
     * @param expansionList - Expansion list the operator acts on.
     * @param elmtToTrace   - Element trace to global trace expansion map.
     * @param TraceData     - Unpaired traces; parallel ones are erased.
     * @param PtsOffset     - Point offset of each element trace within its
     *                        block, as built by the pairing loop.
     */
    void SetUpParallelTraces(
        const MultiRegions::ExpListSharedPtr &expansionList,
        Array<OneD, Array<OneD, LocalRegions::ExpansionSharedPtr>> &elmtToTrace,
        std::map<size_t, TraceInfo> &TraceData,
        std::map<size_t, std::map<size_t, size_t>> &PtsOffset)
    {
        auto session = expansionList->GetSession();
        auto comm    = session ? session->GetComm()->GetRowComm() : nullptr;

        if (!comm || comm->GetSize() == 1)
        {
            return;
        }

        m_rowComm = comm;

        auto trace = expansionList->GetTrace();

        auto transport =
            std::make_shared<LibUtilities::AlltoallvTransport>(comm);
        LibUtilities::SharedPayloadResolver<TraceInfo> resolver(transport);

        // Registration order is irrelevant - the resolver is keyed - but the
        // keys have to be remembered, since a periodic trace is not named by
        // its own id, and so does the payload, which carries more than the
        // TraceInfo left in TraceData does. Filled in here with everything but
        // the neighbour, which the resolve below supplies.
        std::vector<ParallelTrace> candidates;
        candidates.reserve(TraceData.size());

        for (const auto &it : TraceData)
        {
            const auto &tinfo = it.second;
            const size_t gid =
                elmtToTrace[tinfo.eid][tinfo.t]->GetGeom()->GetGlobalID();

            // A trace on a boundary region of this rank takes its exterior
            // state from the boundary condition and cannot also have a
            // neighbouring element, so there is nothing to ask about it. Most
            // unpaired traces are of this kind, and skipping them keeps the
            // rendezvous carrying only the traces that might really be shared.
            // Periodic regions hold no boundary values and so are not in that
            // storage, which is what lets a periodic trace through to the
            // resolver, where it needs to go.
            if (this->m_BndCondOp && this->m_BndCondOp->HasTrace(gid))
            {
                continue;
            }

            // Carry the global trace this rank built for the geometry, so that
            // the far side can check the two agree. They should:
            // ExpList's trace constructor reduces every trace's order over all
            // ranks and raises each to the maximum, expressly so that the
            // trace carries the highest order "even on parition interfaces".
            // If that ever stops holding, the two ranks evaluate the flux on
            // different traces and quietly disagree about it, which is worth
            // one array's worth of message to rule out.
            TraceInfo payload = tinfo;
            auto telmt        = elmtToTrace[tinfo.eid][tinfo.t];
            for (unsigned d = 0; d < m_traceDim; ++d)
            {
                payload.gn[d]     = telmt->GetNumPoints(d);
                payload.gptype[d] = telmt->GetPointsType(d);
            }

            // How the far side must read this trace. Identical to orient for
            // a genuinely shared geometry; for a periodic pair the receiver's
            // trace is a different, congruent geometry, and the periodic
            // relative orientation folds in here.
            //
            // Uninverted, which needs saying because the local pairing loop
            // *does* invert - but it inverts the orientation stored under the
            // partner's entry, GetFaceOrientation(partner, mine), where this
            // rank holds only its own entry, GetFaceOrientation(mine,
            // partner). The two are inverses of each other, so inverting here
            // as well would double-invert. Every self-inverse orientation -
            // all the aligned and 180-degree cases, and every 2D edge - hides
            // that mistake completely; the two 90-degree rotations, which
            // invert to each other, are the only witnesses, and were.
            payload.porient = tinfo.orient;
            auto per        = m_periodicRemote.find(it.first);
            if (per != m_periodicRemote.end())
            {
                payload.porient =
                    ComposeOrient(tinfo.orient, per->second.orient);
            }

            ParallelTrace c;
            c.tid   = it.first;
            c.key   = ParallelTraceKey(it.first, gid);
            c.rank  = -1;
            c.local = payload;

            candidates.push_back(c);
            resolver.Register(c.key, payload);
        }

        resolver.Resolve();

        const int myRank = comm->GetRank();
        std::vector<ParallelTrace> par;
        par.reserve(candidates.size());

        for (auto p : candidates)
        {
            for (const auto &[rank, tinfo] : resolver.GetSharedPayloads(p.key))
            {
                if (rank == myRank)
                {
                    continue;
                }

                ASSERTL0(p.rank < 0,
                         "A trace has at most one neighbouring element, so at "
                         "most one other rank may share it");
                p.rank   = rank;
                p.remote = tinfo;
            }

            if (p.rank < 0)
            {
                // Nothing claims this trace. With no boundary condition
                // operator that is expected - the Riemann unit test and the
                // profiler build their fields directly, and everything
                // unpaired is a boundary to them. With one, every trace it did
                // not claim was offered here precisely because it had to be
                // shared, and one that is neither is a trace with no exterior
                // state at all. Downstream, GetTraceLocation() would look it
                // up in the boundary storage, miss, and - in a release build,
                // where its ASSERTL1 does not run - read whatever is there.
                ASSERTL0(!this->m_BndCondOp,
                         "Trace is on no boundary region of this rank and is "
                         "shared with no other: it has no exterior state");
                continue;
            }

            par.push_back(p);
        }

        for (const auto &p : par)
        {
            TraceData.erase(p.tid);
        }

        if (par.empty())
        {
            return;
        }

        std::sort(par.begin(), par.end(), ByRankThenKey);

        BuildParallelBlocks(par, elmtToTrace, trace, PtsOffset);
    }

    /**
     * @brief Group the resolved parallel traces into blocks.
     *
     * The interior counterpart of this work is inlined in the constructor, and
     * this follows it closely: traces agreeing on point count and type on both
     * sides share a block, and so share one set of interpolation matrices and
     * one set of loop bounds. What differs is only where the backward side
     * comes from, and that its address is not yet known.
     *
     * @param par         - Parallel traces, already in exchange order.
     * @param elmtToTrace - Element trace to global trace expansion map.
     * @param trace       - The global trace expansion list.
     * @param PtsOffset   - Point offset of each element trace within its block.
     */
    void BuildParallelBlocks(
        const std::vector<ParallelTrace> &par,
        Array<OneD, Array<OneD, LocalRegions::ExpansionSharedPtr>> &elmtToTrace,
        const MultiRegions::ExpListSharedPtr &trace,
        std::map<size_t, std::map<size_t, size_t>> &PtsOffset)
    {
        std::vector<std::pair<TraceInfo, TraceInfo>> SaveTraceData;

        m_parSendPts.resize(par.size());
        m_parRecvPts.resize(par.size());
        m_parSendPtsOffset.resize(par.size());
        m_parRecvPtsOffset.resize(par.size());
        size_t sendPrefix = 0, recvPrefix = 0;
        m_parSendOffsetSt.resize(par.size());
        m_parSendOffset.resize(par.size());
        m_parSendCompSize.resize(par.size());

        for (size_t e = 0; e < par.size(); ++e)
        {
            TraceInfo tinfo0 = par[e].local;
            TraceInfo tinfo1 = par[e].remote;

            // The remote side is consumed against this rank's trace geometry,
            // which is what porient describes; orient describes the remote
            // rank's own trace, a different geometry when the pair is
            // periodic.
            tinfo1.orient = tinfo1.porient;

            // Each side needs the other's point counts, which is what decides
            // whether it is collocated with the global trace or must be
            // interpolated onto it.
            tinfo0.nc = tinfo1.n;
            tinfo1.nc = tinfo0.n;

            // The two elements need not agree - one side of a trace may carry
            // a lower order than the other, here as anywhere else, and the
            // interpolation below handles it. What must agree is the *global*
            // trace the two ranks each interpolate onto, or they evaluate the
            // flux on different traces and quietly disagree about it.
            // ExpList's trace constructor reduces trace orders across ranks to
            // make that so; this is the check that it did.
            //
            // The remote reports its global trace in its own frame. A periodic
            // relation that transposes the two frames - possible only for
            // faces - swaps the directions, and that transposition is exactly
            // the difference in transposedness between orient and porient.
            TraceInfo remoteGlo = tinfo1;
            const bool remTransposed =
                (m_traceDim == 2) && ((par[e].remote.orient >=
                                       StdRegions::eDir1FwdDir2_Dir2FwdDir1) !=
                                      (par[e].remote.porient >=
                                       StdRegions::eDir1FwdDir2_Dir2FwdDir1));
            if (remTransposed)
            {
                std::swap(remoteGlo.gn[0], remoteGlo.gn[1]);
                std::swap(remoteGlo.gptype[0], remoteGlo.gptype[1]);
            }
            ASSERTL0(tinfo0.gn == remoteGlo.gn &&
                         tinfo0.gptype == remoteGlo.gptype,
                     "The two ranks sharing this trace built different global "
                     "traces for it, and would evaluate the flux on each");

            m_parSharedEntries[par[e].rank].push_back(e);
            m_parSendPts[e]      = NumTracePoints(tinfo0);
            m_parRecvPts[e]      = NumTracePoints(tinfo1);
            m_parSendOffsetSt[e] = tinfo0.compSizeSum;
            m_parSendOffset[e]   = PtsOffset[tinfo0.eid][tinfo0.t];
            m_parSendCompSize[e] = tinfo0.compSize;

            // Where this entry starts in the buffers, counted in points
            // rather than in values. The buffers are laid out as the running
            // sum of points times the component count, so the point figure is
            // the same whatever is being exchanged - which is what lets one
            // set of trace details serve exchanges of different widths.
            m_parSendPtsOffset[e] = sendPrefix;
            m_parRecvPtsOffset[e] = recvPrefix;
            sendPrefix += m_parSendPts[e];
            recvPrefix += m_parRecvPts[e];

            bool SaveTraceInfo = false;

            auto b = HaveSimilarParTraceInfo(SaveTraceData, tinfo0, tinfo1);

            if (b == m_parT0.size())
            {
                m_parT0.emplace_back();
                m_parT1.emplace_back();
                m_parEntry.emplace_back();
                m_parGloTraceOffset.emplace_back();
                m_parGloTraceCompSize.emplace_back();

                SaveTraceInfo = true;
            }

            m_parT0[b].offset_st.push_back(tinfo0.compSizeSum);
            m_parT0[b].offset.push_back(PtsOffset[tinfo0.eid][tinfo0.t]);
            m_parT0[b].compSize.push_back(tinfo0.compSize);

            // The remote side is read out of a receive buffer. Its start is
            // given in points and multiplied by the component count by the
            // gather kernel itself, which is why offset_st carries it and
            // offset is zero: the same three numbers then address a buffer of
            // any width.
            m_parT1[b].offset_st.push_back(m_parRecvPtsOffset[e]);
            m_parT1[b].offset.push_back(0);
            m_parT1[b].compSize.push_back(m_parRecvPts[e]);

            m_parEntry[b].push_back(e);
            m_parGloTraceOffset[b].push_back(trace->GetPhys_Offset(par[e].tid));

            if (m_traceDim)
            {
                auto telmt = elmtToTrace[tinfo0.eid][tinfo0.t];

                TraceInfo tinfoglo;
                for (unsigned d = 0; d < m_traceDim; ++d)
                {
                    tinfoglo.n[d]     = telmt->GetNumPoints(d);
                    tinfoglo.ptype[d] = telmt->GetPointsType(d);
                }
                tinfoglo.orient = (m_traceDim == 1)
                                      ? StdRegions::eForwards
                                      : StdRegions::eDir1FwdDir1_Dir2FwdDir2;

                PopulateTraceVecs(telmt, tinfo0, tinfoglo, m_parT0[b]);
                PopulateTraceVecs(telmt, tinfo1, tinfoglo, m_parT1[b]);
            }
            else
            {
                // segment case with point trace
                m_parT0[0].npts.push_back(1);
                m_parT1[0].npts.push_back(1);
            }

            if (SaveTraceInfo)
            {
                SaveTraceData.push_back(std::make_pair(tinfo0, tinfo1));
            }
        }

        for (unsigned b = 0; b < m_parT0.size(); ++b)
        {
            switch (m_traceDim)
            {
                case 0:
                    SetParallelParams<0>(b);
                    break;
                case 1:
                    SetParallelParams<1>(b);
                    break;
                case 2:
                    SetParallelParams<2>(b);
                    break;
            }

            m_parT0[b].m_nTraceXnPtsPad =
                PadToVectorWidth(m_npTot * m_parT0[b].offset.size());

            AccumulateWorkSpaceBound(m_parT0[b].offset.size());
        }
    }

    /**
     * @brief Find the parallel block a trace belongs to.
     *
     * As HaveSimilarTraceInfo(), but both sides are compared from TraceInfo:
     * the backward side is an element on another rank, with no expansion here
     * to interrogate.
     *
     * @return Index of the matching block, or `m_parT0.size()` if a new block
     *         must be started.
     */
    NEK_FORCE_INLINE unsigned HaveSimilarParTraceInfo(
        const std::vector<std::pair<TraceInfo, TraceInfo>> &SaveTraceData,
        const TraceInfo &tinfo0, const TraceInfo &tinfo1)
    {
        if (m_traceDim == 0)
        {
            return 0; // always same
        }

        unsigned b = 0;

        for (; b < m_parT0.size(); ++b)
        {
            bool IsSimilar = true;

            for (unsigned d = 0; d < m_traceDim; ++d)
            {
                if ((tinfo0.n[d] != SaveTraceData[b].first.n[d]) ||
                    (tinfo0.ptype[d] != SaveTraceData[b].first.ptype[d]) ||
                    (tinfo1.n[d] != SaveTraceData[b].second.n[d]) ||
                    (tinfo1.ptype[d] != SaveTraceData[b].second.ptype[d]))
                {
                    IsSimilar = false;
                    break;
                }
            }

            if (IsSimilar)
            {
                return b;
            }
        }

        // no previous similar faces so return next b value
        return b;
    }

    /**
     * @brief The id both ranks sharing a trace name it by.
     *
     * A trace cut by the partitioner is one mesh entity seen from two sides,
     * so its own global id serves. A periodic pair is two distinct entities
     * that no single id names, and the two ranks hold different halves, so
     * neither half's id is common ground; the smaller of the pair is, and
     * RegisterPeriodicPairs() recorded both.
     */
    NEK_FORCE_INLINE int64_t ParallelTraceKey(const size_t tid,
                                              const size_t gid) const
    {
        auto per = m_periodicRemote.find(tid);
        if (per != m_periodicRemote.end())
        {
            return static_cast<int64_t>(
                std::min(per->second.gidSelf, per->second.gidPartner));
        }

        return static_cast<int64_t>(gid);
    }

    /**
     * @brief Record the orientation and interpolation needed to map one local
     * trace onto the global trace.
     *
     * Appends @p tinfo's orientation to @p T unconditionally. The point counts
     * and interpolation matrices are recorded only for the first trace in the
     * block, since by construction every trace in a block shares them.
     *
     * The two sides are treated as collocated, a null entry in both
     * `interpFwd` and `interpBwd` meaning reorientation alone is enough, when
     * the point counts and types agree, allowing for the transposition implied
     * by a `Dir1FwdDir2` orientation, *and* the connected trace carries the
     * same number of points. Otherwise an interpolation matrix is fetched from
     * the data warehouse for each direction that needs one.
     *
     * @param telmt     - The global trace expansion.
     * @param tinfo     - The local element trace being mapped.
     * @param tinfobase - The global trace to map onto.
     * @param T         - Block details to append to.
     *
     * @note The backward direction is an orthogonal projection, never a plain
     *       interpolation, and where the point counts already agree but the
     *       neighbour carries fewer modes it is a modal filter rather than
     *       nothing at all. Both matter: interpolation is not the adjoint of
     *       the forward map, and a flux left holding modes the lower-order
     *       neighbour cannot represent is not the same flux on the two sides
     *       of the trace. This used to be selectable through an
     *       `ORTHOPROJECT` macro; the alternative is simply wrong at variable
     *       order, and choosing it made `Advection3D_DG_prism_varP` miss the
     *       legacy answer by 2.6e-8 - which is what that case was parked on
     *       for a fortnight.
     */
    // Given two trace Tinfo structure the first representing the
    // local element trace and the second representing the base or
    // global trace populate the relevant maps
    NEK_FORCE_INLINE void PopulateTraceVecs(
        LocalRegions::ExpansionSharedPtr &telmt, TraceInfo &tinfo,
        TraceInfo &tinfobase, TraceDetails &T)
    {
        // Check to  see if traces are at same points
        bool IsSame = false;
        if (tinfo.orient < StdRegions::eDir1FwdDir2_Dir2FwdDir1)
        {
            // Whole-array compares, which reach past m_traceDim. Safe because
            // TraceInfo value-initialises every direction, so the unused
            // slots agree trivially.
            if (tinfo.n == tinfobase.n && tinfo.ptype == tinfobase.ptype)
            {
                IsSame = true;
            }
        }
        else // transpose face
        {
            if (tinfo.n[0] == tinfobase.n[1] && tinfo.n[1] == tinfobase.n[0] &&
                tinfo.ptype[0] == tinfobase.ptype[1] &&
                tinfo.ptype[1] == tinfobase.ptype[0])
            {
                IsSame = true;
            }
        }

        // check points on either side are same
        for (unsigned d = 0; d < m_traceDim; ++d)
        {
            if (tinfo.n[d] != tinfo.nc[d])
            {
                IsSame = false;
            }
        }

        T.orient.push_back(tinfo.orient);

        // only needs one value in block;
        if (T.npts.size() == 0)
        {
            if (IsSame)
            {
                for (unsigned d = 0; d < m_traceDim; ++d)
                {
                    T.npts.push_back(tinfo.n[d]);

                    // no interpolation neccesary
                    T.interpFwd.push_back((TData *)nullptr);

                    // no interpolation necessary
                    T.interpBwd.push_back((TData *)nullptr);
                }
            }
            else
            {
                unsigned d1[2];
                if (tinfo.orient < StdRegions::eDir1FwdDir2_Dir2FwdDir1)
                {
                    d1[0] = 0;
                    d1[1] = 1;
                }
                else
                {
                    d1[0] = 1;
                    d1[1] = 0;
                }

                for (unsigned d = 0; d < m_traceDim; ++d)
                {
                    // Local extents, in local direction order, matching the
                    // IsSame branch above. The interpolation matrices below
                    // are indexed by *global* direction instead, so they are
                    // built from tinfo.n[d1[d]]; consumers that work in the
                    // global frame reorder with ReorientedFaceExtents().
                    T.npts.push_back(tinfo.n[d]);

                    // get hold of trace basis key
                    auto bkey = telmt->GetBasis(d)->GetBasisKey();

                    // set up interpolation basis to
                    // trace (which should be
                    // same as other trace but might
                    // be transposed) - not sure this case is needed since
                    // already seems to be dealt with above.
                    if (tinfo.n[d1[d]] == bkey.GetNumPoints() &&
                        tinfo.ptype[d1[d]] == bkey.GetPointsType())
                    {
                        // no interpolation neccesary
                        T.interpFwd.push_back((TData *)nullptr);

                        // Nor on the way back. The two sides of a variable
                        // order trace do not receive the same function: each
                        // tests the one flux against its own basis, so the
                        // lower-order element legitimately sees the flux
                        // projected onto its space while this one sees the
                        // flux itself. Conservation needs only the constant
                        // mode to agree, and a projection preserves that.
                        //
                        // There was a modal filter here, cutting this side
                        // down to the modes the neighbour could represent. It
                        // had never once run: it was gated on the collocation
                        // flags, which are taken from interpFwd, and interpFwd
                        // is null exactly when the filter was built. Applying
                        // it moves every variable order case away from the
                        // legacy answer, including the 2D one that otherwise
                        // matches exactly, so it is gone rather than fixed.
                        T.interpBwd.push_back((TData *)nullptr);
                    }
                    else
                    {
                        // basis key with points at current local order and
                        // type
                        LibUtilities::BasisKey bbkey(
                            bkey.GetBasisType(), bkey.GetNumModes(),
                            LibUtilities::PointsKey(tinfo.n[d1[d]],
                                                    tinfo.ptype[d1[d]]));

                        // Construct interpolation matrix for local trace to
                        // global trace given by tinfobase
                        T.interpFwd.push_back(
                            this->m_dataWarehouse->template GetData<MemSpace>(
                                LibUtilities::BasisDataKey<TData>(
                                    bbkey, LibUtilities::eInterp,
                                    tinfobase.n[d], tinfobase.ptype[d])));

                        // basis key with points at global trace given by
                        // tinfobase
                        bbkey = LibUtilities::BasisKey(
                            bkey.GetBasisType(), bkey.GetNumModes(),
                            LibUtilities::PointsKey(tinfobase.n[d],
                                                    tinfobase.ptype[d]));

                        // set up an ortho projection (for variable p) from
                        // global trace back to local trace
                        T.interpBwd.push_back(
                            this->m_dataWarehouse->template GetData<MemSpace>(
                                LibUtilities::BasisDataKey<TData>(
                                    bbkey, LibUtilities::eOrthoProject,
                                    tinfo.n[d1[d]], tinfo.ptype[d1[d]])));
                    }
                }
            }
        }
    }

    // helper function for later definitions of flux evaluations

    /**
     * @brief Widen #m_maxTraceXnPts to cover the block whose sizes are
     * currently loaded in the trace-size scratch.
     *
     * The gather kernels run one thread per (trace, component) pair, and a
     * thread that has to interpolate needs two private scratch slices: one
     * holding the reoriented local trace, of `nptsIn0 * nptsIn1` points, and
     * one for the interpolation itself, of `nqto0 * nqfrom1`. The "from"
     * extents are the local trace and the "to" extents the global one when
     * gathering, and the reverse when scattering, so taking the larger of the
     * two in each direction bounds both slices, for both sides of an interior
     * trace and for the interior side of a boundary trace.
     *
     * Multiplying by @p numTrace then bounds a whole block's launch.
     * SetInteriorParams() or SetBoundaryParams() must have been called for
     * the block first.
     *
     * @param numTrace - Traces in the block.
     */
    NEK_FORCE_INLINE void AccumulateWorkSpaceBound(const size_t numTrace)
    {
        // Largest single trace, in points, on either side or on the global
        // trace. Only index 0 is meaningful below two trace dimensions.
        const size_t np0 = std::max({m_npT[0], m_npT0[0], m_npT1[0]});
        const size_t np1 =
            (m_traceDim == 2) ? std::max({m_npT[1], m_npT0[1], m_npT1[1]}) : 1;

        // A transposed trace swaps the two directions, so a scratch slice can
        // pair either extent with either; squaring the larger bounds every
        // combination without needing the per-trace orientation here.
        const size_t npm = std::max(np0, np1);

        m_maxTraceXnPts = std::max(m_maxTraceXnPts, numTrace * npm * npm);
    }

    /**
     * @brief Copy the per-trace fields of @p T into the execution space.
     *
     * @param T - Host-side block details, complete.
     * @param d - Destination, overwritten.
     */
    NEK_FORCE_INLINE void PackTraceBlock(const TraceDetails &T,
                                         TraceDeviceData &d)
    {
        d.numTrace = T.offset.size();

        d.offset_st =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                T.offset_st);
        d.offset =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                T.offset);
        d.compSize =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                T.compSize);

        // Point traces carry no orientation, so leave the region empty rather
        // than allocating nothing; GetTraceBlockView() hands on a null pointer.
        if (!T.orient.empty())
        {
            std::vector<int> orient(T.orient.begin(), T.orient.end());
            d.orient =
                LibUtilities::MemoryRegion<int>::template FromVector<MemSpace>(
                    orient);
        }
    }

    /**
     * @brief Copy one family's global-trace offsets into the execution space.
     *
     * @param gloTOffset   - Per block, per trace, as #m_intGloTraceOffset.
     * @param scalarOffset - Per block, per trace, as
     *                       #m_intGloTraceScalarOffset.
     * @param compSize     - Per block, per trace, as #m_intGloTraceCompSize.
     * @param dev          - Destination, one entry per block, overwritten.
     */
    NEK_FORCE_INLINE void PackGloTraceOffsets(
        const std::vector<std::vector<size_t>> &gloTOffset,
        const std::vector<std::vector<size_t>> &scalarOffset,
        const std::vector<std::vector<size_t>> &compSize,
        std::vector<GloTraceOffsetDeviceData> &dev)
    {
        dev.clear();
        dev.resize(gloTOffset.size());

        for (unsigned b = 0; b < gloTOffset.size(); ++b)
        {
            dev[b].gloTOffset = LibUtilities::MemoryRegion<
                size_t>::template FromVector<MemSpace>(gloTOffset[b]);
            dev[b].scalarOffset = LibUtilities::MemoryRegion<
                size_t>::template FromVector<MemSpace>(scalarOffset[b]);
            dev[b].compSize = LibUtilities::MemoryRegion<
                size_t>::template FromVector<MemSpace>(compSize[b]);
        }
    }

    /**
     * @brief Hand out the pointers a kernel needs to read @p d.
     */
    NEK_FORCE_INLINE GloTraceOffsetView
    GetGloTraceOffsetView(GloTraceOffsetDeviceData &d)
    {
        GloTraceOffsetView v;

        v.gloTOffset   = d.gloTOffset.template GetPtr<MemSpace, ReadOnly>();
        v.scalarOffset = d.scalarOffset.template GetPtr<MemSpace, ReadOnly>();
        v.compSize     = d.compSize.template GetPtr<MemSpace, ReadOnly>();

        return v;
    }

    /**
     * @brief Hand out the pointers a kernel needs to read @p d.
     */
    NEK_FORCE_INLINE TraceBlockView GetTraceBlockView(TraceDeviceData &d)
    {
        TraceBlockView v;

        v.offset_st = d.offset_st.template GetPtr<MemSpace, ReadOnly>();
        v.offset    = d.offset.template GetPtr<MemSpace, ReadOnly>();
        v.compSize  = d.compSize.template GetPtr<MemSpace, ReadOnly>();
        v.orient    = d.orient.size()
                          ? d.orient.template GetPtr<MemSpace, ReadOnly>()
                          : nullptr;
        v.numTrace  = d.numTrace;

        return v;
    }

    /**
     * @brief Refresh the per-block scratch parameters for interior block @p b.
     *
     * Sets #m_npT0, #m_npT1, #m_T0Collocated and #m_T1Collocated from the
     * block's stored details, then derives the global trace size #m_npT and
     * #m_npTot from whichever side is collocated. Must be called before any of
     * the interior gather/scatter helpers are used on that block.
     *
     * @tparam TRACEDIM Trace dimension; must equal #m_traceDim.
     * @param  b        Interior block index.
     *
     * @warning Asserts that at least one side of the block is collocated with
     *          the global trace. The reason both sides cannot require
     *          interpolation is not recorded here.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void SetInteriorParams([[maybe_unused]] const unsigned b)
    {
        if constexpr (TRACEDIM == 0)
        {
            m_npT0[0] = m_npT1[0] = m_npT[0] = m_npTot = 1;
        }
        else if constexpr (TRACEDIM == 1)
        {
            m_T0Collocated[0]    = (m_intT0[b].interpFwd[0] == nullptr);
            m_T0CollocatedBwd[0] = (m_intT0[b].interpBwd[0] == nullptr);
            m_T1Collocated[0]    = (m_intT1[b].interpFwd[0] == nullptr);
            m_T1CollocatedBwd[0] = (m_intT1[b].interpBwd[0] == nullptr);

            // trace sizes
            m_npT0[0] = m_intT0[b].npts[0];
            m_npT1[0] = m_intT1[b].npts[0];

            ASSERTL1(m_T0Collocated[0] || m_T1Collocated[0],
                     "One side must be collocated");

            m_npT[0] = m_T0Collocated[0] ? m_npT0[0] : m_npT1[0];
            m_npTot  = m_npT[0];
        }
        else if constexpr (TRACEDIM == 2)
        {
            m_T0Collocated[0]    = (m_intT0[b].interpFwd[0] == nullptr);
            m_T0CollocatedBwd[0] = (m_intT0[b].interpBwd[0] == nullptr);
            m_T0Collocated[1]    = (m_intT0[b].interpFwd[1] == nullptr);
            m_T0CollocatedBwd[1] = (m_intT0[b].interpBwd[1] == nullptr);
            m_T1Collocated[0]    = (m_intT1[b].interpFwd[0] == nullptr);
            m_T1CollocatedBwd[0] = (m_intT1[b].interpBwd[0] == nullptr);
            m_T1Collocated[1]    = (m_intT1[b].interpFwd[1] == nullptr);
            m_T1CollocatedBwd[1] = (m_intT1[b].interpBwd[1] == nullptr);

            // trace sizes
            m_npT0[0] = m_intT0[b].npts[0];
            m_npT0[1] = m_intT0[b].npts[1];
            m_npT1[0] = m_intT1[b].npts[0];
            m_npT1[1] = m_intT1[b].npts[1];

            ASSERTL1((m_T0Collocated[0] && m_T0Collocated[1]) ||
                         (m_T1Collocated[0] && m_T1Collocated[1]),
                     "One trace must be collocated");
            bool T0IsCollocated =
                (m_T0Collocated[0] && m_T0Collocated[1]) ? true : false;

            // one side has to be collocated and so choose the size of
            // collocated trace to define global trace details
            m_npT[0] = T0IsCollocated ? m_npT0[0] : m_npT1[0];
            m_npT[1] = T0IsCollocated ? m_npT0[1] : m_npT1[1];
            m_npTot  = m_npT[0] * m_npT[1];
        }
    }

    /**
     * @brief Gather both sides of every interior trace in block @p b into the
     * packed global-trace layout.
     *
     * For each trace the forward and backward values are read from their
     * element-local positions in @p inPtr, reoriented onto the global trace
     * and written to @p gloT0 and @p gloT1 with a component stride of the
     * block's padded trace size. Where the local and global traces are not
     * collocated the reorientation is followed by an interpolation.
     *
     * SetInteriorParams() must have been called for the same block first.
     *
     * The traces of a block are gathered by a single launch each for the
     * forward and backward sides, one thread per (trace, component) pair, so
     * nothing here may read a member from inside a kernel: NEKTAR_LAMBDA
     * captures by value in the device space, and naming a member captures
     * `this`, a host pointer. Per-trace data therefore arrives through a
     * TraceBlockView and per-block data as kernel arguments.
     *
     * @tparam TRACEDIM Trace dimension; must equal #m_traceDim.
     * @param  b        Interior block index.
     * @param  numflux  Number of components per trace point.
     * @param  inPtr    Physical-space input in element-local block layout.
     * @param  gloT0    Forward-side output, usually #m_gloT0.
     * @param  gloT1    Backward-side output, usually #m_gloT1.
     */
    // fill fwd & bwd interior global traces NOte this performs a
    // series of loops over the traces and most likely performs a copy
    // that realigns the points into a stack of traces. There is a
    // possibility of an interpolation but only for variable p. We
    // could (but am not) copy the points into an interleaved format
    // for AVX but since we are not often doing the interpolation not
    // sure this would have much benefit

    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void GetInteriorTraces(const unsigned b,
                                            const unsigned numflux,
                                            const TData *inPtr, TData *gloT0,
                                            TData *gloT1)
    {
        const auto T0 = GetTraceBlockView(m_intT0Dev[b]);
        const auto T1 = GetTraceBlockView(m_intT1Dev[b]);

        const auto npTBlock = m_intT0[b].m_nTraceXnPtsPad;

        if constexpr (TRACEDIM == 0)
        {
            // component packed vector which is padded in AVX case
            LocPointToGloPointBlock<ExecSpace>(numflux, T0, inPtr, gloT0,
                                               npTBlock);
            LocPointToGloPointBlock<ExecSpace>(numflux, T1, inPtr, gloT1,
                                               npTBlock);
        }
        else if constexpr (TRACEDIM == 1)
        {
            // reorientate/copy and interpolate Traces into
            // component packed vector
            LocEdgeToGloEdgeBlock<ExecSpace>(
                numflux, T0, m_intT0[b].interpFwd[0], m_npT0[0], inPtr, m_loc,
                m_npTot, gloT0, npTBlock, m_T0Collocated[0]);

            LocEdgeToGloEdgeBlock<ExecSpace>(
                numflux, T1, m_intT1[b].interpFwd[0], m_npT1[0], inPtr, m_loc,
                m_npTot, gloT1, npTBlock, m_T1Collocated[0]);
        }
        else if constexpr (TRACEDIM == 2)
        {
            // reorientate/copy and interpolate Traces into
            // component packed vector
            LocFaceToGloFaceBlock<ExecSpace>(
                numflux, T0, m_intT0[b].interpFwd[0], m_intT0[b].interpFwd[1],
                m_npT0[0], m_npT0[1], inPtr, m_loc, m_wsp, m_npT[0], m_npT[1],
                gloT0, npTBlock, m_T0Collocated[0], m_T0Collocated[1]);

            LocFaceToGloFaceBlock<ExecSpace>(
                numflux, T1, m_intT1[b].interpFwd[0], m_intT1[b].interpFwd[1],
                m_npT1[0], m_npT1[1], inPtr, m_loc, m_wsp, m_npT[0], m_npT[1],
                gloT1, npTBlock, m_T1Collocated[0], m_T1Collocated[1]);
        }
    }

    /**
     * @brief Scatter the computed interior flux from #m_flux back to the two
     * element-local trace layouts.
     *
     * The inverse of GetInteriorTraces(): each trace's flux is projected back
     * from the global trace, reoriented to element-local ordering and written
     * into @p fluxPtr. The forward side receives the flux as computed and the
     * backward side its negation, so that the same numerical flux leaves one
     * element and enters the other.
     *
     * @tparam TRACEDIM Trace dimension; must equal #m_traceDim.
     * @tparam APPEND   Accumulate into @p fluxPtr rather than overwriting it.
     * @param  b        Interior block index.
     * @param  numflux  Number of components per trace point.
     * @param  fluxPtr  Output in element-local block layout.
     *
     * @note #m_flux is read twice and left unchanged; the backward side's sign
     *       is applied on the way out of the scatter rather than by a separate
     *       pass over it.
     */
    // fill fwd & bwd interior  global traces
    template <unsigned TRACEDIM, bool APPEND>
    NEK_FORCE_INLINE void InterpBackInteriorFlux(const unsigned b,
                                                 const unsigned numflux,
                                                 TData *fluxPtr)
    {

        const auto T0 = GetTraceBlockView(m_intT0Dev[b]);
        const auto T1 = GetTraceBlockView(m_intT1Dev[b]);

        const auto npTBlock = m_intT0[b].m_nTraceXnPtsPad;

        if constexpr (TRACEDIM == 0)
        {
            GloPointToLocPointBlock<APPEND, ExecSpace>(
                numflux, T0, npTBlock, TData(1.0), m_flux, fluxPtr);

            // The backward trace sees the opposite normal. Apply the sign
            // during scattering to avoid a separate pass over the flux data.
            GloPointToLocPointBlock<APPEND, ExecSpace>(
                numflux, T1, npTBlock, TData(-1.0), m_flux, fluxPtr);
        }
        else if constexpr (TRACEDIM == 1)
        {
            // Global extent then local, which differ whenever this side
            // carries a lower order than the trace. Passing the global
            // extent for both was harmless only while T0 was always the
            // collocated side; which element becomes T0 depends on the
            // order they are visited in, and so on the partitioning.
            GloEdgeToLocEdgeBlock<APPEND, ExecSpace, false>(
                numflux, T0, m_intT0[b].interpBwd[0], m_npT[0], m_flux,
                npTBlock, m_loc, m_npT0[0], fluxPtr, m_T0CollocatedBwd[0]);

            GloEdgeToLocEdgeBlock<APPEND, ExecSpace, true>(
                numflux, T1, m_intT1[b].interpBwd[0], m_npT[0], m_flux,
                npTBlock, m_loc, m_npT1[0], fluxPtr, m_T1CollocatedBwd[0]);
        }
        else if constexpr (TRACEDIM == 2)
        {
            GloFaceToLocFaceBlock<APPEND, ExecSpace, false>(
                numflux, T0, m_intT0[b].interpBwd[0], m_intT0[b].interpBwd[1],
                m_npT[0], m_npT[1], m_flux, npTBlock, m_loc, m_wsp, m_npT0[0],
                m_npT0[1], fluxPtr, m_T0CollocatedBwd[0], m_T0CollocatedBwd[1]);

            GloFaceToLocFaceBlock<APPEND, ExecSpace, true>(
                numflux, T1, m_intT1[b].interpBwd[0], m_intT1[b].interpBwd[1],
                m_npT[0], m_npT[1], m_flux, npTBlock, m_loc, m_wsp, m_npT1[0],
                m_npT1[1], fluxPtr, m_T1CollocatedBwd[0], m_T1CollocatedBwd[1]);
        }
    }

    void v_BeginParallelExchange(
        LibUtilities::Field<TData, FieldState::Phys> &trace,
        const unsigned chan) override
    {
        if (m_parT0.empty())
        {
            return;
        }

        // The exchange path is correct on a device because it runs in stream
        // 0. The numbered streams are created blocking, so a kernel launched
        // into the legacy stream is a full ordering point against every
        // per-block stream - including the ones that extracted the trace this
        // pack is about to read. Pin that here rather than inheriting it from
        // the previous operator's reset, which is a convention, not a
        // guarantee.
        Nektar::LoopExecutionSetStreamID(0);

        auto &c = m_parChannel[chan];

        ASSERTL1(!c.inFlight,
                 "An exchange is already in flight on this channel; "
                 "EndParallelExchange() must be called before another begins");

        const auto numComp = trace.GetNumComponents();

        SetUpParallelComm(chan, numComp);

        auto inPtr = trace.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        PackParallelSendBuffer(chan, inPtr, numComp);

        SyncSendBufferForComm(c);

        c.comm->BeginComm();
        c.inFlight = true;
    }

    void v_EndParallelExchange() override
    {
        if (m_parT0.empty())
        {
            return;
        }

        // Pin stream 0 for the receive side too: the H2D fetch below and the
        // gather, flux and scatter kernels of the ApplyParallel() that
        // follows all launch on the current stream, and stream 0 is what
        // orders them against the per-block streams (see the note in
        // v_BeginParallelExchange).
        Nektar::LoopExecutionSetStreamID(0);

        [[maybe_unused]] bool any = false;
        for (auto &c : m_parChannel)
        {
            if (!c.inFlight)
            {
                continue;
            }
            any = true;
            c.comm->EndComm();
            c.inFlight = false;
            SyncRecvBufferAfterComm(c);
        }

        ASSERTL1(any, "No exchange is in flight; BeginParallelExchange() must "
                      "be called first");
    }

    /// Bring a packed send buffer to wherever the communicator will read it.
    NEK_FORCE_INLINE void SyncSendBufferForComm(ParallelChannel &c)
    {
#if defined(NEKTAR_ENABLE_SYCL)
        // SYCL has no legacy-stream semantics: queue 0 gives no ordering
        // against the per-block queues that wrote the trace the pack read,
        // so join every queue before anything - the D2H fetch or MPI itself
        // - consumes the send buffer.
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            nekDeviceSynchronize();
        }
#endif
        if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
            !m_rowComm->IsGPUAware())
        {
            // Without GPU-aware MPI the message is sent from the host, so the
            // pack has to be brought back before it is posted. Fetching the
            // host pointer is what performs the copy.
            c.sendBuffer.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        }
        else if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            // Draining the legacy stream is enough on CUDA and HIP: the pack
            // kernel ran in stream 0, and launching there already made it
            // wait on every blocking per-block stream, so its completion
            // means the whole chain that produced this buffer is done.
            nekStreamSynchronize(0);
        }
    }

    /// Push a received buffer back to wherever the gather will read it.
    NEK_FORCE_INLINE void SyncRecvBufferAfterComm(ParallelChannel &c)
    {
        if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
            !m_rowComm->IsGPUAware())
        {
            c.recvBuffer.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
        }
    }

    /**
     * @brief Exchange @p numComp components of @p inPtr on @p chan and wait.
     *
     * For a caller with no local work to overlap the messages with. The
     * advection velocity is the case this exists for: it is gathered onto the
     * global trace once at setup, and a parallel trace needs its neighbour's
     * half of that average exactly as it needs the neighbour's state, but
     * there is nothing useful to be doing meanwhile.
     */
    NEK_FORCE_INLINE void ExchangeParallelTracesBlocking(const unsigned chan,
                                                         const TData *inPtr,
                                                         const unsigned numComp)
    {
        if (m_parT0.empty())
        {
            return;
        }

        // Same stream discipline as the split Begin/End pair.
        Nektar::LoopExecutionSetStreamID(0);

        auto &c = m_parChannel[chan];

        SetUpParallelComm(chan, numComp);
        PackParallelSendBuffer(chan, inPtr, numComp);
        SyncSendBufferForComm(c);

        c.comm->BeginComm();
        c.inFlight = true;
        c.comm->EndComm();
        c.inFlight = false;

        SyncRecvBufferAfterComm(c);
    }

    /**
     * @brief Where one parallel trace's data is read from and written to when
     * the send buffer is packed, in the execution space.
     *
     * The pack runs one thread per (entry, component) pair, so like the gather
     * kernels it cannot reach a member; everything it needs arrives through
     * these.
     */
    struct ParallelSendData
    {
        LibUtilities::MemoryRegion<size_t> offset_st;
        LibUtilities::MemoryRegion<size_t> offset;
        LibUtilities::MemoryRegion<size_t> compSize;
        /// Points per component this entry sends.
        LibUtilities::MemoryRegion<size_t> npts;
        /// Start of this entry's payload in the send buffer.
        LibUtilities::MemoryRegion<size_t> bufOffset;
        size_t numEntry = 0;
    };
    ParallelSendData m_parSendDev;

    /**
     * @brief Build the exchange, and the buffers it runs over, for @p numComp
     * components per trace point.
     *
     * Deferred out of the constructor because the component count is a
     * property of the field being applied to, not of the mesh: the same
     * operator serves a scalar and a five-component compressible state. Cheap
     * to call on every apply, and rebuilds only when the count changes, in the
     * manner of SetWorkSpace().
     *
     * This is also where the backward side of every parallel trace finally
     * learns its address. #m_parT1 was built by the constructor with its
     * offsets left at zero, because they point into the receive buffer, whose
     * layout is the running sum of per-entry payloads and so depends on
     * @p numComp.
     */
    void SetUpParallelComm(const unsigned chan, const unsigned numComp)
    {
        auto &c = m_parChannel[chan];

        if (m_parT0.empty() || c.numComp == numComp)
        {
            return;
        }

        ASSERTL1(!c.inFlight,
                 "The buffers an exchange is reading and writing cannot be "
                 "replaced while it is in flight");

        c.numComp = numComp;

        const size_t numEntry = m_parSendPts.size();

        std::vector<size_t> sendCounts(numEntry), recvCounts(numEntry);
        for (size_t e = 0; e < numEntry; ++e)
        {
            sendCounts[e] = m_parSendPts[e] * numComp;
            recvCounts[e] = m_parRecvPts[e] * numComp;
        }

        c.comm = std::make_unique<MultiRegions::AssemblyComm<TData>>(
            m_rowComm, m_parSharedEntries);

        const size_t nSend =
            std::accumulate(sendCounts.begin(), sendCounts.end(), size_t{0});
        const size_t nRecv =
            std::accumulate(recvCounts.begin(), recvCounts.end(), size_t{0});

        c.sendBuffer = LibUtilities::MemoryRegion<TData>(nSend, eHostPinned);
        c.recvBuffer = LibUtilities::MemoryRegion<TData>(nRecv, eHostPinned);

        // Pointers are taken once and handed to the persistent requests, so
        // they are fetched in whichever space the communicator will read them
        // from and not synchronised again until the exchange begins.
        TData *sendPtr, *recvPtr;
        if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
            m_rowComm->IsGPUAware())
        {
            c.sendBuffer.template Initialize<NektarSpaces::DeviceSpace>(0.0);
            c.recvBuffer.template Initialize<NektarSpaces::DeviceSpace>(0.0);
            sendPtr =
                (TData *)c.sendBuffer
                    .template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
            recvPtr =
                (TData *)c.recvBuffer
                    .template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
        }
        else
        {
            c.sendBuffer.template Initialize<NektarSpaces::HostSpace>(0.0);
            c.recvBuffer.template Initialize<NektarSpaces::HostSpace>(0.0);
            sendPtr = (TData *)c.sendBuffer
                          .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            recvPtr = (TData *)c.recvBuffer
                          .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        }

        c.comm->InitSendRecvCommsVar(sendPtr, sendCounts, recvPtr, recvCounts);
    }

    /// Mirror the send-buffer pack tables into the execution space. Built
    /// once: every entry is described in points, so the tables serve any
    /// component count.
    void PackParallelSendLocations()
    {
        m_parSendDev.numEntry = m_parSendPts.size();
        m_parSendDev.offset_st =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                m_parSendOffsetSt);
        m_parSendDev.offset =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                m_parSendOffset);
        m_parSendDev.compSize =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                m_parSendCompSize);
        m_parSendDev.npts =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                m_parSendPts);
        m_parSendDev.bufOffset =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                m_parSendPtsOffset);
    }

    /**
     * @brief Copy this rank's side of every parallel trace into a send buffer.
     *
     * Element-local order throughout, unreoriented and uninterpolated: the
     * receiving rank knows this trace's orientation and point distribution,
     * having been told them when the connectivity was resolved, and applies
     * both itself. Doing it here instead would need this rank to know the
     * neighbour's global trace, which is exactly the knowledge the exchange
     * exists to avoid needing.
     */
    NEK_FORCE_INLINE void PackParallelSendBuffer(const unsigned chan,
                                                 const TData *inPtr,
                                                 const unsigned numComp)
    {
        const size_t numEntry = m_parSendDev.numEntry;

        const auto offset_st =
            m_parSendDev.offset_st.template GetPtr<MemSpace, ReadOnly>();
        const auto offset =
            m_parSendDev.offset.template GetPtr<MemSpace, ReadOnly>();
        const auto compSize =
            m_parSendDev.compSize.template GetPtr<MemSpace, ReadOnly>();
        const auto npts =
            m_parSendDev.npts.template GetPtr<MemSpace, ReadOnly>();
        const auto bufOffset =
            m_parSendDev.bufOffset.template GetPtr<MemSpace, ReadOnly>();

        TData *sendPtr = m_parChannel[chan]
                             .sendBuffer.template GetPtr<MemSpace, WriteOnly>();

        PackParallelSendBlock<ExecSpace>(numEntry, numComp, offset_st, offset,
                                         compSize, npts, bufOffset, inPtr,
                                         sendPtr);
    }

    /**
     * @brief Refresh the per-block scratch parameters for parallel block @p b.
     *
     * SetInteriorParams() for a trace whose backward side is remote. Same
     * arithmetic, reading #m_parT0 and #m_parT1 - the backward side is
     * described in exactly the same terms whether it was gathered or received.
     *
     * @tparam TRACEDIM Trace dimension; must equal #m_traceDim.
     * @param  b        Parallel block index.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void SetParallelParams([[maybe_unused]] const unsigned b)
    {
        if constexpr (TRACEDIM == 0)
        {
            m_npT0[0] = m_npT1[0] = m_npT[0] = m_npTot = 1;
        }
        else if constexpr (TRACEDIM == 1)
        {
            m_T0Collocated[0]    = (m_parT0[b].interpFwd[0] == nullptr);
            m_T0CollocatedBwd[0] = (m_parT0[b].interpBwd[0] == nullptr);
            m_T1Collocated[0]    = (m_parT1[b].interpFwd[0] == nullptr);
            m_T1CollocatedBwd[0] = (m_parT1[b].interpBwd[0] == nullptr);

            // trace sizes
            m_npT0[0] = m_parT0[b].npts[0];
            m_npT1[0] = m_parT1[b].npts[0];

            ASSERTL1(m_T0Collocated[0] || m_T1Collocated[0],
                     "One side must be collocated");

            m_npT[0] = m_T0Collocated[0] ? m_npT0[0] : m_npT1[0];
            m_npTot  = m_npT[0];
        }
        else if constexpr (TRACEDIM == 2)
        {
            m_T0Collocated[0]    = (m_parT0[b].interpFwd[0] == nullptr);
            m_T0CollocatedBwd[0] = (m_parT0[b].interpBwd[0] == nullptr);
            m_T0Collocated[1]    = (m_parT0[b].interpFwd[1] == nullptr);
            m_T0CollocatedBwd[1] = (m_parT0[b].interpBwd[1] == nullptr);
            m_T1Collocated[0]    = (m_parT1[b].interpFwd[0] == nullptr);
            m_T1CollocatedBwd[0] = (m_parT1[b].interpBwd[0] == nullptr);
            m_T1Collocated[1]    = (m_parT1[b].interpFwd[1] == nullptr);
            m_T1CollocatedBwd[1] = (m_parT1[b].interpBwd[1] == nullptr);

            // trace sizes
            m_npT0[0] = m_parT0[b].npts[0];
            m_npT0[1] = m_parT0[b].npts[1];
            m_npT1[0] = m_parT1[b].npts[0];
            m_npT1[1] = m_parT1[b].npts[1];

            ASSERTL1((m_T0Collocated[0] && m_T0Collocated[1]) ||
                         (m_T1Collocated[0] && m_T1Collocated[1]),
                     "One trace must be collocated");
            bool T0IsCollocated =
                (m_T0Collocated[0] && m_T0Collocated[1]) ? true : false;

            // one side has to be collocated and so choose the size of
            // collocated trace to define global trace details
            m_npT[0] = T0IsCollocated ? m_npT0[0] : m_npT1[0];
            m_npT[1] = T0IsCollocated ? m_npT0[1] : m_npT1[1];
            m_npTot  = m_npT[0] * m_npT[1];
        }
    }

    /**
     * @brief Gather both sides of every parallel trace in block @p b into the
     * packed global-trace layout.
     *
     * GetInteriorTraces() with the backward side read from the receive buffer
     * rather than from the local trace field. Nothing else changes: the
     * neighbour ships its trace points in its own element-local order, which
     * is the order the reorientation kernels expect, and #m_parT1 carries the
     * orientation and interpolation that map it onto this rank's global trace
     * just as #m_intT1 does for a local neighbour.
     *
     * SetParallelParams() must have been called for the same block, and the
     * exchange must have completed - see EndParallelExchange().
     *
     * @tparam TRACEDIM Trace dimension; must equal #m_traceDim.
     * @param  b        Parallel block index.
     * @param  numflux  Number of components per trace point.
     * @param  inPtr    Physical-space input in element-local block layout.
     * @param  gloT0    Forward-side output, usually #m_gloT0.
     * @param  gloT1    Backward-side output, usually #m_gloT1.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void GetParallelTraces(const unsigned b,
                                            const unsigned numflux,
                                            const TData *inPtr, TData *gloT0,
                                            TData *gloT1,
                                            const unsigned chan = 0)
    {
        ASSERTL1(!m_parChannel[chan].inFlight,
                 "EndParallelExchange() must be called before the received "
                 "traces are read");

        const auto T0 = GetTraceBlockView(m_parT0Dev[b]);
        const auto T1 = GetTraceBlockView(m_parT1Dev[b]);

        const auto npTBlock = m_parT0[b].m_nTraceXnPtsPad;

        const TData *recvPtr =
            m_parChannel[chan].recvBuffer.template GetPtr<MemSpace, ReadOnly>();

        if constexpr (TRACEDIM == 0)
        {
            LocPointToGloPointBlock<ExecSpace>(numflux, T0, inPtr, gloT0,
                                               npTBlock);
            LocPointToGloPointBlock<ExecSpace>(numflux, T1, recvPtr, gloT1,
                                               npTBlock);
        }
        else if constexpr (TRACEDIM == 1)
        {
            LocEdgeToGloEdgeBlock<ExecSpace>(
                numflux, T0, m_parT0[b].interpFwd[0], m_npT0[0], inPtr, m_loc,
                m_npTot, gloT0, npTBlock, m_T0Collocated[0]);

            LocEdgeToGloEdgeBlock<ExecSpace>(
                numflux, T1, m_parT1[b].interpFwd[0], m_npT1[0], recvPtr, m_loc,
                m_npTot, gloT1, npTBlock, m_T1Collocated[0]);
        }
        else if constexpr (TRACEDIM == 2)
        {
            LocFaceToGloFaceBlock<ExecSpace>(
                numflux, T0, m_parT0[b].interpFwd[0], m_parT0[b].interpFwd[1],
                m_npT0[0], m_npT0[1], inPtr, m_loc, m_wsp, m_npT[0], m_npT[1],
                gloT0, npTBlock, m_T0Collocated[0], m_T0Collocated[1]);

            LocFaceToGloFaceBlock<ExecSpace>(
                numflux, T1, m_parT1[b].interpFwd[0], m_parT1[b].interpFwd[1],
                m_npT1[0], m_npT1[1], recvPtr, m_loc, m_wsp, m_npT[0], m_npT[1],
                gloT1, npTBlock, m_T1Collocated[0], m_T1Collocated[1]);
        }
    }

    /**
     * @brief Scatter the computed parallel flux from #m_flux back to the local
     * trace layout.
     *
     * InterpBackInteriorFlux() with the backward half dropped. The neighbour
     * owns the other element and scatters the same flux to it from its own
     * side, having computed it independently from the same two states; there
     * is nothing to send back.
     *
     * @tparam TRACEDIM Trace dimension; must equal #m_traceDim.
     * @tparam APPEND   Accumulate into @p fluxPtr rather than overwriting it.
     * @param  b        Parallel block index.
     * @param  numflux  Number of components per trace point.
     * @param  fluxPtr  Output in element-local block layout.
     */
    template <unsigned TRACEDIM, bool APPEND>
    NEK_FORCE_INLINE void InterpBackParallelFlux(const unsigned b,
                                                 const unsigned numflux,
                                                 TData *fluxPtr)
    {
        const auto T0 = GetTraceBlockView(m_parT0Dev[b]);

        const auto npTBlock = m_parT0[b].m_nTraceXnPtsPad;

        if constexpr (TRACEDIM == 0)
        {
            GloPointToLocPointBlock<APPEND, ExecSpace>(
                numflux, T0, npTBlock, TData(1.0), m_flux, fluxPtr);
        }
        else if constexpr (TRACEDIM == 1)
        {
            // The global extent and the local one, which are not the same
            // number: the local element may carry a lower order than the
            // trace, and on a partition boundary it is as likely to be
            // the forward side that does as the backward. Passing the
            // global extent for both, as the interior edge scatter does,
            // is only right while T0 is collocated.
            GloEdgeToLocEdgeBlock<APPEND, ExecSpace, false>(
                numflux, T0, m_parT0[b].interpBwd[0], m_npT[0], m_flux,
                npTBlock, m_loc, m_npT0[0], fluxPtr, m_T0CollocatedBwd[0]);
        }
        else if constexpr (TRACEDIM == 2)
        {
            GloFaceToLocFaceBlock<APPEND, ExecSpace, false>(
                numflux, T0, m_parT0[b].interpBwd[0], m_parT0[b].interpBwd[1],
                m_npT[0], m_npT[1], m_flux, npTBlock, m_loc, m_wsp, m_npT0[0],
                m_npT0[1], fluxPtr, m_T0CollocatedBwd[0], m_T0CollocatedBwd[1]);
        }
    }

    /**
     * @brief Refresh the per-block scratch parameters for boundary block @p b.
     *
     * The counterpart of SetInteriorParams() for Dirichlet boundary traces.
     * Only the interior side can require interpolation here, so the global
     * trace size is taken from the boundary trace itself (#m_bndNptsT1)
     * without the collocation test the interior case needs.
     *
     * @tparam TRACEDIM Trace dimension; must equal #m_traceDim.
     * @param  b        Dirichlet boundary block index.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void SetBoundaryParams([[maybe_unused]] const unsigned b)
    {
        if constexpr (TRACEDIM == 0)
        {
            m_npT0[0] = m_npT1[0] = m_npT[0] = m_npTot = 1;
        }
        else if constexpr (TRACEDIM == 1)
        {
            m_T0Collocated[0]    = (m_bndT0[b].interpFwd[0] == nullptr);
            m_T0CollocatedBwd[0] = (m_bndT0[b].interpBwd[0] == nullptr);

            // trace sizes
            m_npT0[0] = m_bndT0[b].npts[0];
            m_npT1[0] = m_bndNptsT1[b][0];

            // trace always collocated
            m_npTot = m_npT[0] = m_npT1[0];
        }
        else if constexpr (TRACEDIM == 2)
        {
            m_T0Collocated[0]    = (m_bndT0[b].interpFwd[0] == nullptr);
            m_T0CollocatedBwd[0] = (m_bndT0[b].interpBwd[0] == nullptr);
            m_T0Collocated[1]    = (m_bndT0[b].interpFwd[1] == nullptr);
            m_T0CollocatedBwd[1] = (m_bndT0[b].interpBwd[1] == nullptr);

            // trace sizes
            m_npT0[0] = m_bndT0[b].npts[0];
            m_npT0[1] = m_bndT0[b].npts[1];
            m_npT1[0] = m_bndNptsT1[b][0];
            m_npT1[1] = m_bndNptsT1[b][1];

            // trace always collocated
            m_npT[0] = m_npT1[0];
            m_npT[1] = m_npT1[1];
            m_npTot  = m_npT[0] * m_npT[1];
        }
    }

    /**
     * @brief Gather the interior side of every Dirichlet boundary trace in
     * block @p b into the packed global-trace layout.
     *
     * The boundary counterpart of GetInteriorTraces(), and structured the
     * same way: one launch per block, one thread per (trace, component)
     * pair, with the per-trace metadata reaching the kernel through a
     * TraceBlockView rather than the host `std::vector`s.
     * Only the T0 side is touched; the exterior comes from GetDirBCTrace().
     *
     * SetBoundaryParams() must have been called for the same block first.
     *
     * @tparam TRACEDIM Trace dimension; must equal #m_traceDim.
     * @param  b        Dirichlet boundary block index.
     * @param  numflux  Number of components per trace point.
     * @param  inPtr    Physical-space input in element-local block layout.
     * @param  gloT0    Interior-side output, usually #m_gloT0.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void GetLocalBndTrace(const unsigned b,
                                           const unsigned numflux,
                                           const TData *inPtr, TData *gloT0)
    {
        const auto T0 = GetTraceBlockView(m_bndT0Dev[b]);

        const auto npTBlock = m_bndT0[b].m_nTraceXnPtsPad;

        if constexpr (TRACEDIM == 0)
        {
            // component packed vector which is padded in AVX case
            LocPointToGloPointBlock<ExecSpace>(numflux, T0, inPtr, gloT0,
                                               npTBlock);
        }
        else if constexpr (TRACEDIM == 1)
        {
            LocEdgeToGloEdgeBlock<ExecSpace>(
                numflux, T0, m_bndT0[b].interpFwd[0], m_npT0[0], inPtr, m_loc,
                m_npTot, gloT0, npTBlock, m_T0Collocated[0]);
        }
        else if constexpr (TRACEDIM == 2)
        {
            LocFaceToGloFaceBlock<ExecSpace>(
                numflux, T0, m_bndT0[b].interpFwd[0], m_bndT0[b].interpFwd[1],
                m_npT0[0], m_npT0[1], inPtr, m_loc, m_wsp, m_npT[0], m_npT[1],
                gloT0, npTBlock, m_T0Collocated[0], m_T0Collocated[1]);
        }
    }

    /**
     * @brief Fill the exterior side of boundary block @p b.
     *
     * The counterpart of GetLocalBndTrace() for the other side of the trace.
     * With @p FillDirData the values come from `TraceFluxOp::m_BndCondOp`,
     * fetched by global trace id and reoriented forwards; without it the
     * exterior is set equal to the interior, giving a zero jump.
     *
     * Only components carrying a Dirichlet condition take an exterior value.
     * The operator also holds Neumann and Robin values, but those prescribe a
     * flux rather than a state, so imposing them here would be wrong; such
     * components get the zero-jump treatment instead, per component and per
     * trace, since one block can draw on regions of differing type.
     *
     * The trace-to-(block, offset, stride) mapping is resolved once by
     * PackBndBCLocations(), so the only host work left is refetching one base
     * pointer per boundary-data block, which must go through
     * MemoryRegion::GetPtr() to stay synchronised.
     *
     * @tparam TRACEDIM    Trace dimension; must equal #m_traceDim.
     * @tparam FillDirData Consult the boundary condition operator.
     * @param  b           Dirichlet boundary block index.
     * @param  numflux     Number of components per trace point.
     * @param  gloT0       Interior side, already gathered.
     * @param  gloT1       Exterior-side output, usually #m_gloT1.
     */
    template <unsigned TRACEDIM, bool FillDirData = true>
    NEK_FORCE_INLINE void GetDirBCTrace(const unsigned b,
                                        const unsigned numflux,
                                        const TData *gloT0, TData *gloT1)
    {
        const auto npTBlock = m_bndT0[b].m_nTraceXnPtsPad;
        const auto numBlock = m_bndT0[b].offset.size();

        if constexpr (!FillDirData)
        {
            // No boundary values for this quantity: copy the interior side
            // across so the trace carries no jump.
            CopyGloTraceFwdToBwd<ExecSpace>(numflux, numBlock, m_npTot,
                                            npTBlock, gloT0, gloT1);

            return;
        }

        auto &loc       = m_bndCondLoc[b];
        const auto *off = loc.offset.template GetPtr<MemSpace, ReadOnly>();

        // The trace-to-block mapping is precomputed; only the base pointers are
        // refetched, once per boundary-data block rather than once per trace,
        // so the memory manager still sees every access.
        //
        // When the block draws on a single storage block at a single stride,
        // take that base pointer straight from the operator: the kernel then
        // needs neither the per-trace blk/compOffset loads nor the dependent
        // load through the base-pointer array. GetBlockPtr() is still called
        // once per apply either way, so the synchronisation point is unchanged
        // - it just becomes finer grained, covering only the block being read.
        const bool singleBlock = loc.singleBlock;

        const unsigned singleBlockId = loc.singleBlockId;
        const TData *singleBlockBase =
            singleBlock ? this->m_BndCondOp->GetBlockPtr(singleBlockId)
                        : nullptr;
        const size_t compStride = loc.compStride;

        const TData *const *bases = nullptr;
        const unsigned *blk       = nullptr;
        const size_t *cOff        = nullptr;

        if (!singleBlock)
        {
            {
                auto hostBases =
                    m_bndCondBase
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned k = 0; k < this->m_BndCondOp->GetNumBlocks(); ++k)
                {
                    hostBases[k] = this->m_BndCondOp->GetBlockPtr(k);
                }
            }

            bases = m_bndCondBase.template GetPtr<MemSpace, ReadOnly>();
            blk   = loc.blk.template GetPtr<MemSpace, ReadOnly>();
            cOff  = loc.compOffset.template GetPtr<MemSpace, ReadOnly>();
        }

        ASSERTL1(numflux <= m_numBCComp,
                 "More flux components than the boundary condition operator "
                 "holds, so their condition type cannot be resolved.");

        const auto *bcType = m_bcTypeMask.template GetPtr<MemSpace, ReadOnly>();
        const auto numBCComp = m_numBCComp;
        const unsigned dirCode =
            static_cast<unsigned>(SpatialDomains::eDirichlet);

        const auto npTot = m_npTot;
        const auto np0   = m_npT1[0];
        const auto np1   = m_npT1[1];

        FillDirBCTraceBlock<ExecSpace, TRACEDIM>(
            numBlock, numflux, singleBlock, singleBlockId, singleBlockBase,
            compStride, blk, off, cOff, bases, bcType, numBCComp, dirCode,
            npTot, npTBlock, np0, np1, gloT0, gloT1);
    }

    /**
     * @brief Put the interior state of every boundary trace into the boundary
     * storage.
     *
     * The gather is the same one GetLocalBndTrace() does for the interior side
     * of the flux, reorientation included, so what lands in the storage is in
     * the orientation an evaluated boundary condition would have been. A
     * caller can then transform it pointwise - reversing momentum for a
     * no-slip wall, say - with no knowledge of either layout.
     *
     * Every boundary trace is seeded, not only the ones a caller cares about;
     * regions it does not transform are overwritten again by the next
     * UpdateBndPhys() or simply carry values it will not read.
     */
    void v_RefreshBndCondTypes() override
    {
        if (!this->m_BndCondOp)
        {
            return;
        }

        const unsigned numStore = this->m_BndCondOp->GetNumBlocks();
        std::vector<unsigned> bcType(numStore * m_numBCComp,
                                     SpatialDomains::eNotDefined);
        for (unsigned k = 0; k < numStore; ++k)
        {
            for (unsigned c = 0; c < m_numBCComp; ++c)
            {
                bcType[k * m_numBCComp + c] = static_cast<unsigned>(
                    this->m_BndCondOp->GetBndCondType(k, c));
            }
        }
        m_bcTypeMask =
            LibUtilities::MemoryRegion<unsigned>::template FromVector<MemSpace>(
                bcType);
    }

    void v_GatherBndInteriorState(
        LibUtilities::Field<TData, FieldState::Phys> &trace,
        const std::vector<bool> &ownedBlocks) override
    {
        if (!this->m_BndCondOp || m_bndT0.empty())
        {
            return;
        }

        const unsigned numComp = trace.GetNumComponents();

        auto tracePtr =
            trace.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();

        {
            auto hostBases =
                m_bndCondBaseW
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned k = 0; k < this->m_BndCondOp->GetNumBlocks(); ++k)
            {
                hostBases[k] = this->m_BndCondOp->UpdateBlockPtr(k);
            }
        }
        TData *const *bases =
            m_bndCondBaseW.template GetPtr<MemSpace, ReadOnly>();

        SetWorkSpace(numComp);

        std::vector<unsigned> ownedFlags(this->m_BndCondOp->GetNumBlocks(), 0u);
        for (size_t k = 0; k < ownedFlags.size() && k < ownedBlocks.size(); ++k)
        {
            ownedFlags[k] = ownedBlocks[k] ? 1u : 0u;
        }
        auto ownedDev =
            LibUtilities::MemoryRegion<unsigned>::template FromVector<MemSpace>(
                ownedFlags);

        for (unsigned b = 0; b < m_bndT0.size(); ++b)
        {
            switch (m_traceDim)
            {
                case 0:
                    SetBoundaryParams<0>(b);
                    GetLocalBndTrace<0>(b, numComp, tracePtr, m_gloT0);
                    break;
                case 1:
                    SetBoundaryParams<1>(b);
                    GetLocalBndTrace<1>(b, numComp, tracePtr, m_gloT0);
                    break;
                case 2:
                    SetBoundaryParams<2>(b);
                    GetLocalBndTrace<2>(b, numComp, tracePtr, m_gloT0);
                    break;
            }

            const auto npTBlock = m_bndT0[b].m_nTraceXnPtsPad;
            const auto numBlock = m_bndT0[b].offset.size();
            const auto npTot    = m_npTot;

            auto &loc       = m_bndCondLoc[b];
            const auto *blk = loc.blk.template GetPtr<MemSpace, ReadOnly>();
            const auto *off = loc.offset.template GetPtr<MemSpace, ReadOnly>();
            const auto *cOff =
                loc.compOffset.template GetPtr<MemSpace, ReadOnly>();

            const TData *gloT0 = m_gloT0;
            const auto *owned  = ownedDev.template GetPtr<MemSpace, ReadOnly>();

            ScatterGloTraceToBndStore<ExecSpace>(numComp, numBlock, npTot,
                                                 npTBlock, blk, off, cOff,
                                                 owned, bases, gloT0);
        }
    }

    /**
     * @brief Fill the exterior gradient of boundary block @p b from the
     * Neumann conditions.
     *
     * The gradient counterpart of GetDirBCTrace(). A Neumann condition
     * prescribes \f$f = \partial u/\partial n\f$, which constrains the
     * gradient rather than the state, so it is imposed here on the exterior
     * side of the gradient trace. Components that are not Neumann keep the
     * exterior gradient equal to the interior one, as before.
     *
     * ### Why a reflected state rather than \f$\nabla u = f\,n\f$
     *
     * DiffuseScalarTraceFluxKernel forms the gradient term from the *average*
     * of the two sides, \f$n\cdot D\,\tfrac{1}{2}(g_f + g_b)\f$, with weights
     * of one half on a boundary as much as in the interior (the boundary-only
     * weighting is the disabled `#if 0` branch of that kernel). Setting the
     * exterior gradient to \f$f\,n\f$ directly would therefore give
     *
     * \f[ n\cdot\tfrac{1}{2}(g_f + f\,n)
     *      = \tfrac{1}{2}\left(\left.\frac{\partial u}{\partial n}\right|_{int}
     *        + f\right), \f]
     *
     * i.e. the prescribed flux blended half and half with the interior one
     * instead of enforced, and the tangential part of the average zeroed with
     * it. Reflecting the normal component instead,
     *
     * \f[ g_b = g_f + 2\,(f - n\cdot g_f)\,n, \f]
     *
     * makes the average \f$g_f + (f - n\cdot g_f)\,n\f$, whose normal
     * component is exactly \f$f\f$ while its tangential component stays at the
     * interior value. The kernel contracts the average with
     * \f$\sum_n n_n D_{nd}\f$, which is only parallel to \f$n\f$ for isotropic
     * \f$D\f$, so keeping the tangential part matters once \f$D\f$ is a tensor.
     *
     * The two forms agree wherever the interior normal gradient has already
     * converged on \f$f\f$, since the blend is then between two equal values.
     * They part company exactly where it has not - under-resolved regions, and
     * early transients started from data inconsistent with the boundary
     * condition - which is where the blended form degrades the imposed
     * condition to a half-weighted one. The accompanying tests use exact steady
     * solutions and so cannot tell the two apart; separating them needs a
     * convergence study rather than a single-mesh comparison.
     *
     * @note With the condition written on the gradient, the resulting flux term
     *       is \f$n^{T} D\,\nabla u = f\,(n^{T} D\,n)\f$, which equals \f$f\f$
     *       only when \f$n^{T} D\,n = 1\f$. That holds for isotropic \f$D\f$
     *       with unit normals; a tensor \f$D\f$ needs the condition restated on
     *       the flux instead.
     *
     * @tparam TRACEDIM   Trace dimension; must equal #m_traceDim.
     * @param  b          Boundary block index.
     * @param  numflux    Number of solution components, not gradient ones.
     * @param  ndim       Coordinate dimension.
     * @param  norms      Trace normals for this block, #m_norms layout.
     * @param  gloDerivT0 Interior gradient, already gathered.
     * @param  gloDerivT1 Exterior-side gradient output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void GetNeuBCTrace(const unsigned b,
                                        const unsigned numflux,
                                        const unsigned ndim, const TData *norms,
                                        const TData *gloDerivT0,
                                        TData *gloDerivT1)
    {
        const auto npTBlock = m_bndT0[b].m_nTraceXnPtsPad;
        const auto numBlock = m_bndT0[b].offset.size();

        auto &loc       = m_bndCondLoc[b];
        const auto *off = loc.offset.template GetPtr<MemSpace, ReadOnly>();

        // Same base-pointer handling as GetDirBCTrace(): resolved per apply so
        // the memory manager sees the access, and collapsed to a single pointer
        // when the block draws on one storage block at one stride.
        const bool singleBlock       = loc.singleBlock;
        const unsigned singleBlockId = loc.singleBlockId;
        const TData *singleBlockBase =
            singleBlock ? this->m_BndCondOp->GetBlockPtr(singleBlockId)
                        : nullptr;
        const size_t compStride = loc.compStride;

        const TData *const *bases = nullptr;
        const unsigned *blk       = nullptr;
        const size_t *cOff        = nullptr;

        if (!singleBlock)
        {
            {
                auto hostBases =
                    m_bndCondBase
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned k = 0; k < this->m_BndCondOp->GetNumBlocks(); ++k)
                {
                    hostBases[k] = this->m_BndCondOp->GetBlockPtr(k);
                }
            }

            bases = m_bndCondBase.template GetPtr<MemSpace, ReadOnly>();
            blk   = loc.blk.template GetPtr<MemSpace, ReadOnly>();
            cOff  = loc.compOffset.template GetPtr<MemSpace, ReadOnly>();
        }

        ASSERTL1(numflux <= m_numBCComp,
                 "More flux components than the boundary condition operator "
                 "holds, so their condition type cannot be resolved.");

        const auto *bcType = m_bcTypeMask.template GetPtr<MemSpace, ReadOnly>();
        const auto numBCComp = m_numBCComp;
        const unsigned neuCode =
            static_cast<unsigned>(SpatialDomains::eNeumann);

        const auto npTot = m_npTot;

        FillNeuBCDerivBlock<ExecSpace>(
            numBlock, numflux, ndim, singleBlock, singleBlockId,
            singleBlockBase, compStride, blk, off, cOff, bases, bcType,
            numBCComp, neuCode, npTot, npTBlock, norms, gloDerivT0, gloDerivT1);
    }

    /**
     * @brief Gather both sides of every Dirichlet boundary trace in block
     * @p b into the packed global-trace layout.
     *
     * A thin wrapper over GetLocalBndTrace() for the interior side and
     * GetDirBCTrace() for the exterior, kept so callers have one entry point.
     *
     * @tparam TRACEDIM    Trace dimension; must equal #m_traceDim.
     * @tparam FillDirData When false, the boundary condition operator is not
     *                     consulted and the exterior state is set equal to the
     *                     interior state, giving a zero jump across the trace.
     *                     Used for quantities the boundary condition operator
     *                     does not hold values for: `AdvDiffTraceFluxCFEOpImpl`
     *                     gathers the solution with `FillDirData` true and the
     *                     gradient with it false.
     * @param  b           Dirichlet boundary block index.
     * @param  numflux     Number of components per trace point.
     * @param  inPtr       Physical-space input in element-local block layout.
     * @param  gloT0       Interior-side output, usually #m_gloT0.
     * @param  gloT1       Exterior-side output, usually #m_gloT1.
     */
    template <unsigned TRACEDIM, bool FillDirData = true>
    NEK_FORCE_INLINE void GetBoundaryTraces(const unsigned b,
                                            const unsigned numflux,
                                            const TData *inPtr, TData *gloT0,
                                            TData *gloT1)
    {
        GetLocalBndTrace<TRACEDIM>(b, numflux, inPtr, gloT0);

        GetDirBCTrace<TRACEDIM, FillDirData>(b, numflux, gloT0, gloT1);
    }

    /**
     * @brief Gather both sides of the gradient on every boundary trace in
     * block @p b, imposing the Neumann conditions on the exterior side.
     *
     * The gradient counterpart of GetBoundaryTraces(): GetLocalBndTrace() for
     * the interior side and GetNeuBCTrace() for the exterior. Callers that hold
     * no Neumann data should keep using `GetBoundaryTraces<TRACEDIM, false>`,
     * which leaves the exterior gradient equal to the interior one throughout.
     *
     * @tparam TRACEDIM   Trace dimension; must equal #m_traceDim.
     * @param  b          Boundary block index.
     * @param  numflux    Number of solution components, not gradient ones.
     * @param  ndim       Coordinate dimension.
     * @param  inPtr      Physical-space gradient in element-local block layout.
     * @param  norms      Trace normals for this block, #m_norms layout.
     * @param  gloT0      Interior-side output.
     * @param  gloT1      Exterior-side output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void GetNeumannBoundaryTraces(
        const unsigned b, const unsigned numflux, const unsigned ndim,
        const TData *inPtr, const TData *norms, TData *gloT0, TData *gloT1)
    {
        GetLocalBndTrace<TRACEDIM>(b, numflux * ndim, inPtr, gloT0);

        GetNeuBCTrace<TRACEDIM>(b, numflux, ndim, norms, gloT0, gloT1);
    }

    /**
     * @brief Scatter the computed boundary flux from #m_flux back to the
     * element-local trace layout.
     *
     * The counterpart of InterpBackInteriorFlux() for Dirichlet boundary
     * traces. Only the interior side is written, and the flux is not negated:
     * there is no second element to balance.
     *
     * @tparam TRACEDIM Trace dimension; must equal #m_traceDim.
     * @tparam APPEND   Accumulate into @p fluxPtr rather than overwriting it.
     * @param  b        Dirichlet boundary block index.
     * @param  numflux  Number of components per trace point.
     * @param  fluxPtr  Output in element-local block layout.
     */
    // fill fwd & bwd interior  global traces
    template <unsigned TRACEDIM, bool APPEND>
    NEK_FORCE_INLINE void InterpBackDirichletFlux(const unsigned b,
                                                  const unsigned numflux,
                                                  TData *fluxPtr)
    {
        const auto T0 = GetTraceBlockView(m_bndT0Dev[b]);

        const auto npTBlock = m_bndT0[b].m_nTraceXnPtsPad;

        if constexpr (TRACEDIM == 0)
        {
            GloPointToLocPointBlock<APPEND, ExecSpace>(
                numflux, T0, npTBlock, TData(1.0), m_flux, fluxPtr);
        }
        else if constexpr (TRACEDIM == 1)
        {
            GloEdgeToLocEdgeBlock<APPEND, ExecSpace, false>(
                numflux, T0, m_bndT0[b].interpBwd[0], m_npT[0], m_flux,
                npTBlock, m_loc, m_npT0[0], fluxPtr, m_T0CollocatedBwd[0]);
        }
        else if constexpr (TRACEDIM == 2)
        {
            GloFaceToLocFaceBlock<APPEND, ExecSpace, false>(
                numflux, T0, m_bndT0[b].interpBwd[0], m_bndT0[b].interpBwd[1],
                m_npT[0], m_npT[1], m_flux, npTBlock, m_loc, m_wsp, m_npT0[0],
                m_npT0[1], fluxPtr, m_T0CollocatedBwd[0], m_T0CollocatedBwd[1]);
        }
    }
};

/**
 * @brief Intermediate base for scalar diffusion trace-flux operators.
 *
 * Adds no state or behaviour of its own; it exists so that diffusion
 * trace-flux implementations such as `DiffusionScalarIPTraceFluxOpImpl` share
 * a distinct base type. The constructor is protected, so the class can only be
 * reached through a derived operator.
 *
 * @tparam ExecSpace Execution space the operator runs in.
 * @tparam TData     Floating-point representation used by the field data.
 */
template <typename ExecSpace, typename TData>
class ScalarDiffusionTraceFluxOpImpl : public TraceFluxOpImpl<ExecSpace, TData>
{

protected:
    ScalarDiffusionTraceFluxOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : TraceFluxOpImpl<ExecSpace, TData>(expansionList, components)
    {
    }

    ~ScalarDiffusionTraceFluxOpImpl() override = default;
};

} // namespace Nektar::SolverCore::detail
