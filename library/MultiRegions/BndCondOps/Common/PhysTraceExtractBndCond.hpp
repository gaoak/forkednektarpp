///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractBndCond.hpp
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
// Description: Extraction of a volume field onto the points of a named
// boundary region of a continuous field, built on PhysTraceExtractOp, for the
// boundary conditions that need volume data on their points.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/BndCondOps/Common/CGBndCondKernels.hpp>
#include <MultiRegions/BndCondOps/DirBndCond/DirBndCondKernels.hpp>
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp>

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>

#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Extracts a volume field onto the points of a boundary region of a
 * continuous field.
 *
 * @details
 * PhysTraceExtractOp::ExtractTrace() extracts one local trace for every
 * element of a block. A boundary region instead touches a subset of the
 * elements, through different local traces and orientations. At
 * construction this class
 * - matches every element of the boundary expansion list to the volume
 *   element owning it, by geometry id: both lists come from one mesh graph;
 * - sorts the matched elements into buckets of one shape, order, local
 *   trace, boundary block and volume block, and builds for every bucket a
 *   volume expansion list restricted to its elements, with its own
 *   PhysTraceExtractOp;
 * - extracts onto the Gauss-Lobatto trace points where the boundary
 *   expansion carries them, on the collapsed direction of a face of a
 *   tetrahedron, prism or pyramid (see
 *   PhysTraceExtractOp::SetUseGLLTracePoints());
 * - builds for every bucket the map from its extracted trace points to the
 *   boundary region's points, which reorients every face or edge.
 *
 * Apply() then gathers every bucket's elements from the volume field,
 * extracts their trace and scatters it onto the boundary field.
 *
 * All work runs on stream #m_opStreamID, the stream of the first block of
 * every bucket's PhysTraceExtractOp. Apply() does not synchronise the host:
 * it orders that stream after the work queued on the streams of the volume
 * and boundary blocks it reads and writes, and those streams after its own
 * work, with events.
 *
 * @tparam ExecSpace Execution space of the kernels.
 * @tparam TData     Floating-point type of the field data.
 */
template <typename ExecSpace, typename TData> class PhysTraceExtractBndCond
{
public:
    /**
     * @param volumeExpList Expansion list of the volume field.
     * @param components    Variables of the field.
     * @param bcExpList     Expansion list of the boundary region.
     * @param execStr       Execution space of the PhysTraceExtractOps.
     * @param graph         Mesh graph holding the expansions of the volume
     *                      elements. Defaults to that of @p volumeExpList,
     *                      which an expansion list built from an
     *                      ExpansionInfoMap does not have.
     */
    PhysTraceExtractBndCond(
        const MultiRegions::ExpListSharedPtr &volumeExpList,
        const std::vector<std::string> &components,
        const MultiRegions::ExpListSharedPtr &bcExpList,
        const std::string &execStr                      = "",
        const SpatialDomains::MeshGraphSharedPtr &graph = nullptr);

    /**
     * @brief Extract @p in onto @p out.
     *
     * @details
     * Work queued on the stream of a block of @p in or @p out before the
     * call runs before the extraction, and work queued on it after the call
     * runs after the extraction, without the host waiting on either.
     *
     * @param in  Volume field, laid out as
     *            MultiRegions::GetBlockAttributes(volumeExpList).
     * @param out Boundary field, laid out as
     *            MultiRegions::GetBlockAttributes(bcExpList).
     */
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &out);

protected:
    using MemSpace = typename ExecSpace::memory_space;

    /// Elements of one shape, order, local trace, boundary block and volume
    /// block, with the operator and storage to extract their trace.
    struct Bucket
    {
        std::shared_ptr<PhysTraceExtractOp<TData>> op;
        unsigned int traceId = 0;
        /// Block of the volume field the bucket gathers from.
        unsigned int volumeBlock = 0;
        /// Block of the boundary field the bucket scatters into.
        unsigned int bcBlock = 0;
        size_t nElmts        = 0;
        /// Number of trace points of one element.
        size_t nDst = 0;
        /// Number of volume points of one element.
        size_t nVolume = 0;
        /// Element of the volume block at every position of #volumePhys.
        std::vector<size_t> volumeElmt;
        /// Point of the volume block of every point of #volumePhys, by the
        /// interleave width of the volume block, built when a width is first
        /// seen and kept.
        std::map<unsigned int, LibUtilities::MemoryRegion<size_t>> gatherIndex;
        /// Point of the boundary block of every point of #tracePhys.
        LibUtilities::MemoryRegion<size_t> scatterIndex;
        /// The bucket's elements, laid out as the single block of the
        /// restricted expansion list of #op, and their extracted trace.
        LibUtilities::BlockAccessor<TData, FieldState::Phys> volumePhys;
        LibUtilities::BlockAccessor<TData, FieldState::Phys> tracePhys;

        Bucket(LibUtilities::BlockAccessor<TData, FieldState::Phys> &&volume,
               LibUtilities::BlockAccessor<TData, FieldState::Phys> &&trace)
            : volumePhys(std::move(volume)), tracePhys(std::move(trace))
        {
        }
    };

    /// Stream of the first block of every bucket's PhysTraceExtractOp, on
    /// which all the work runs.
    static constexpr unsigned int m_opStreamID = 1;

    std::vector<Bucket> m_buckets;
    /// Streams of the volume and boundary blocks Apply() reads and writes,
    /// one per block index: block @em b runs on stream @em b + 1.
    std::vector<unsigned int> m_blockStreams;

    /// Add to @p bucket.gatherIndex the index for a volume block at
    /// interleave width @p width.
    static void BuildGatherIndex(Bucket &bucket, const unsigned int width);
};

template <typename ExecSpace, typename TData>
PhysTraceExtractBndCond<ExecSpace, TData>::PhysTraceExtractBndCond(
    const MultiRegions::ExpListSharedPtr &volumeExpList,
    const std::vector<std::string> &components,
    const MultiRegions::ExpListSharedPtr &bcExpList, const std::string &execStr,
    const SpatialDomains::MeshGraphSharedPtr &graph)
{
    // The volume element owning every trace, by trace geometry id.
    struct VolumeFace
    {
        unsigned int block = 0;
        size_t elmt        = 0;
        unsigned int trace = 0;
        LocalRegions::ExpansionSharedPtr expansion;
    };
    std::map<int, VolumeFace> faceOwner;

    const auto volumeBlocks =
        MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
            volumeExpList);
    for (unsigned int blk = 0; blk < volumeBlocks.size(); ++blk)
    {
        auto collection = MultiRegions::GetCollection(volumeExpList, blk);
        const auto &expansions = collection.GetExpVector();
        for (unsigned int e = 0; e < expansions.size(); ++e)
        {
            for (unsigned int t = 0; t < expansions[e]->GetNtraces(); ++t)
            {
                faceOwner[expansions[e]->GetGeom()->GetTid(t)] = {
                    blk, e, t, expansions[e]};
            }
        }
    }

    // (shape, local trace, modes in the three directions, boundary block,
    // volume block).
    using BucketKey =
        std::tuple<LibUtilities::ShapeType, unsigned int, unsigned int,
                   unsigned int, unsigned int, unsigned int, unsigned int>;
    // One boundary element and the volume element owning it.
    struct MatchedFace
    {
        unsigned int ownerBlock = 0;
        size_t ownerElmt        = 0;
        int ownerGeomId         = -1;
        size_t bcElmtFlat       = 0;
        /// Boundary point of every extracted trace point.
        std::vector<int> idmap;
    };
    struct BucketBuild
    {
        std::vector<MatchedFace> faces;
        unsigned int traceId     = 0;
        unsigned int volumeBlock = 0;
        unsigned int bcBlock     = 0;
        size_t nDst              = 0;
        bool useGLL              = false;
    };
    std::map<BucketKey, BucketBuild> bucketBuild;

    const auto bcBlocks =
        bcExpList->GetNumElmts() == 0
            ? std::vector<LibUtilities::BlockAttributes<FieldState::Phys>>{}
            : MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  bcExpList);
    for (unsigned int blk = 0; blk < bcBlocks.size(); ++blk)
    {
        auto collection        = MultiRegions::GetCollection(bcExpList, blk);
        const auto &expansions = collection.GetExpVector();
        for (unsigned int e = 0; e < expansions.size(); ++e)
        {
            const auto bcElmt = expansions[e];
            const auto ownerIt =
                faceOwner.find(bcElmt->GetGeom()->GetGlobalID());
            if (ownerIt == faceOwner.end())
            {
                continue;
            }
            const auto &owner = ownerIt->second;
            const auto orient = owner.expansion->GetTraceOrient(owner.trace);

            // The boundary element carries Gauss-Lobatto points in every
            // direction. The owner's own trace points, which
            // GetLocTraceBlockAttributes() sizes the extraction for, are
            // Gauss-Radau on a collapsed direction, so there the
            // Gauss-Lobatto variant is extracted instead. The points are
            // then the boundary element's, and only their order differs:
            // a face is transposed or flipped as in
            // Expansion3D::v_GetTracePhysVals(), an edge is reversed.
            size_t nDst;
            std::vector<int> idmap;
            bool useGLL;
            if (bcElmt->GetNumBases() > 1)
            {
                int id0 = 0, id1 = 1;
                if (orient >= StdRegions::eDir1FwdDir2_Dir2FwdDir1)
                {
                    id0 = 1;
                    id1 = 0;
                }
                const int nq0 = bcElmt->GetNumPoints(id0);
                const int nq1 = bcElmt->GetNumPoints(id1);
                nDst = static_cast<size_t>(nq0) * static_cast<size_t>(nq1);

                const int ownerNq0 =
                    owner.expansion->GetTraceBasisKey(owner.trace, 0)
                        .GetNumPoints();
                const int ownerNq1 =
                    owner.expansion->GetTraceBasisKey(owner.trace, 1)
                        .GetNumPoints();
                useGLL = ownerNq0 != nq0 || ownerNq1 != nq1;

                // The backward map, local[q] = global[map[q]], gives the
                // boundary point of every extracted point. The forward map
                // is its inverse, which differs for a face rotated by 90
                // degrees and for a transposed face with nq0 != nq1.
                Array<OneD, int> map;
                owner.expansion->ReOrientTracePhysMap(orient, map, nq0, nq1,
                                                      false);
                idmap.assign(&map[0], &map[0] + nDst);
            }
            else
            {
                const int nq0 = bcElmt->GetNumPoints(0);
                nDst          = static_cast<size_t>(nq0);
                useGLL        = owner.expansion->GetTraceBasisKey(owner.trace)
                             .GetNumPoints() != nq0;

                idmap.resize(nDst);
                for (size_t q = 0; q < nDst; ++q)
                {
                    idmap[q] = static_cast<int>(
                        orient == StdRegions::eBackwards ? nDst - 1 - q : q);
                }
            }

            const unsigned int nm0 = owner.expansion->GetBasisNumModes(0);
            const unsigned int nm1 = owner.expansion->GetNumBases() > 1
                                         ? owner.expansion->GetBasisNumModes(1)
                                         : 0;
            const unsigned int nm2 = owner.expansion->GetNumBases() > 2
                                         ? owner.expansion->GetBasisNumModes(2)
                                         : 0;
            BucketKey key{owner.expansion->DetShapeType(),
                          owner.trace,
                          nm0,
                          nm1,
                          nm2,
                          blk,
                          owner.block};
            auto &build       = bucketBuild[key];
            build.traceId     = owner.trace;
            build.volumeBlock = owner.block;
            build.bcBlock     = blk;
            build.nDst        = nDst;
            // The key fixes the shape, order and trace, and with them
            // whether the Gauss-Lobatto variant is needed.
            ASSERTL1(build.faces.empty() || build.useGLL == useGLL,
                     "Elements of one bucket disagree on the trace points "
                     "to extract onto.");
            build.useGLL = useGLL;

            MatchedFace face;
            face.ownerBlock  = owner.block;
            face.ownerElmt   = owner.elmt;
            face.ownerGeomId = owner.expansion->GetGeom()->GetGlobalID();
            face.bcElmtFlat  = e;
            face.idmap       = std::move(idmap);
            build.faces.push_back(std::move(face));
        }
    }

    m_buckets.reserve(bucketBuild.size());
    const auto meshGraph = graph ? graph : volumeExpList->GetGraph();
    ASSERTL0(meshGraph, "PhysTraceExtractBndCond needs a mesh graph: pass "
                        "one when volumeExpList has none.");
    const auto &volumeExpansionInfo =
        meshGraph->GetExpansionInfo(components[0]);
    // Each bucket's storage is one block, built as Field builds its blocks
    // but belonging to no Field: its owner is null, and its memory region
    // allocates its own storage on first use (see BlockAccessor::GetPtr()).
    const unsigned int nComp = components.size();
    const auto makeBlock =
        [nComp](const std::string &name,
                const LibUtilities::BlockAttributes<FieldState::Phys> &attr) {
            return LibUtilities::BlockAccessor<TData, FieldState::Phys>(
                attr,
                LibUtilities::MemoryRegion<TData>(
                    name, attr.CompSize() * nComp, eHostPageable,
                    NektarSpaces::host_memory_alignment),
                nullptr, nComp, 1);
        };
    for (auto &[key, build] : bucketBuild)
    {
        // The volume expansion list restricted to the bucket's elements. It
        // orders them by geometry id, not in the order they were matched.
        SpatialDomains::ExpansionInfoMap restrictedInfo;
        for (const auto &face : build.faces)
        {
            const auto info = volumeExpansionInfo.find(face.ownerGeomId);
            ASSERTL1(info != volumeExpansionInfo.end(),
                     "No expansion for a volume element of the boundary.");
            restrictedInfo.emplace(*info);
        }

        auto restrictedExpList =
            MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                volumeExpList->GetSession(), restrictedInfo, false);
        restrictedExpList->SetDataWarehouse();

        const auto restrictedBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                restrictedExpList);
        // The bucket's elements come from one volume block, so they share
        // what splits a collection (basis, deformation, MAXSIZE) and the
        // restricted list is one block. Apply() relies on this.
        ASSERTL0(restrictedBlocks.size() == 1,
                 "The elements of a bucket must form one block.");

        // ExtractTrace() walks the element groups of the input block, which
        // is padded to the largest vector width, so the output block is
        // padded the same way.
        const size_t nElmts    = build.faces.size();
        const auto vectorWidth = NektarSpaces::max_vector_width<TData>::value;
        const size_t nElmtsWithPadding =
            ((nElmts + vectorWidth - 1) / vectorWidth) * vectorWidth;
        const LibUtilities::BlockAttributes<FieldState::Phys> traceBlockAttr(
            nElmts, nElmtsWithPadding, static_cast<unsigned int>(build.nDst),
            1);

        Bucket bucket(
            makeBlock("PhysTraceExtractBndCondVolume", restrictedBlocks[0]),
            makeBlock("PhysTraceExtractBndCondTrace", traceBlockAttr));
        bucket.traceId     = build.traceId;
        bucket.volumeBlock = build.volumeBlock;
        bucket.bcBlock     = build.bcBlock;
        bucket.nElmts      = nElmts;
        bucket.nDst        = build.nDst;

        bucket.op = PhysTraceExtractOp<TData>::Create(restrictedExpList,
                                                      components, execStr);
        if (build.useGLL)
        {
            bucket.op->SetUseGLLTracePoints(true);
        }

        // Position of every element in the restricted expansion list.
        std::map<int, size_t> geomIdToPos;
        {
            auto restrictedCollection =
                MultiRegions::GetCollection(restrictedExpList, 0);
            const auto &restrictedExpansions =
                restrictedCollection.GetExpVector();
            for (size_t p = 0; p < restrictedExpansions.size(); ++p)
            {
                geomIdToPos[restrictedExpansions[p]->GetGeom()->GetGlobalID()] =
                    p;
            }
        }

        // The bucket's elements are those of one volume block, with the
        // same points.
        bucket.nVolume = restrictedBlocks[0].GetNumData();
        ASSERTL1(volumeBlocks[bucket.volumeBlock].GetNumData() ==
                     bucket.nVolume,
                 "The elements of a bucket differ in their number of points "
                 "from those of their volume block.");

        bucket.volumeElmt.resize(bucket.nElmts);
        std::vector<size_t> scatterIndex(bucket.nElmts * bucket.nDst);
        for (const auto &face : build.faces)
        {
            const auto pos         = geomIdToPos.at(face.ownerGeomId);
            bucket.volumeElmt[pos] = face.ownerElmt;
            for (size_t q = 0; q < bucket.nDst; ++q)
            {
                scatterIndex[pos * bucket.nDst + q] =
                    face.bcElmtFlat * bucket.nDst +
                    static_cast<size_t>(face.idmap[q]);
            }
        }
        bucket.scatterIndex = LibUtilities::MemoryRegion<
            size_t>::template FromVector<MemSpace, size_t>(scatterIndex);

        m_buckets.push_back(std::move(bucket));
    }

    std::set<unsigned int> streamSet;
    for (const auto &bucket : m_buckets)
    {
        streamSet.insert(bucket.volumeBlock + 1);
        streamSet.insert(bucket.bcBlock + 1);
    }
    m_blockStreams.assign(streamSet.begin(), streamSet.end());
}

template <typename ExecSpace, typename TData>
void PhysTraceExtractBndCond<ExecSpace, TData>::Apply(
    LibUtilities::Field<TData, FieldState::Phys> &in,
    LibUtilities::Field<TData, FieldState::Phys> &out)
{
    const unsigned int nComp = in.GetNumComponents();
    ASSERTL0(in.GetNumHomoModes() == 1,
             "PhysTraceExtractBndCond does not support homogeneous fields.");
    ASSERTL0(out.GetNumComponents() == nComp &&
                 (m_buckets.empty() ||
                  m_buckets[0].volumePhys.GetNumComponents() >= nComp),
             "PhysTraceExtractBndCond: the input and output fields must have "
             "the same number of components, at most the number it was "
             "built for.");

    // Order #m_opStreamID after the work already queued on the streams of the
    // blocks read and written, on the device rather than on the host.
    JoinStreams<ExecSpace>(m_opStreamID, m_blockStreams);

    for (auto &bucket : m_buckets)
    {
        // Every kernel below indexes only into the block whose pointer it
        // takes: one block of in and out, volumePhys or tracePhys.
        auto &srcBlock = in.GetBlocks()[bucket.volumeBlock];
        // The interleave width of the input is known only here. An index is
        // built for every width seen and kept, so that none is overwritten
        // while a kernel of an earlier call may still read it.
        const unsigned int width = srcBlock.GetInterleaveWidth();
        if (bucket.gatherIndex.count(width) == 0)
        {
            BuildGatherIndex(bucket, width);
        }
        auto srcPtr =
            srcBlock.template GetPtr<MemSpace, ReadOnly>(m_opStreamID);
        auto volumePtr = bucket.volumePhys.template GetPtr<MemSpace, WriteOnly>(
            m_opStreamID);
        auto gatherIdx =
            bucket.gatherIndex.at(width).template GetPtr<MemSpace, ReadOnly>(
                m_opStreamID);

        for (unsigned int comp = 0; comp < nComp; ++comp)
        {
            CGBndCondGatherKernel<ExecSpace>(
                bucket.nElmts * bucket.nVolume, gatherIdx,
                srcPtr + comp * srcBlock.CompSize(),
                volumePtr + comp * bucket.volumePhys.CompSize(), m_opStreamID);
        }

        // Block 0 of the operator is the restricted list's only block.
        bucket.op->ExtractTrace(bucket.traceId, bucket.volumePhys,
                                bucket.tracePhys, 0, 0);

        auto &outBlock = out.GetBlocks()[bucket.bcBlock];
        // The scatter index addresses the boundary block at width 1.
        ASSERTL0(outBlock.GetInterleaveWidth() == 1,
                 "PhysTraceExtractBndCond writes a boundary field at "
                 "interleave width 1 only.");
        auto outPtr =
            outBlock.template GetPtr<MemSpace, WriteOnly>(m_opStreamID);
        auto tracePtr =
            bucket.tracePhys.template GetPtr<MemSpace, ReadOnly>(m_opStreamID);
        auto scatterIdx =
            bucket.scatterIndex.template GetPtr<MemSpace, ReadOnly>(
                m_opStreamID);

        for (unsigned int comp = 0; comp < nComp; ++comp)
        {
            DirBndCondKernel<ExecSpace>(
                bucket.nElmts * bucket.nDst, scatterIdx,
                tracePtr + comp * bucket.tracePhys.CompSize(),
                outPtr + comp * outBlock.CompSize(), m_opStreamID);
        }
    }

    // Order the work queued next on the streams of the blocks read and
    // written after the extraction.
    ReleaseStreams<ExecSpace>(m_opStreamID, m_blockStreams);
}

template <typename ExecSpace, typename TData>
void PhysTraceExtractBndCond<ExecSpace, TData>::BuildGatherIndex(
    Bucket &bucket, const unsigned int width)
{
    // Point q of element e lies at (e / width) * width * nVolume + q * width
    // + e % width of a block at interleave width @p width.
    const size_t nVolume = bucket.nVolume;
    std::vector<size_t> gatherIndex(bucket.nElmts * nVolume);
    for (size_t pos = 0; pos < bucket.nElmts; ++pos)
    {
        const size_t elmt  = bucket.volumeElmt[pos];
        const size_t group = elmt / width;
        const size_t lane  = elmt % width;
        for (size_t q = 0; q < nVolume; ++q)
        {
            gatherIndex[pos * nVolume + q] =
                group * width * nVolume + q * width + lane;
        }
    }
    bucket.gatherIndex.emplace(
        width, LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                       size_t>(
                   gatherIndex));
}

} // namespace Nektar::MultiRegions::detail
