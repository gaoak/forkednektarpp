///////////////////////////////////////////////////////////////////////////////
//
// File: HistoryPointsOpImpl.hpp
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
// Description: Serial/AVX and Device implementations of HistoryPointsOp.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

#include <SolverCore/HistoryPoints/HistoryPointsKernels.hpp>
#include <SolverCore/HistoryPoints/HistoryPointsOp.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class HistoryPointsOpImpl : public HistoryPointsOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    HistoryPointsOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                        const std::vector<std::string> &components)
        : HistoryPointsOp<TData>(expansionList, components)
    {
    }

    static std::string className;

    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<HistoryPointsOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    struct VolumeLocation
    {
        unsigned int block = 0;
        size_t element     = 0;
    };

    /// The history points in the elements of one block, in the order they
    /// are found, with the interpolation weights of each point stored
    /// point-major.
    struct BlockPointsHost
    {
        unsigned int nq0 = 1;
        unsigned int nq1 = 1;
        unsigned int nq2 = 1;
        std::vector<size_t> elements;
        std::vector<size_t> pointIds;
        std::array<std::vector<TData>, 3> weights;
    };

    /// The history points in the elements of one block, sorted by element.
    struct BlockPoints
    {
        size_t nPoints   = 0;
        unsigned int nq0 = 1;
        unsigned int nq1 = 1;
        unsigned int nq2 = 1;
        /// Element of the block holding each point, on the host.
        std::vector<size_t> elements;
        /// Storage offset of the first quadrature point of the element
        /// holding each point within a component, for a block at interleave
        /// width one (first) and at m_implInterleaveWidth (second); empty
        /// until a field is first read at that width.
        std::array<LibUtilities::MemoryRegion<size_t>, 2> elementOffset;
        LibUtilities::MemoryRegion<size_t> pointIds;
        /// Interpolation weights of each direction, stored point-minor.
        std::array<LibUtilities::MemoryRegion<TData>, 3> weights;
    };

    /// Interleave width of the execution space: an input block is read at
    /// this width or at width one.
    static constexpr unsigned int m_implInterleaveWidth =
        NektarSpaces::vector_width<ExecSpace, TData>::value;

    /// The points held by each block, indexed by block.
    std::vector<BlockPoints> m_blockPoints;
    size_t m_numPoints       = 0;
    unsigned int m_dimension = 0;

    std::map<int, VolumeLocation> BuildVolumeLocations() const
    {
        std::map<int, VolumeLocation> locations;
        const auto blockAttributes =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                this->m_expansionList);
        for (unsigned int block = 0; block < blockAttributes.size(); ++block)
        {
            auto collection =
                MultiRegions::GetCollection(this->m_expansionList, block);
            const auto &elements = collection.GetExpVector();
            for (size_t element = 0; element < elements.size(); ++element)
            {
                locations[elements[element]->GetGeom()->GetGlobalID()] = {
                    block, element};
            }
        }
        return locations;
    }

    static std::vector<TData> BuildWeights(
        const LibUtilities::BasisSharedPtr &basis, const TData coordinate)
    {
        const auto &points = basis->GetZ();
        const auto &bary   = basis->GetBaryWeights();
        std::vector<TData> weights(points.size(), TData(0));

        for (size_t i = 0; i < points.size(); ++i)
        {
            if (static_cast<TData>(points[i]) == coordinate)
            {
                weights[i] = TData(1);
                return weights;
            }
        }

        TData denominator = TData(0);
        for (size_t i = 0; i < points.size(); ++i)
        {
            weights[i] = static_cast<TData>(bary[i]) /
                         (static_cast<TData>(points[i]) - coordinate);
            denominator += weights[i];
        }
        ASSERTL1(std::abs(denominator) > std::numeric_limits<TData>::epsilon(),
                 "HistoryPoints encountered singular interpolation weights.");
        for (auto &weight : weights)
        {
            weight /= denominator;
        }
        return weights;
    }

    /// Sort the points of one block by element and copy them to the
    /// operator's memory space, with their weights stored point-minor.
    BlockPoints UploadBlockPoints(const BlockPointsHost &host) const
    {
        BlockPoints result;
        result.nPoints = host.pointIds.size();
        result.nq0     = host.nq0;
        result.nq1     = host.nq1;
        result.nq2     = host.nq2;

        std::vector<size_t> order(result.nPoints);
        std::iota(order.begin(), order.end(), size_t(0));
        std::stable_sort(order.begin(), order.end(),
                         [&host](const size_t a, const size_t b) {
                             return host.elements[a] < host.elements[b];
                         });

        result.elements.resize(result.nPoints);
        std::vector<size_t> pointIds(result.nPoints);
        for (size_t i = 0; i < result.nPoints; ++i)
        {
            result.elements[i] = host.elements[order[i]];
            pointIds[i]        = host.pointIds[order[i]];
        }
        result.pointIds =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace>(
                "HistoryPoints point IDs", pointIds);

        const std::array<unsigned int, 3> nq{host.nq0, host.nq1, host.nq2};
        for (unsigned int d = 0; d < m_dimension; ++d)
        {
            std::vector<TData> weights(nq[d] * result.nPoints);
            for (size_t i = 0; i < result.nPoints; ++i)
            {
                for (unsigned int q = 0; q < nq[d]; ++q)
                {
                    weights[q * result.nPoints + i] =
                        host.weights[d][order[i] * nq[d] + q];
                }
            }
            result.weights[d] =
                LibUtilities::MemoryRegion<TData>::template FromVector<
                    MemSpace>("HistoryPoints weights", weights);
        }
        return result;
    }

    /// Copy to the operator's memory space the storage offset of the first
    /// quadrature point of the element holding each point of @p points
    /// within a component, for a block at interleave width @p interleaveWidth.
    static LibUtilities::MemoryRegion<size_t> UploadElementOffsets(
        const BlockPoints &points, const size_t interleaveWidth)
    {
        const size_t nqTot = size_t(points.nq0) * points.nq1 * points.nq2;
        std::vector<size_t> elementOffset(points.nPoints);
        for (size_t i = 0; i < points.nPoints; ++i)
        {
            const size_t element = points.elements[i];
            elementOffset[i] =
                (element / interleaveWidth) * interleaveWidth * nqTot +
                element % interleaveWidth;
        }
        return LibUtilities::MemoryRegion<size_t>::template FromVector<
            MemSpace>("HistoryPoints element offsets", elementOffset);
    }

    void v_Configure(const std::vector<std::array<TData, 3>> &points) override
    {
        m_numPoints = points.size();
        ASSERTL0(m_numPoints > 0, "HistoryPoints requires at least one point.");

        m_dimension = this->m_expansionList->GetShapeDimension();
        ASSERTL0(m_dimension >= 1 && m_dimension <= 3,
                 "HistoryPoints supports dimensions 1, 2 and 3.");
        ASSERTL0(this->m_expansionList->GetCoordim(0) == m_dimension,
                 "HistoryPoints does not support embedded manifolds.");
        const size_t nBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                this->m_expansionList)
                .size();

        auto comm            = this->m_expansionList->GetComm();
        const int rank       = comm->GetRank();
        const auto locations = BuildVolumeLocations();
        std::vector<int> elementIds(m_numPoints, -1);
        std::vector<int> owners(m_numPoints, -1);
        std::vector<Array<OneD, NekDouble>> localCoordinates;
        localCoordinates.reserve(m_numPoints);

        for (size_t point = 0; point < m_numPoints; ++point)
        {
            Array<OneD, NekDouble> global(m_dimension, 0.0);
            Array<OneD, NekDouble> local(m_dimension, 0.0);
            for (unsigned int d = 0; d < m_dimension; ++d)
            {
                global[d] = points[point][d];
            }
            elementIds[point] = this->m_expansionList->GetExpIndex(
                global, local, NekConstants::kGeomFactorsTol);
            if (elementIds[point] >= 0)
            {
                owners[point] = rank;
                for (unsigned int d = 0; d < m_dimension; ++d)
                {
                    local[d] = std::max(-1.0, std::min(1.0, local[d]));
                }
            }
            localCoordinates.emplace_back(local);
        }

        comm->AllReduce(owners, LibUtilities::ReduceMax);
        std::vector<BlockPointsHost> blockPoints(nBlocks);
        for (size_t point = 0; point < m_numPoints; ++point)
        {
            ASSERTL0(owners[point] >= 0, "History point " +
                                             std::to_string(point) +
                                             " cannot be found in the mesh.");
            if (owners[point] != rank)
            {
                continue;
            }

            const int elementId = elementIds[point];
            ASSERTL1(elementId >= 0,
                     "HistoryPoints owner has no local element.");
            auto expansion       = this->m_expansionList->GetExp(elementId);
            const int geometryId = expansion->GetGeom()->GetGlobalID();
            const auto location  = locations.find(geometryId);
            ASSERTL1(location != locations.end(),
                     "HistoryPoints could not map an element to a block.");

            auto &host = blockPoints[location->second.block];
            host.nq0   = expansion->GetNumPoints(0);
            host.nq1   = m_dimension >= 2 ? expansion->GetNumPoints(1) : 1;
            host.nq2   = m_dimension == 3 ? expansion->GetNumPoints(2) : 1;
            host.elements.push_back(location->second.element);
            host.pointIds.push_back(point);

            Array<OneD, NekDouble> collapsed(m_dimension, 0.0);
            expansion->LocCoordToLocCollapsed(localCoordinates[point],
                                              collapsed);
            for (unsigned int d = 0; d < m_dimension; ++d)
            {
                const auto weights =
                    BuildWeights(expansion->GetBasis(d), collapsed[d]);
                host.weights[d].insert(host.weights[d].end(), weights.begin(),
                                       weights.end());
            }
        }

        m_blockPoints.clear();
        m_blockPoints.resize(nBlocks);
        for (size_t block = 0; block < nBlocks; ++block)
        {
            if (!blockPoints[block].pointIds.empty())
            {
                m_blockPoints[block] = UploadBlockPoints(blockPoints[block]);
            }
        }

        // Point-major, as the result of Apply().
        this->m_result = LibUtilities::MemoryRegion<TData>(
            "HistoryPoints result", m_numPoints * this->m_components.size(),
            eHostPinned);
    }

    template <unsigned int DIMENSION>
    void ApplyBlocks(LibUtilities::Field<TData, FieldState::Phys> &field)
    {
        // Each point is owned by one rank and evaluated by the block holding
        // it, on the block's stream; a block holding no point launches
        // nothing.
        for (unsigned int blk = 0; blk < m_blockPoints.size(); ++blk)
        {
            auto &points = m_blockPoints[blk];
            if (points.nPoints == 0)
            {
                continue;
            }

            const unsigned int streamID = blk + 1;
            auto &fieldBlock            = field.GetBlocks()[blk];
            const unsigned int interleaveWidth =
                fieldBlock.GetInterleaveWidth();
            ASSERTL0(interleaveWidth == 1 ||
                         interleaveWidth == m_implInterleaveWidth,
                     "HistoryPoints reads a field at interleave width 1 or " +
                         std::to_string(m_implInterleaveWidth) + ", not " +
                         std::to_string(interleaveWidth) + ".");

            auto &elementOffsetRegion =
                points.elementOffset[interleaveWidth == 1 ? 0 : 1];
            if (elementOffsetRegion.size() == 0)
            {
                elementOffsetRegion =
                    UploadElementOffsets(points, interleaveWidth);
            }
            const auto elementOffset =
                elementOffsetRegion.template GetPtr<MemSpace, ReadOnly>(
                    streamID);
            const auto pointIds =
                points.pointIds.template GetPtr<MemSpace, ReadOnly>(streamID);
            const auto weights0 =
                points.weights[0].template GetPtr<MemSpace, ReadOnly>(streamID);
            const TData *weights1 = nullptr;
            const TData *weights2 = nullptr;
            if constexpr (DIMENSION >= 2)
            {
                weights1 =
                    points.weights[1].template GetPtr<MemSpace, ReadOnly>(
                        streamID);
            }
            if constexpr (DIMENSION == 3)
            {
                weights2 =
                    points.weights[2].template GetPtr<MemSpace, ReadOnly>(
                        streamID);
            }
            const auto fieldPtr =
                fieldBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto result =
                this->m_result.template GetPtr<MemSpace, ReadWrite>(streamID);

            if (interleaveWidth == 1)
            {
                HistoryPointsKernel<DIMENSION, 1u, ExecSpace>(
                    points.nPoints, field.GetNumComponents(), points.nq0,
                    points.nq1, points.nq2, fieldBlock.CompSize(),
                    elementOffset, pointIds, weights0, weights1, weights2,
                    fieldPtr, result, streamID);
            }
            else
            {
                HistoryPointsKernel<DIMENSION, m_implInterleaveWidth,
                                    ExecSpace>(
                    points.nPoints, field.GetNumComponents(), points.nq0,
                    points.nq1, points.nq2, fieldBlock.CompSize(),
                    elementOffset, pointIds, weights0, weights1, weights2,
                    fieldPtr, result, streamID);
            }
        }
    }

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &field) override
    {
        ASSERTL1(field.GetNumHomoModes() == 1,
                 "HistoryPoints homogeneous expansions are not yet "
                 "supported.");
        ASSERTL1(field.GetNumComponents() == this->m_components.size(),
                 "HistoryPoints field component count changed.");
        ASSERTL1(field.GetBlocks().size() == m_blockPoints.size(),
                 "HistoryPoints field block layout changed.");

        // The points are evaluated on the block streams, between the zeroing
        // of the result and its collection on the null stream.
        this->m_result.template Initialize<MemSpace>(TData{0});

        switch (m_dimension)
        {
            case 1:
                ApplyBlocks<1>(field);
                break;
            case 2:
                ApplyBlocks<2>(field);
                break;
            case 3:
                ApplyBlocks<3>(field);
                break;
            default:
                ASSERTL0(false,
                         "HistoryPoints supports dimensions 1, 2 and 3.");
        }

        // Collect the point values on rank zero, the rank
        // FilterHistoryPoints writes from. Each point is owned by one rank,
        // which alone writes its entries; on every other rank they keep the
        // zero set above, so the sum over the ranks is the owner's value.
        // The sum is one collective over the whole result, in the operator's
        // own memory space - GPU-aware MPI when available - where a gather
        // would need per-rank counts and a reordering of the gathered values
        // into point order on rank zero. Every rank must make this call;
        // GetValues() copies the result to the host on rank zero.
        auto comm = this->m_expansionList->GetComm();
        comm->template Reduce<MemSpace>(this->m_result, LibUtilities::ReduceSum,
                                        0);
    }
};

} // namespace Nektar::SolverCore::detail
