///////////////////////////////////////////////////////////////////////////////
//
// File: AeroForcesOpImpl.hpp
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
// Description: Serial/AVX and Device implementations of AeroForcesOp.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

#include <SolverCore/AeroForces/AeroForcesKernels.hpp>
#include <SolverCore/AeroForces/AeroForcesOp.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class AeroForcesOpImpl : public AeroForcesOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AeroForcesOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : AeroForcesOp<TData>(expansionList, components)
    {
    }

    static std::string className;

    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AeroForcesOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    struct VolumeLocation
    {
        unsigned int block = 0;
        size_t element     = 0;
        LocalRegions::ExpansionSharedPtr expansion;
    };

    /// Restriction of boundary quadrature points from the elements of one
    /// block, as it is gathered at configuration: point @c i has the
    /// entries from rowOffset[i] up to rowOffset[i + 1], each reading
    /// quadrature point sourcePoint of element sourceElement of the block
    /// with its weight. A point whose restriction is a gather has one
    /// entry of weight one.
    struct RestrictionHost
    {
        std::vector<size_t> rowOffset{0};
        std::vector<size_t> sourceElement;
        std::vector<size_t> sourcePoint;
        std::vector<TData> weight;
    };

    /// The boundary quadrature points on the elements of one block, as they
    /// are gathered at configuration, with the normals and moment arms
    /// stored point-major.
    struct BlockBoundaryHost
    {
        size_t nPressureSrc = 0;
        size_t nGradientSrc = 0;
        RestrictionHost pressure;
        RestrictionHost gradient;
        std::vector<TData> weightedNormals;
        std::vector<TData> momentArms;
    };

    /// The boundary quadrature points on the elements of one block, in the
    /// operator's memory space, for a velocity gradient at one interleave
    /// width. The points are sorted by the storage offset of their first
    /// gradient source at that width, so that neighbouring threads read
    /// neighbouring entries of the gradient wherever the boundary allows.
    /// Each source is held as its storage offset within a component: that
    /// of the gradient at the width the points are sorted for, and that of
    /// the pressure at interleave width one (first) and at
    /// m_implInterleaveWidth (second), so that the pressure is read at
    /// either width. The normals and moment arms are stored point-minor,
    /// component @c d of point @c p at <tt>d * nPoints + p</tt>.
    struct BlockBoundary
    {
        size_t nPoints = 0;
        LibUtilities::MemoryRegion<size_t> pressureRowOffset;
        std::array<LibUtilities::MemoryRegion<size_t>, 2> pressureSource;
        LibUtilities::MemoryRegion<TData> pressureWeight;
        LibUtilities::MemoryRegion<size_t> gradientRowOffset;
        LibUtilities::MemoryRegion<size_t> gradientSource;
        LibUtilities::MemoryRegion<TData> gradientWeight;
        LibUtilities::MemoryRegion<TData> weightedNormals;
        LibUtilities::MemoryRegion<TData> momentArms;
    };

    /// Interleave width of the execution space: an input block is read at
    /// this width or at width one.
    static constexpr unsigned int m_implInterleaveWidth =
        NektarSpaces::vector_width<ExecSpace, TData>::value;

    MultiRegions::ExpListSharedPtr m_pressureExpansion;
    /// The boundary points of each block, indexed by block, as they are
    /// gathered at configuration.
    std::vector<BlockBoundaryHost> m_boundaryHost;
    /// The boundary points of each block, indexed by block, for a velocity
    /// gradient at interleave width one (first) and at m_implInterleaveWidth
    /// (second); empty until a gradient is first read at that width.
    std::vector<std::array<BlockBoundary, 2>> m_boundary;
    std::array<unsigned int, 3> m_velocityComponents{};
    unsigned int m_pressureComponent = 0;
    unsigned int m_nDim              = 0;
    TData m_rho                      = TData(1);
    TData m_mu                       = TData(1);

    std::map<int, VolumeLocation> BuildVolumeLocations(
        const MultiRegions::ExpListSharedPtr &expansion)
    {
        std::map<int, VolumeLocation> locations;
        const auto blockAttributes =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansion);
        for (unsigned int block = 0; block < blockAttributes.size(); ++block)
        {
            auto collection = MultiRegions::GetCollection(expansion, block);
            const auto &elements = collection.GetExpVector();
            for (size_t element = 0; element < elements.size(); ++element)
            {
                locations[elements[element]->GetGeom()->GetGlobalID()] = {
                    block, element, elements[element]};
            }
        }
        return locations;
    }

    static bool CompressRestriction(const std::vector<TData> &restriction,
                                    const size_t nSrc, const size_t nDst,
                                    std::vector<size_t> &pointMap)
    {
        const TData tolerance =
            TData(1000) * std::numeric_limits<TData>::epsilon();
        pointMap.resize(nDst);
        for (size_t point = 0; point < nDst; ++point)
        {
            size_t source = 0;
            TData largest = TData(0);
            for (size_t q = 0; q < nSrc; ++q)
            {
                const TData value = std::abs(restriction[point * nSrc + q]);
                if (value > largest)
                {
                    largest = value;
                    source  = q;
                }
            }

            if (std::abs(restriction[point * nSrc + source] - TData(1)) >
                tolerance)
            {
                pointMap.clear();
                return false;
            }
            for (size_t q = 0; q < nSrc; ++q)
            {
                if (q != source &&
                    std::abs(restriction[point * nSrc + q]) > tolerance)
                {
                    pointMap.clear();
                    return false;
                }
            }
            pointMap[point] = source;
        }
        return true;
    }

    static std::vector<TData> BuildRestriction(
        const LocalRegions::ExpansionSharedPtr &volume,
        const unsigned int trace,
        const LocalRegions::ExpansionSharedPtr &boundary)
    {
        // Probe the exact oriented restriction once during configuration.
        // Apply only uses the resulting device-resident gather/sparse map.
        const size_t nSrc = volume->GetTotPoints();
        const size_t nDst = boundary->GetTotPoints();
        std::vector<TData> restriction(nDst * nSrc, TData(0));
        Array<OneD, NekDouble> source(nSrc, 0.0);
        Array<OneD, NekDouble> destination(nDst, 0.0);
        for (size_t q = 0; q < nSrc; ++q)
        {
            source[q] = 1.0;
            volume->GetTracePhysVals(trace, boundary, source, destination);
            for (size_t point = 0; point < nDst; ++point)
            {
                restriction[point * nSrc + q] =
                    static_cast<TData>(destination[point]);
            }
            source[q] = 0.0;
        }
        return restriction;
    }

    /// Append the rows of the restriction of one boundary face, read from
    /// element @p element of a block, to @p host: one entry of weight one per
    /// point if the restriction is a gather, else the nonzero entries of
    /// each row.
    static void AppendRestriction(const std::vector<TData> &restriction,
                                  const size_t nSrc, const size_t nDst,
                                  const size_t element, RestrictionHost &host)
    {
        std::vector<size_t> pointMap;
        const bool gather =
            CompressRestriction(restriction, nSrc, nDst, pointMap);
        for (size_t point = 0; point < nDst; ++point)
        {
            if (gather)
            {
                host.sourceElement.push_back(element);
                host.sourcePoint.push_back(pointMap[point]);
                host.weight.push_back(TData(1));
            }
            else
            {
                [[maybe_unused]] const size_t rowStart = host.weight.size();
                for (size_t q = 0; q < nSrc; ++q)
                {
                    const TData weight = restriction[point * nSrc + q];
                    if (weight != TData(0))
                    {
                        host.sourceElement.push_back(element);
                        host.sourcePoint.push_back(q);
                        host.weight.push_back(weight);
                    }
                }
                ASSERTL1(host.weight.size() > rowStart,
                         "AeroForces encountered an empty boundary "
                         "restriction row.");
            }
            host.rowOffset.push_back(host.weight.size());
        }
    }

    /// Storage offset of quadrature point @p point of element @p element
    /// within a component of a block of interleave width @p interleaveWidth
    /// whose elements have @p nPoints quadrature points.
    static size_t StorageOffset(const size_t element, const size_t point,
                                const size_t nPoints,
                                const size_t interleaveWidth)
    {
        return (element / interleaveWidth) * interleaveWidth * nPoints +
               point * interleaveWidth + element % interleaveWidth;
    }

    /// Copy the boundary points of one block to the operator's memory space,
    /// sorted and laid out for a velocity gradient at interleave width
    /// @p gradientInterleaveWidth.
    static BlockBoundary UploadBlockBoundary(
        const BlockBoundaryHost &host, const unsigned int nDim,
        const unsigned int gradientInterleaveWidth)
    {
        BlockBoundary boundary;
        const size_t nPoints = host.gradient.rowOffset.size() - 1;
        boundary.nPoints     = nPoints;

        const auto &gradient = host.gradient;
        const auto &pressure = host.pressure;
        auto gradientOffset  = [&](const size_t entry) {
            return StorageOffset(gradient.sourceElement[entry],
                                  gradient.sourcePoint[entry], host.nGradientSrc,
                                  gradientInterleaveWidth);
        };

        std::vector<size_t> order(nPoints);
        std::iota(order.begin(), order.end(), size_t(0));
        std::stable_sort(order.begin(), order.end(),
                         [&](const size_t a, const size_t b) {
                             return gradientOffset(gradient.rowOffset[a]) <
                                    gradientOffset(gradient.rowOffset[b]);
                         });

        std::vector<size_t> gradientRowOffset{0};
        std::vector<size_t> gradientSource;
        std::vector<TData> gradientWeight;
        std::vector<size_t> pressureRowOffset{0};
        std::array<std::vector<size_t>, 2> pressureSource;
        std::vector<TData> pressureWeight;
        std::vector<TData> weightedNormals(nDim * nPoints);
        std::vector<TData> momentArms(3 * nPoints);
        const std::array<unsigned int, 2> interleaveWidths{
            1u, m_implInterleaveWidth};
        for (size_t i = 0; i < nPoints; ++i)
        {
            const size_t point = order[i];
            for (size_t entry = gradient.rowOffset[point];
                 entry < gradient.rowOffset[point + 1]; ++entry)
            {
                gradientSource.push_back(gradientOffset(entry));
                gradientWeight.push_back(gradient.weight[entry]);
            }
            gradientRowOffset.push_back(gradientSource.size());

            for (size_t entry = pressure.rowOffset[point];
                 entry < pressure.rowOffset[point + 1]; ++entry)
            {
                for (unsigned int w = 0; w < interleaveWidths.size(); ++w)
                {
                    pressureSource[w].push_back(
                        StorageOffset(pressure.sourceElement[entry],
                                      pressure.sourcePoint[entry],
                                      host.nPressureSrc, interleaveWidths[w]));
                }
                pressureWeight.push_back(pressure.weight[entry]);
            }
            pressureRowOffset.push_back(pressureWeight.size());

            for (unsigned int d = 0; d < nDim; ++d)
            {
                weightedNormals[d * nPoints + i] =
                    host.weightedNormals[point * nDim + d];
            }
            for (unsigned int d = 0; d < 3; ++d)
            {
                momentArms[d * nPoints + i] = host.momentArms[point * 3 + d];
            }
        }

        using SizeRegion           = LibUtilities::MemoryRegion<size_t>;
        using TDataRegion          = LibUtilities::MemoryRegion<TData>;
        boundary.pressureRowOffset = SizeRegion::template FromVector<MemSpace>(
            "AeroForces pressure rows", pressureRowOffset);
        for (unsigned int w = 0; w < interleaveWidths.size(); ++w)
        {
            boundary.pressureSource[w] =
                SizeRegion::template FromVector<MemSpace>(
                    "AeroForces pressure sources", pressureSource[w]);
        }
        boundary.pressureWeight = TDataRegion::template FromVector<MemSpace>(
            "AeroForces pressure weights", pressureWeight);
        boundary.gradientRowOffset = SizeRegion::template FromVector<MemSpace>(
            "AeroForces gradient rows", gradientRowOffset);
        boundary.gradientSource = SizeRegion::template FromVector<MemSpace>(
            "AeroForces gradient sources", gradientSource);
        boundary.gradientWeight = TDataRegion::template FromVector<MemSpace>(
            "AeroForces gradient weights", gradientWeight);
        boundary.weightedNormals = TDataRegion::template FromVector<MemSpace>(
            "AeroForces weighted normals", weightedNormals);
        boundary.momentArms = TDataRegion::template FromVector<MemSpace>(
            "AeroForces moment arms", momentArms);
        return boundary;
    }

    void v_Configure(const std::vector<unsigned int> &velocityComponents,
                     const MultiRegions::ExpListSharedPtr &pressureExpansion,
                     const unsigned int pressureComponent,
                     const std::vector<bool> &selectedBoundaryRegions,
                     const TData rho, const TData mu,
                     const std::array<TData, 3> &pivot) override
    {
        m_pressureExpansion = pressureExpansion;
        m_nDim              = this->m_expansionList->GetCoordim(0);
        ASSERTL0(m_nDim == 2 || m_nDim == 3,
                 "AeroForces supports only 2D and 3D expansions.");
        ASSERTL0(this->m_expansionList->GetShapeDimension() == m_nDim,
                 "AeroForces does not support embedded-manifold expansions.");
        ASSERTL0(this->m_components.size() >= m_nDim,
                 "AeroForces requires one velocity component per dimension.");
        ASSERTL0(velocityComponents.size() == m_nDim,
                 "AeroForces velocity selection has the wrong dimension.");
        ASSERTL0(pressureExpansion,
                 "AeroForces requires a pressure expansion.");
        std::copy(velocityComponents.begin(), velocityComponents.end(),
                  m_velocityComponents.begin());
        m_pressureComponent = pressureComponent;
        m_rho               = rho;
        m_mu                = mu;

        const auto &boundaryExpansions =
            this->m_expansionList->GetBndCondExpansions();
        ASSERTL0(selectedBoundaryRegions.size() == boundaryExpansions.size(),
                 "AeroForces boundary selection size mismatch.");

        const auto velocityLocations =
            BuildVolumeLocations(this->m_expansionList);
        const auto pressureLocations =
            BuildVolumeLocations(m_pressureExpansion);
        Array<OneD, int> boundaryToElement;
        Array<OneD, int> boundaryToTrace;
        this->m_expansionList->GetBoundaryToElmtMap(boundaryToElement,
                                                    boundaryToTrace);

        const size_t nBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                this->m_expansionList)
                .size();
        m_boundaryHost.assign(nBlocks, BlockBoundaryHost{});
        size_t boundaryOffset = 0;
        for (unsigned int region = 0; region < boundaryExpansions.size();
             ++region)
        {
            auto boundaryList = boundaryExpansions[region];
            if (!selectedBoundaryRegions[region])
            {
                boundaryOffset += boundaryList->GetExpSize();
                continue;
            }

            Array<OneD, Array<OneD, NekDouble>> normals;
            this->m_expansionList->GetBoundaryNormals(region, normals);
            for (size_t face = 0; face < boundaryList->GetExpSize(); ++face)
            {
                const size_t mapIndex = boundaryOffset + face;
                const int elementId   = boundaryToElement[mapIndex];
                const int traceId     = boundaryToTrace[mapIndex];
                const auto velocityElement =
                    this->m_expansionList->GetExp(elementId);
                const int geometry = velocityElement->GetGeom()->GetGlobalID();
                const auto velocityLocation = velocityLocations.at(geometry);
                const auto pressureLocationIt =
                    pressureLocations.find(geometry);
                ASSERTL0(pressureLocationIt != pressureLocations.end(),
                         "AeroForces could not match a pressure element to a "
                         "selected velocity boundary.");
                const auto pressureLocation = pressureLocationIt->second;
                ASSERTL0(pressureLocation.block == velocityLocation.block &&
                             pressureLocation.element ==
                                 velocityLocation.element,
                         "AeroForces requires the pressure to share the "
                         "blocks of the velocity.");
                const int boundaryGeometry =
                    velocityElement->GetGeom()->GetTid(traceId);
                bool pressureTraceFound      = false;
                unsigned int pressureTraceId = 0;
                for (unsigned int pressureTrace = 0;
                     pressureTrace < pressureLocation.expansion->GetNtraces();
                     ++pressureTrace)
                {
                    if (pressureLocation.expansion->GetGeom()->GetTid(
                            pressureTrace) == boundaryGeometry)
                    {
                        pressureTraceFound = true;
                        pressureTraceId    = pressureTrace;
                        break;
                    }
                }
                ASSERTL0(pressureTraceFound,
                         "AeroForces pressure and velocity boundary topology "
                         "differs.");

                const auto boundary       = boundaryList->GetExp(face);
                const size_t nGradientSrc = velocityElement->GetTotPoints();
                const size_t nPressureSrc =
                    pressureLocation.expansion->GetTotPoints();
                const size_t nDst = boundary->GetTotPoints();

                auto &host        = m_boundaryHost[velocityLocation.block];
                host.nGradientSrc = nGradientSrc;
                host.nPressureSrc = nPressureSrc;
                AppendRestriction(
                    BuildRestriction(velocityElement, traceId, boundary),
                    nGradientSrc, nDst, velocityLocation.element,
                    host.gradient);
                AppendRestriction(BuildRestriction(pressureLocation.expansion,
                                                   pressureTraceId, boundary),
                                  nPressureSrc, nDst, pressureLocation.element,
                                  host.pressure);

                Array<OneD, NekDouble> one(nDst, 1.0);
                Array<OneD, NekDouble> metric(nDst, 0.0);
                boundary->MultiplyByQuadratureMetric(one, metric);
                const size_t boundaryPhysOffset =
                    boundaryList->GetPhys_Offset(face);
                for (size_t point = 0; point < nDst; ++point)
                {
                    for (unsigned int d = 0; d < m_nDim; ++d)
                    {
                        host.weightedNormals.push_back(static_cast<TData>(
                            metric[point] *
                            normals[d][boundaryPhysOffset + point]));
                    }
                }

                Array<OneD, NekDouble> x(nDst, 0.0), y(nDst, 0.0), z(nDst, 0.0);
                boundary->GetCoords(x, y, z);
                for (size_t point = 0; point < nDst; ++point)
                {
                    host.momentArms.push_back(static_cast<TData>(x[point]) -
                                              pivot[0]);
                    host.momentArms.push_back(static_cast<TData>(y[point]) -
                                              pivot[1]);
                    host.momentArms.push_back(static_cast<TData>(z[point]) -
                                              pivot[2]);
                }
            }
            boundaryOffset += boundaryList->GetExpSize();
        }

        // The points of each block are copied to the operator's memory space
        // when a gradient is first read at each interleave width.
        m_boundary.clear();
        m_boundary.resize(nBlocks);

        this->m_result = LibUtilities::MemoryRegion<TData>(
            "AeroForces result", AeroForcesOp<TData>::resultSize, eHostPinned);
    }

    template <unsigned int DIMENSION>
    void ApplyBlocks(
        LibUtilities::Field<TData, FieldState::Phys> &pressure,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient)
    {
        // Each block integrates the boundary points on its elements on its
        // own stream; a block with no selected boundary launches nothing.
        for (unsigned int blk = 0; blk < m_boundary.size(); ++blk)
        {
            auto &pressureBlock = pressure.GetBlocks()[blk];
            auto &gradientBlock = velocityGradient.GetBlocks()[blk];
            const unsigned int pressureInterleaveWidth =
                pressureBlock.GetInterleaveWidth();
            const unsigned int gradientInterleaveWidth =
                gradientBlock.GetInterleaveWidth();
            ASSERTL0((pressureInterleaveWidth == 1 ||
                      pressureInterleaveWidth == m_implInterleaveWidth) &&
                         (gradientInterleaveWidth == 1 ||
                          gradientInterleaveWidth == m_implInterleaveWidth),
                     "AeroForces reads its inputs at interleave width 1 or " +
                         std::to_string(m_implInterleaveWidth) + ".");

            const auto &host = m_boundaryHost[blk];
            if (host.gradient.rowOffset.size() == 1)
            {
                continue;
            }
            auto &boundary =
                m_boundary[blk][gradientInterleaveWidth == 1 ? 0 : 1];
            if (boundary.nPoints == 0)
            {
                boundary =
                    UploadBlockBoundary(host, m_nDim, gradientInterleaveWidth);
            }

            const unsigned int streamID = blk + 1;
            const auto pressurePtr =
                pressureBlock.template GetPtr<MemSpace, ReadOnly>(streamID) +
                m_pressureComponent * pressureBlock.CompSize();
            const auto gradientPtr =
                gradientBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto result =
                this->m_result.template GetPtr<MemSpace, ReadWrite>(streamID);

            AeroForcesKernel<DIMENSION, ExecSpace>(
                boundary.nPoints,
                boundary.pressureRowOffset.template GetPtr<MemSpace, ReadOnly>(
                    streamID),
                boundary.pressureSource[pressureInterleaveWidth == 1 ? 0 : 1]
                    .template GetPtr<MemSpace, ReadOnly>(streamID),
                boundary.pressureWeight.template GetPtr<MemSpace, ReadOnly>(
                    streamID),
                boundary.gradientRowOffset.template GetPtr<MemSpace, ReadOnly>(
                    streamID),
                boundary.gradientSource.template GetPtr<MemSpace, ReadOnly>(
                    streamID),
                boundary.gradientWeight.template GetPtr<MemSpace, ReadOnly>(
                    streamID),
                boundary.weightedNormals.template GetPtr<MemSpace, ReadOnly>(
                    streamID),
                boundary.momentArms.template GetPtr<MemSpace, ReadOnly>(
                    streamID),
                pressurePtr, gradientPtr, gradientBlock.CompSize(),
                m_velocityComponents[0], m_velocityComponents[1],
                m_velocityComponents[2], m_rho, m_mu, result, streamID);
        }
    }

    void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &pressure,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient) override
    {
        ASSERTL1(pressure.GetNumHomoModes() == 1,
                 "AeroForces does not support homogeneous expansions.");
        ASSERTL1(m_pressureComponent < pressure.GetNumComponents(),
                 "AeroForces pressure component is out of range.");
        ASSERTL1(pressure.GetBlocks().size() == m_boundary.size() &&
                     velocityGradient.GetBlocks().size() == m_boundary.size(),
                 "AeroForces block count changed after configuration.");
        // Each block is integrated on its own stream, between the zeroing of
        // the result and its reduction on the null stream.
        this->m_result.template Initialize<MemSpace>(TData{0});

        switch (m_nDim)
        {
            case 2:
                ApplyBlocks<2>(pressure, velocityGradient);
                break;
            case 3:
                ApplyBlocks<3>(pressure, velocityGradient);
                break;
            default:
                ASSERTL0(false, "AeroForces supports only dimensions 2 and 3.");
        }

        auto comm = this->m_expansionList->GetComm();
        comm->template Reduce<MemSpace>(this->m_result, LibUtilities::ReduceSum,
                                        0);
    }
};

} // namespace Nektar::SolverCore::detail
