///////////////////////////////////////////////////////////////////////////////
//
// File: TraceDataWarehouseDef.hpp
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

#pragma once

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <MultiRegions/DataWarehouse/TraceDataWarehouse.hpp>

#include <MultiRegions/DisContField.h>
#include <MultiRegions/ExpList.h>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::MultiRegions
{
namespace
{
template <typename TData>
std::vector<TData> GetIPTracePenaltyFactor(
    const ExpListSharedPtr &expansionList)
{
    auto tracelist     = expansionList->GetTrace();
    auto traceMap      = expansionList->GetTraceMap();
    auto &elmtToTrace  = traceMap->GetElmtToTrace();
    const int spaceDim = expansionList->GetCoordim(0);

    std::vector<TData> factor(tracelist->GetTotPoints(), TData(0.0));

    const size_t nElmts = expansionList->GetExpSize();
    for (size_t el = 0; el < nElmts; ++el)
    {
        auto exp     = expansionList->GetExp(el);
        int numModes = 0;
        for (int nd = 0; nd < spaceDim; ++nd)
        {
            numModes = std::max(numModes, exp->GetBasisNumModes(nd));
        }
        const TData penalty = static_cast<TData>(numModes * numModes);

        const int nLocalTraces = exp->GetNtraces();
        for (int t = 0; t < nLocalTraces; ++t)
        {
            const int globalTraceId = elmtToTrace[el][t]->GetElmtId();
            const int offset        = tracelist->GetPhys_Offset(globalTraceId);
            const int nTracePts     = tracelist->GetTotPoints(globalTraceId);

            for (int p = 0; p < nTracePts; ++p)
            {
                factor[offset + p] = std::max(factor[offset + p], penalty);
            }
        }
    }

    return factor;
}

template <typename TData>
std::vector<TData> GetIPTraceLengthRecip(const ExpListSharedPtr &expansionList)
{
    const size_t nTracePts = expansionList->GetTrace()->GetTotPoints();
    Array<OneD, double> lengthFwd(nTracePts, 0.0);
    Array<OneD, double> lengthBwd(nTracePts, 0.0);
    expansionList->GetTrace()->GetElmtNormalLength(lengthFwd, lengthBwd);

    if (auto discontField =
            std::dynamic_pointer_cast<DisContField>(expansionList))
    {
        auto &periodicFwdCopy = discontField->GetPeriodicFwdCopy();
        auto &periodicBwdCopy = discontField->GetPeriodicBwdCopy();
        ASSERTL1(periodicFwdCopy.size() == periodicBwdCopy.size(),
                 "Periodic forward/backward copy maps have different sizes.");
        for (size_t i = 0; i < periodicFwdCopy.size(); ++i)
        {
            lengthBwd[periodicBwdCopy[i]] = lengthFwd[periodicFwdCopy[i]];
        }
    }

    std::vector<TData> lengthRecip(nTracePts, TData(0.0));
    for (size_t i = 0; i < nTracePts; ++i)
    {
        if (std::abs(lengthBwd[i]) < NekConstants::kNekMachineEpsilon)
        {
            lengthFwd[i] *= TData(0.5);
            lengthBwd[i] = lengthFwd[i];
        }

        const TData sum = lengthBwd[i] + lengthFwd[i];
        const TData mul = lengthBwd[i] * lengthFwd[i];
        lengthRecip[i]  = TData(0.25) * sum / mul;
    }

    return lengthRecip;
}

template <typename TData, typename FillFunc>
LibUtilities::MemoryRegion<TData> CreateIPTraceBlockData(
    const ExpListSharedPtr &expansionList, const unsigned int blockIdx,
    FillFunc fill)
{
    auto trace  = expansionList->GetTrace();
    auto blocks = GetBlockAttributes<TData, FieldState::Phys>(trace);

    size_t traceOffset = 0;
    for (unsigned int blk = 0; blk < blockIdx; ++blk)
    {
        traceOffset += blocks[blk].GetNumElements() * blocks[blk].GetNumData();
    }

    const auto &block     = blocks[blockIdx];
    const size_t nRealPts = block.GetNumElements() * block.GetNumData();
    auto data             = LibUtilities::MemoryRegion<TData>(block.CompSize());
    auto dataptr = data.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (size_t i = 0; i < block.CompSize(); ++i)
    {
        dataptr[i] = (i < nRealPts) ? fill(traceOffset + i) : TData(0.0);
    }

    return data;
}
} // namespace

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const LocTracePhysToElmtMapsKey<TData> &locTracePhysToElmtMapsKey)
{
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = locTracePhysToElmtMapsKey.m_block_idx;
    const auto interleave_width = locTracePhysToElmtMapsKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    const auto nTraces = expPtr->GetNtraces();

    // Calculate total number of trace points across all traces
    size_t nTracePts = 0;
    for (unsigned int i = 0; i < nTraces; ++i)
    {
        nTracePts += expPtr->GetTraceNumPoints(i);
    }

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto tracefieldmap =
        locTraceToTraceMap->GetTraceFieldMapEssential(block_idx);

    const auto memsize = num_elmt_groups * interleave_width * nTracePts;
    Array<OneD, Array<OneD, int>> mapsArray(interleave_width * nTraces);
    auto maps    = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto mapsptr = maps.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
    {
        for (unsigned int i = 0; i < interleave_width; ++i, ++el)
        {
            for (unsigned int j = 0; j < nTraces; ++j)
            {
                size_t idx = i * nTraces + j;
                if (el < num_elements)
                {
                    mapsArray[idx] = tracefieldmap.m_locTracePhysToElmtMaps[j];
                }
                else
                {
                    mapsArray[idx] = Array<OneD, int>(
                        tracefieldmap.m_locTracePhysToElmtMaps[j].size(), 0);
                }
            }
        }

        for (unsigned int i = 0; i < interleave_width; ++i)
        {
            for (unsigned int j = 0; j < nTraces; ++j)
            {
                size_t idx = i * nTraces + j;
                auto &src  = mapsArray[idx];

                for (size_t k = 0; k < src.size(); ++k)
                {
                    *(mapsptr++) = static_cast<unsigned int>(src[k]);
                }
            }
        }
    }

    return maps;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> TraceEssentialCreator::Create(
    const IPTraceNormalKey<TData> &ipTraceNormalKey)
{
    const size_t nTracePts  = m_expansionList->GetTrace()->GetTotPoints();
    const unsigned int nDim = m_expansionList->GetCoordim(0);
    Array<OneD, Array<OneD, double>> normals(nDim);
    for (size_t d = 0; d < normals.size(); ++d)
    {
        normals[d] = Array<OneD, double>(nTracePts, 0.0);
    }
    m_expansionList->GetTrace()->GetNormals(normals);

    auto blocks = GetBlockAttributes<TData, FieldState::Phys>(
        m_expansionList->GetTrace());
    const auto &block = blocks[ipTraceNormalKey.m_block_idx];

    size_t traceOffset = 0;
    for (unsigned int blk = 0; blk < ipTraceNormalKey.m_block_idx; ++blk)
    {
        traceOffset += blocks[blk].GetNumElements() * blocks[blk].GetNumData();
    }

    const size_t nRealPts = block.GetNumElements() * block.GetNumData();
    auto data    = LibUtilities::MemoryRegion<TData>(nDim * block.CompSize());
    auto dataptr = data.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (unsigned int d = 0; d < nDim; ++d)
    {
        for (size_t i = 0; i < block.CompSize(); ++i)
        {
            dataptr[d * block.CompSize() + i] =
                (i < nRealPts) ? normals[d][traceOffset + i] : TData(0.0);
        }
    }

    return data;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> TraceEssentialCreator::Create(
    const IPTraceScalarKey<TData> &ipTraceScalarKey)
{
    const size_t nTracePts = m_expansionList->GetTrace()->GetTotPoints();
    std::vector<TData> scalarData(nTracePts, TData(0.0));

    switch (ipTraceScalarKey.m_type)
    {
        case IPTraceScalarData::BwdWeightAver:
        {
            Array<OneD, double> bwdWeightAver(nTracePts, 0.0);
            Array<OneD, double> bwdWeightJump(nTracePts, 0.0);
            m_expansionList->GetBwdWeight(bwdWeightAver, bwdWeightJump);
            for (size_t i = 0; i < nTracePts; ++i)
            {
                scalarData[i] = bwdWeightAver[i];
            }
            break;
        }
        case IPTraceScalarData::BwdWeightJump:
        {
            Array<OneD, double> bwdWeightAver(nTracePts, 0.0);
            Array<OneD, double> bwdWeightJump(nTracePts, 0.0);
            m_expansionList->GetBwdWeight(bwdWeightAver, bwdWeightJump);
            for (size_t i = 0; i < nTracePts; ++i)
            {
                scalarData[i] = bwdWeightJump[i];
            }
            break;
        }
        case IPTraceScalarData::LengthRecip:
        {
            scalarData = GetIPTraceLengthRecip<TData>(m_expansionList);
            break;
        }
        case IPTraceScalarData::PenaltyFactor:
        {
            scalarData = GetIPTracePenaltyFactor<TData>(m_expansionList);
            break;
        }
    }

    return CreateIPTraceBlockData<TData>(
        m_expansionList, ipTraceScalarKey.m_block_idx,
        [&](const size_t i) { return scalarData[i]; });
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> TraceEssentialCreator::Create(
    const IPTraceDerivBaseKey<TData> &ipTraceDerivBaseKey)
{
    auto coll = GetCollection(m_expansionList, ipTraceDerivBaseKey.m_block_idx);
    auto exp  = coll.GetExpVector()[0];

    const unsigned int nDim    = m_expansionList->GetCoordim(0);
    const unsigned int nCoeffs = exp->GetNcoeffs();
    unsigned int nLocTracePts  = 0;

    for (int t = 0; t < exp->GetNtraces(); ++t)
    {
        nLocTracePts += exp->GetTraceNumPoints(t);
    }

    auto data =
        LibUtilities::MemoryRegion<TData>(nDim * nCoeffs * nLocTracePts);
    auto ptr = data.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned int traceOffset = 0;
    for (int t = 0; t < exp->GetNtraces(); ++t)
    {
        auto traceExp             = exp->GetLocTraceExp(t);
        const int nPts            = exp->GetTraceNumPoints(t);
        const int nTraceMetricPts = traceExp->GetTotPoints();

        Array<OneD, double> ones(nTraceMetricPts, 1.0);
        Array<OneD, double> metric(nTraceMetricPts, 0.0);
        traceExp->MultiplyByQuadratureMetric(ones, metric);

        Array<OneD, DNekMatSharedPtr> derivBaseOnTrace;
        exp->PhysDerivBaseOnTraceMat(t, derivBaseOnTrace);
        ASSERTL1(nPts <= nTraceMetricPts,
                 "Trace derivative basis has more points than the trace "
                 "quadrature metric.");

        for (unsigned int d = 0; d < nDim; ++d)
        {
            for (unsigned int c = 0; c < nCoeffs; ++c)
            {
                for (int p = 0; p < nPts; ++p)
                {
                    const size_t idx =
                        (d * nCoeffs + c) * nLocTracePts + traceOffset + p;
                    ptr[idx] = static_cast<TData>((*derivBaseOnTrace[d])(c, p) *
                                                  metric[p]);
                }
            }
        }

        traceOffset += nPts;
    }

    return data;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const OrientationMapsKey<TData> &orientationMapsKey)
{
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = orientationMapsKey.m_block_idx;
    const auto interleave_width = orientationMapsKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    const auto nTraces = expPtr->GetNtraces();

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto tracefieldmap =
        locTraceToTraceMap->GetTraceFieldMapEssential(block_idx);

    size_t memsize = 0;
    Array<OneD, Array<OneD, int>> mapsArray(interleave_width * nTraces);

    for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
    {
        for (unsigned int i = 0; i < interleave_width; ++i, ++el)
        {
            for (unsigned int j = 0; j < nTraces; ++j)
            {
                if (el < num_elements)
                {

                    unsigned int orient_id_Ptr =
                        tracefieldmap.m_orientationIds[j][el];
                    memsize +=
                        tracefieldmap.m_orientationMaps[orient_id_Ptr].size();
                }
                else
                {
                    unsigned int orient_id_Ptr =
                        tracefieldmap.m_orientationIds[j][num_elements - 1];

                    memsize +=
                        tracefieldmap.m_orientationMaps[orient_id_Ptr].size();
                }
            }
        }
    }

    auto maps    = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto mapsptr = maps.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
    {
        for (unsigned int i = 0; i < interleave_width; ++i, ++el)
        {
            for (unsigned int j = 0; j < nTraces; ++j)
            {
                size_t idx = i * nTraces + j;

                if (el < num_elements)
                {

                    int orient_id_Ptr = tracefieldmap.m_orientationIds[j][el];

                    mapsArray[idx] =
                        tracefieldmap.m_orientationMaps[orient_id_Ptr];
                }
                else
                {
                    int orient_id_Ptr =
                        tracefieldmap.m_orientationIds[j][num_elements - 1];

                    size_t sz =
                        tracefieldmap.m_orientationMaps[orient_id_Ptr].size();
                    mapsArray[idx] = Array<OneD, int>(sz, 0);
                }
            }
        }

        // Pack into flat buffer
        for (unsigned int i = 0; i < interleave_width; ++i)
        {
            for (unsigned int j = 0; j < nTraces; ++j)
            {
                size_t idx = i * nTraces + j;
                auto &src  = mapsArray[idx];

                for (size_t k = 0; k < src.size(); ++k)
                {
                    *(mapsptr++) = static_cast<unsigned int>(src[k]);
                }
            }
        }
    }

    return maps;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create(
    const OrientationMapsOffsetKey<TData> &orientationMapsOffsetKey)
{
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = orientationMapsOffsetKey.m_block_idx;
    const auto interleave_width = orientationMapsOffsetKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    const auto nTraces = expPtr->GetNtraces();
    // Calculate total number of trace points across all traces
    size_t nTracePts = 0;
    for (unsigned int i = 0; i < nTraces; ++i)
    {
        nTracePts += expPtr->GetTraceNumPoints(i);
    }

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto tracefieldmap =
        locTraceToTraceMap->GetTraceFieldMapEssential(block_idx);

    const auto memsize = num_elmt_groups * interleave_width * nTraces;
    Array<OneD, unsigned int> offsetArray(interleave_width * nTraces);
    auto offset = LibUtilities::MemoryRegion<size_t>(memsize);
    auto offsetptr =
        offset.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned int dataSize = 0;
    for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
    {
        for (unsigned int i = 0; i < interleave_width; ++i, ++el)
        {
            for (unsigned int j = 0; j < nTraces; ++j)
            {
                size_t idx = i * nTraces + j;

                if (el < num_elements)
                {
                    offsetArray[idx] = dataSize;

                    unsigned int orient_id_Ptr =
                        tracefieldmap.m_orientationIds[j][el];

                    dataSize +=
                        tracefieldmap.m_orientationMaps[orient_id_Ptr].size();
                }
                else
                {
                    offsetArray[idx] = dataSize;

                    unsigned int orient_id_Ptr =
                        tracefieldmap.m_orientationIds[j][num_elements - 1];

                    dataSize +=
                        tracefieldmap.m_orientationMaps[orient_id_Ptr].size();
                }
            }
        }

        // Pack into flat buffer
        for (unsigned int i = 0; i < interleave_width; ++i)
        {
            for (unsigned int j = 0; j < nTraces; ++j)
            {
                size_t idx     = i * nTraces + j;
                auto &src      = offsetArray[idx];
                *(offsetptr++) = src;
            }
        }
    }

    return offset;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create(
    const LocToTracePhysOffsetKey<TData> &locToTracePhysOffsetKey)
{
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = locToTracePhysOffsetKey.m_block_idx;
    const auto interleave_width = locToTracePhysOffsetKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    const auto nTraces = expPtr->GetNtraces();

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto tracefieldmap =
        locTraceToTraceMap->GetTraceFieldMapEssential(block_idx);

    const auto memsize = num_elmt_groups * interleave_width * nTraces;
    Array<OneD, Array<OneD, size_t>> offsetArray(nTraces);
    auto offset = LibUtilities::MemoryRegion<size_t>(memsize);
    auto offsetptr =
        offset.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (unsigned int i = 0; i < nTraces; ++i)
    {

        const auto &original = tracefieldmap.m_locToTracePhysOffset[i];
        const size_t padSize =
            num_elmt_groups * interleave_width - num_elements;

        // Combine original + padding into one array
        offsetArray[i] = Array<OneD, size_t>(original.size() + padSize);
        std::copy(original.begin(), original.end(), offsetArray[i].begin());

        if (padSize > 0)
        {
            std::fill(offsetArray[i].begin() + original.size(),
                      offsetArray[i].end(), 0.0);
        }

        auto &src = offsetArray[i];

        for (size_t j = 0; j < src.size(); ++j)
        {
            *(offsetptr++) = src[j];
        }
    }

    return offset;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<bool> TraceEssentialCreator::Create(
    const IsLocTraceLeftAdjacentKey<TData> &isLocTraceLeftAdjacentKey)
{
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = isLocTraceLeftAdjacentKey.m_block_idx;
    const auto interleave_width = isLocTraceLeftAdjacentKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    const auto nTraces = expPtr->GetNtraces();

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto tracefieldmap =
        locTraceToTraceMap->GetTraceFieldMapEssential(block_idx);
    const auto memsize = num_elmt_groups * interleave_width * nTraces;
    Array<OneD, Array<OneD, bool>> isLeftArray(nTraces);
    auto isLeft = LibUtilities::MemoryRegion<bool>(memsize);
    auto isLeftptr =
        isLeft.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (unsigned int i = 0; i < nTraces; ++i)
    {
        const auto &original = tracefieldmap.m_isLocTraceLeftAdjacent[i];
        const size_t padSize =
            num_elmt_groups * interleave_width - num_elements;

        // Allocate array with room for padding
        isLeftArray[i] = Array<OneD, bool>(original.size() + padSize);

        // Copy valid entries
        std::copy(original.begin(), original.end(), isLeftArray[i].begin());

        // Fill padding with `true`
        if (padSize > 0)
        {
            std::fill(isLeftArray[i].begin() + original.size(),
                      isLeftArray[i].end(), true);
        }

        // Write into output memory region
        auto &src = isLeftArray[i];

        for (size_t j = 0; j < src.size(); ++j)
        {
            *(isLeftptr++) = src[j];
        }
    }

    return isLeft;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const InterpTraceIndexKey<TData> &interpTraceIndexKey)
{
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = interpTraceIndexKey.m_block_idx;
    const auto interleave_width = interpTraceIndexKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    const auto nTraces = expPtr->GetNtraces();

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp = locTraceToTraceMap->GetTraceInterpEssential(block_idx);

    auto interpLocTraceToTrace = traceInterp.m_interpTrace;

    unsigned int numTypes = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpLocTraceToTrace[dir].size();
    }

    const auto memsize = num_elmt_groups * interleave_width * nTraces;
    Array<OneD, Array<OneD, unsigned int>> interpTraceIndex(nTraces);
    auto index    = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto indexptr = index.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (unsigned int i = 0; i < nTraces; ++i)
    {
        ASSERTL1(i < traceInterp.m_interpTraceIndex.size(),
                 "Invalid trace index for m_interpTraceIndex");

        const auto &original = traceInterp.m_interpTraceIndex[i];
        const size_t padSize =
            num_elmt_groups * interleave_width - num_elements;

        // Allocate array with room for padding
        interpTraceIndex[i] =
            Array<OneD, unsigned int>(original.size() + padSize);

        // Copy valid entries
        std::copy(original.begin(), original.end(),
                  interpTraceIndex[i].begin());

        // Fill padding with number of types
        if (padSize > 0)
        {
            std::fill(interpTraceIndex[i].begin() + original.size(),
                      interpTraceIndex[i].end(), numTypes);
        }

        // Write into output memory region
        auto &src = interpTraceIndex[i];
        for (size_t j = 0; j < src.size(); ++j)
        {
            *(indexptr++) = src[j];
        }
    }

    return index;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const InterpPointsKey<TData> &interpPointsKey)
{

    const auto block_idx = interpPointsKey.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp  = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpPoints = traceInterp.m_interpPoints;

    unsigned int numTuples = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTuples += interpPoints[dir].size();
    }

    // 4 ints per tuple
    // Add one extra tuple for padding
    const auto memsize = (numTuples + 1) * 4;
    Array<OneD, unsigned int> interpPointsArray(memsize);
    auto interp = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto interpPointsptr =
        interp.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Fill array with number of points for each from/to points key
    unsigned int count = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpPoints[dir])
        {
            interpPointsArray[count++] = std::get<0>(tp).GetNumPoints();
            interpPointsArray[count++] = std::get<1>(tp).GetNumPoints();
            interpPointsArray[count++] = std::get<2>(tp).GetNumPoints();
            interpPointsArray[count++] = std::get<3>(tp).GetNumPoints();
        }
    }

    // Padding entry
    interpPointsArray[count++] = 0;
    interpPointsArray[count++] = 0;
    interpPointsArray[count++] = 0;
    interpPointsArray[count++] = 0;

    ASSERTL1(count == memsize, "Quad range size mismatch!");

    for (unsigned int i = 0; i < memsize; ++i)
    {
        *(interpPointsptr++) = interpPointsArray[i];
    }

    return interp;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const InterpTypesKey<TData> &interpTypesKey)
{

    const auto block_idx = interpTypesKey.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp  = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpPoints = traceInterp.m_interpPoints;

    unsigned int numTuples = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTuples += interpPoints[dir].size();
    }

    const auto memsize      = 1;
    unsigned int typesArray = numTuples;
    auto interp             = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto interpTypesptr =
        interp.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    *(interpTypesptr) = typesArray;

    return interp;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const QuadRangeKey<TData> &quadRangeKey)
{
    const auto block_idx = quadRangeKey.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp  = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpPoints = traceInterp.m_interpPoints;

    unsigned int numTuples = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTuples += interpPoints[dir].size();
    }

    // 4 ints per tuple, 2 directions
    const auto memsize = numTuples * 4 * 2;
    Array<OneD, unsigned int> quadRangeArray(memsize);
    auto quadRange = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto quadRangeptr =
        quadRange.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned int count = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpPoints[dir])
        {
            auto fromPointsKey0 = std::get<0>(tp);
            auto toPointsKey0   = std::get<1>(tp);
            auto fromPointsKey1 = std::get<2>(tp);
            auto toPointsKey1   = std::get<3>(tp);

            int fbegin0, fend0, fbegin1, fend1;
            int tbegin0, tend0, tbegin1, tend1;
            LibUtilities::PointsKey::GetEffectiveQuadRange(fromPointsKey0,
                                                           fbegin0, fend0);
            LibUtilities::PointsKey::GetEffectiveQuadRange(toPointsKey0,
                                                           tbegin0, tend0);
            LibUtilities::PointsKey::GetEffectiveQuadRange(fromPointsKey1,
                                                           fbegin1, fend1);
            LibUtilities::PointsKey::GetEffectiveQuadRange(toPointsKey1,
                                                           tbegin1, tend1);
            quadRangeArray[count++] = fbegin0;
            quadRangeArray[count++] = fend0;
            quadRangeArray[count++] = tbegin0;
            quadRangeArray[count++] = tend0;
            quadRangeArray[count++] = fbegin1;
            quadRangeArray[count++] = fend1;
            quadRangeArray[count++] = tbegin1;
            quadRangeArray[count++] = tend1;
        }
    }

    ASSERTL1(count == memsize, "Quad range size mismatch!");

    for (unsigned int i = 0; i < memsize; ++i)
    {
        *(quadRangeptr++) = quadRangeArray[i];
    }

    return quadRange;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<InterpLocTraceToTrace> TraceEssentialCreator::Create(
    const InterpTraceKey<TData> &interpTraceKey)
{
    const auto block_idx = interpTraceKey.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpLocTraceToTrace = traceInterp.m_interpTrace;

    unsigned int numTypes = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpLocTraceToTrace[dir].size();
    }

    // Add one extra entry for padding elements
    const auto memsize = numTypes + 1;
    Array<OneD, InterpLocTraceToTrace> interpTraceArray(memsize);
    auto interpTrace =
        LibUtilities::MemoryRegion<InterpLocTraceToTrace>(memsize);
    auto interpTraceptr =
        interpTrace.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned int count = 0;

    // Pad with an invalid entry for interpolation type
    constexpr auto padEntry = static_cast<InterpLocTraceToTrace>(7);

    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpLocTraceToTrace[dir])
        {

            interpTraceArray[count++] = tp;
        }
    }
    // Padding entry do nothing
    interpTraceArray[count++] = padEntry;

    ASSERTL1(count == memsize, "InterpTrace size mismatch!");

    for (unsigned int i = 0; i < memsize; ++i)
    {
        *(interpTraceptr++) = interpTraceArray[i];
    }

    return interpTrace;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> TraceEssentialCreator::Create(
    const InterpTraceI0Key<TData> &interpTraceI0Key)
{
    const auto block_idx = interpTraceI0Key.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp   = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpTraceI0 = traceInterp.m_interpTraceI0;

    // Count total number of TData entries needed
    size_t memsize = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpTraceI0[dir])
        {
            if (!tp)
            {
                continue;
            }
            memsize += tp->GetRows() * tp->GetColumns();
        }
    }

    // Allocate one contiguous MemoryRegion for all matrix entries
    auto interpTrace = LibUtilities::MemoryRegion<TData>(memsize);
    auto interpTracePtr =
        interpTrace.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Copy data into MemoryRegion in a flattened layout
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpTraceI0[dir])
        {
            if (!tp)
            {
                continue;
            }

            auto arr = tp->GetPtr();
            auto *I0 = arr.data();

            for (size_t idx = 0; idx < arr.size(); ++idx)
            {
                *(interpTracePtr++) = I0[idx];
            }
        }
    }

    return interpTrace;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const InterpTraceI0OffsetKey<TData> &interpTraceI0OffsetKey)
{
    const auto block_idx = interpTraceI0OffsetKey.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp   = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpTraceI0 = traceInterp.m_interpTraceI0;

    unsigned int numTypes = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpTraceI0[dir].size();
    }

    const auto memsize = numTypes;
    Array<OneD, unsigned int> offsetArray(memsize);
    auto offset = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto offsetptr =
        offset.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned int dataSize = 0;
    unsigned int count    = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpTraceI0[dir])
        {
            offsetArray[count++] = dataSize;

            if (tp)
            {
                dataSize += tp->GetRows() * tp->GetColumns();
            }
        }
    }

    ASSERTL1(count == memsize, "InterpTraceI0 size mismatch!");

    for (unsigned int i = 0; i < memsize; ++i)
    {
        offsetptr[i] = offsetArray[i];
    }

    return offset;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> TraceEssentialCreator::Create(
    const InterpTraceI1Key<TData> &interpTraceI1Key)
{
    const auto block_idx = interpTraceI1Key.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp   = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpTraceI1 = traceInterp.m_interpTraceI1;

    // Count total number of TData entries needed
    size_t memsize = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpTraceI1[dir])
        {
            if (!tp)
            {
                continue;
            }
            memsize += tp->GetRows() * tp->GetColumns();
        }
    }

    // Allocate one contiguous MemoryRegion for all matrix entries
    auto interpTrace = LibUtilities::MemoryRegion<TData>(memsize);
    auto interpTracePtr =
        interpTrace.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Copy data into MemoryRegion in a flattened layout
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpTraceI1[dir])
        {
            if (!tp)
            {
                continue;
            }

            auto arr = tp->GetPtr();
            auto *I1 = arr.data();
            for (size_t idx = 0; idx < arr.size(); ++idx)
            {
                *(interpTracePtr++) = I1[idx];
            }
        }
    }

    return interpTrace;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const InterpTraceI1OffsetKey<TData> &interpTraceI1OffsetKey)
{
    const auto block_idx = interpTraceI1OffsetKey.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp   = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpTraceI1 = traceInterp.m_interpTraceI1;

    unsigned int numTypes = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpTraceI1[dir].size();
    }

    const auto memsize = numTypes;
    Array<OneD, unsigned int> offsetArray(memsize);
    auto offset = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto offsetptr =
        offset.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned int dataSize = 0;
    unsigned int count    = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpTraceI1[dir])
        {
            offsetArray[count++] = dataSize;

            if (tp)
            {
                dataSize += tp->GetRows() * tp->GetColumns();
            }
        }
    }

    ASSERTL1(count == memsize, "InterpTraceI1 size mismatch!");
    for (unsigned int i = 0; i < memsize; ++i)
    {
        offsetptr[i] = offsetArray[i];
    }

    return offset;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::Create(
    const InterpFromTraceI0Key<TData> &interpFromTraceI0Key)
{
    const auto block_idx = interpFromTraceI0Key.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpFromTraceI0 = traceInterp.m_interpFromTraceI0;

    unsigned int numTypes = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpFromTraceI0[dir].size();
    }

    const auto memsize = numTypes;
    Array<OneD, DNekMatSharedPtr> interpFromTraceI0Array(memsize);
    auto interpTrace = LibUtilities::MemoryRegion<DNekMatSharedPtr>(memsize);
    auto interpTraceptr =
        interpTrace.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned int count = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpFromTraceI0[dir])
        {
            interpFromTraceI0Array[count++] = tp;
        }
    }

    ASSERTL1(count == memsize, "InterpFromTraceI0 size mismatch!");

    for (unsigned int i = 0; i < memsize; ++i)
    {
        *(interpTraceptr++) = interpFromTraceI0Array[i];
    }

    return interpTrace;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::Create(
    const InterpFromTraceI1Key<TData> &interpFromTraceI1Key)
{
    const auto block_idx = interpFromTraceI1Key.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpFromTraceI1 = traceInterp.m_interpFromTraceI1;

    unsigned int numTypes = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpFromTraceI1[dir].size();
    }

    const auto memsize = numTypes;
    Array<OneD, DNekMatSharedPtr> interpFromTraceI1Array(memsize);
    auto interpTrace = LibUtilities::MemoryRegion<DNekMatSharedPtr>(memsize);
    auto interpTraceptr =
        interpTrace.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned int count = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpFromTraceI1[dir])
        {
            interpFromTraceI1Array[count++] = tp;
        }
    }

    ASSERTL1(count == memsize, "InterpFromTraceI1 size mismatch!");

    for (unsigned int i = 0; i < memsize; ++i)
    {
        *(interpTraceptr++) = interpFromTraceI1Array[i];
    }

    return interpTrace;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> TraceEssentialCreator::Create(
    const InterpEndPtI0Key<TData> &interpEndPtI0Key)
{
    const auto block_idx = interpEndPtI0Key.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp   = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpEndPtI0 = traceInterp.m_interpEndPtI0;

    unsigned int numTypes = 0;
    size_t memsize        = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpEndPtI0[dir].size();
        for (unsigned int i = 0; i < interpEndPtI0[dir].size(); ++i)
        {
            memsize += interpEndPtI0[dir][i].size();
        }
    }

    Array<OneD, Array<OneD, double>> interpEndPtI0Array(numTypes);
    auto interpTrace = LibUtilities::MemoryRegion<TData>(memsize);
    auto interpTraceptr =
        interpTrace.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    size_t count = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpEndPtI0[dir])
        {

            interpEndPtI0Array[count++] = tp;
        }
    }

    for (unsigned int i = 0; i < numTypes; ++i)
    {
        for (unsigned int j = 0; j < interpEndPtI0Array[i].size(); ++j)
        {
            *(interpTraceptr++) = interpEndPtI0Array[i][j];
        }
    }

    return interpTrace;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const InterpEndPtI0OffsetKey<TData> &interpEndPtI0OffsetKey)
{
    const auto block_idx = interpEndPtI0OffsetKey.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp   = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpEndPtI0 = traceInterp.m_interpEndPtI0;

    unsigned int numTypes = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpEndPtI0[dir].size();
    }

    Array<OneD, unsigned int> offsetArray(numTypes);

    size_t memsize = numTypes;
    auto offset    = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto offsetptr =
        offset.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    size_t count          = 0;
    unsigned int dataSize = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (unsigned int i = 0; i < interpEndPtI0[dir].size(); ++i)
        {
            offsetArray[count++] = dataSize;
            dataSize += interpEndPtI0[dir][i].size();
        }
    }

    for (unsigned int i = 0; i < numTypes; ++i)
    {
        *(offsetptr++) = offsetArray[i];
    }

    return offset;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> TraceEssentialCreator::Create(
    const InterpEndPtI1Key<TData> &interpEndPtI1Key)
{
    const auto block_idx = interpEndPtI1Key.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp   = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpEndPtI1 = traceInterp.m_interpEndPtI1;

    unsigned int numTypes = 0;
    size_t memsize        = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpEndPtI1[dir].size();
        for (unsigned int i = 0; i < interpEndPtI1[dir].size(); ++i)
        {
            memsize += interpEndPtI1[dir][i].size();
        }
    }

    Array<OneD, Array<OneD, double>> interpEndPtI1Array(numTypes);
    auto interpTrace = LibUtilities::MemoryRegion<TData>(memsize);
    auto interpTraceptr =
        interpTrace.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    size_t count = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (const auto &tp : interpEndPtI1[dir])
        {

            interpEndPtI1Array[count++] = tp;
        }
    }

    for (unsigned int i = 0; i < numTypes; ++i)
    {
        for (unsigned int j = 0; j < interpEndPtI1Array[i].size(); ++j)
        {
            *(interpTraceptr++) = interpEndPtI1Array[i][j];
        }
    }

    return interpTrace;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create(
    const InterpEndPtI1OffsetKey<TData> &interpEndPtI1OffsetKey)
{
    const auto block_idx = interpEndPtI1OffsetKey.m_block_idx;

    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    const LocTraceToTraceMapSharedPtr locTraceToTraceMap =
        m_expansionList->GetLocTraceToTraceMap();
    auto traceInterp   = locTraceToTraceMap->GetTraceInterpEssential(block_idx);
    auto interpEndPtI1 = traceInterp.m_interpEndPtI1;

    unsigned int numTypes = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        numTypes += interpEndPtI1[dir].size();
    }

    Array<OneD, unsigned int> offsetArray(numTypes);

    size_t memsize = numTypes;
    auto offset    = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto offsetptr =
        offset.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    size_t count          = 0;
    unsigned int dataSize = 0;
    for (unsigned int dir = 0; dir < 2; ++dir)
    {
        for (unsigned int i = 0; i < interpEndPtI1[dir].size(); ++i)
        {
            offsetArray[count++] = dataSize;
            dataSize += interpEndPtI1[dir][i].size();
        }
    }

    for (unsigned int i = 0; i < numTypes; ++i)
    {
        *(offsetptr++) = offsetArray[i];
    }

    return offset;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> TraceEssentialCreator::Create(
    const Interp1DKey<TData> &interp1DKey)
{
    using namespace Nektar::LibUtilities;
    const auto fromKey = interp1DKey.m_fromKey;
    const auto toKey   = interp1DKey.m_toKey;

    const auto nq  = fromKey.GetNumPoints();
    const auto nqe = toKey.GetNumPoints();
    Array<OneD, double> tmp(nq), t;
    Array<OneD, double> mat(nqe * nq);
    for (unsigned int i = 0; i < nq; ++i)
    {
        Vmath::Zero(nq, tmp, 1);
        tmp[i] = 1.0;
        LibUtilities::Interp1D(fromKey, tmp, toKey, t = mat + i * nqe);
    }
    return LibUtilities::MemoryRegion<TData>::template FromArray<MemSpace>(mat);
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> TraceEssentialCreator::Create(
    const Interp2DKey<TData> &interp2DKey)
{
    using namespace Nektar::LibUtilities;
    const auto fromKey0 = interp2DKey.m_fromKey0;
    const auto fromKey1 = interp2DKey.m_fromKey1;
    const auto toKey0   = interp2DKey.m_toKey0;
    const auto toKey1   = interp2DKey.m_toKey1;

    const auto nqe     = fromKey0.GetNumPoints() * fromKey1.GetNumPoints();
    const auto nq_face = toKey0.GetNumPoints() * toKey1.GetNumPoints();
    Array<OneD, double> tmp(nqe), t;
    Array<OneD, double> mat(nq_face * nqe);
    for (unsigned int i = 0; i < nqe; ++i)
    {
        Vmath::Zero(nqe, tmp, 1);
        tmp[i] = 1.0;
        LibUtilities::Interp2D(fromKey0, fromKey1, tmp, toKey0, toKey1,
                               t = mat + i * nq_face);
    }

    return LibUtilities::MemoryRegion<TData>::template FromArray<MemSpace>(mat);
}

} // namespace Nektar::MultiRegions
