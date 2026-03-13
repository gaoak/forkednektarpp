///////////////////////////////////////////////////////////////////////////////
//
// File: GetFwdBwdTracePhysDevice.hpp
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

#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysDeviceKernels.hpp"
#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysOp.hpp"

#include "Operators/Utils/UtilsKernels.hpp"

using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class GetFwdBwdTracePhysBlockOpImpl : public GetFwdBwdTracePhysBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    GetFwdBwdTracePhysBlockOpImpl(const unsigned int block_idx,
                                  const LocalRegions::ExpansionSharedPtr &exp,
                                  NekDataWarehouseSharedPtr dataWarehouse)
        : GetFwdBwdTracePhysBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_shapeType = exp->DetShapeType();
        m_nqTot     = exp->GetTotPoints();
        m_nTraces   = exp->GetNtraces();
        m_dimension = exp->GetShapeDimension();
        m_tracePts  = 0;
        std::vector<unsigned int> nqOffset(m_nTraces, 0u);
        for (unsigned int i = 0; i < m_nTraces; ++i)
        {
            unsigned int nq = exp->GetTraceNumPoints(i);
            nqOffset[i]     = m_tracePts;
            m_tracePts += nq;
        }

        m_nqOffset = MemoryRegion<unsigned int>::template FromVector<
            MemSpace, unsigned int>(nqOffset);

        m_locTracePhysToElmtMaps =
            this->m_dataWarehouse->template GetData<MemSpace>(
                LocTracePhysToElmtMapsKey<TData>(block_idx,
                                                 m_implInterleaveWidth));

        m_orientationMaps = this->m_dataWarehouse->template GetData<MemSpace>(
            OrientationMapsKey<TData>(block_idx, m_implInterleaveWidth));

        m_orientationMapsOffset =
            this->m_dataWarehouse->template GetData<MemSpace>(
                OrientationMapsOffsetKey<TData>(block_idx,
                                                m_implInterleaveWidth));

        m_locToTracePhysOffset =
            this->m_dataWarehouse->template GetData<MemSpace>(
                LocToTracePhysOffsetKey<TData>(block_idx,
                                               m_implInterleaveWidth));

        m_isLocTraceLeftAdjacent =
            this->m_dataWarehouse->template GetData<MemSpace>(
                IsLocTraceLeftAdjacentKey<TData>(block_idx,
                                                 m_implInterleaveWidth));

        m_interpTraceIndex = this->m_dataWarehouse->template GetData<MemSpace>(
            InterpTraceIndexKey<TData>(block_idx, m_implInterleaveWidth));

        m_interpPoints = this->m_dataWarehouse->template GetData<MemSpace>(
            InterpPointsKey<TData>(block_idx));

        m_interpTypes = this->m_dataWarehouse->template GetData<MemSpace>(
            InterpTypesKey<TData>(block_idx));

        m_quadRange = this->m_dataWarehouse->template GetData<MemSpace>(
            QuadRangeKey<TData>(block_idx));

        m_interpTrace = this->m_dataWarehouse->template GetData<MemSpace>(
            InterpTraceKey<TData>(block_idx));

        m_interpTraceI0 = this->m_dataWarehouse->template GetData<MemSpace>(
            InterpTraceI0Key<TData>(block_idx));

        m_interpTraceI0Offset =
            this->m_dataWarehouse->template GetData<MemSpace>(
                InterpTraceI0OffsetKey<TData>(block_idx));
        m_interpEndPtI0 = this->m_dataWarehouse->template GetData<MemSpace>(
            InterpEndPtI0Key<TData>(block_idx));

        m_interpEndPtI0Offset =
            this->m_dataWarehouse->template GetData<MemSpace>(
                InterpEndPtI0OffsetKey<TData>(block_idx));

        if (m_dimension == 3)
        {
            m_interpTraceI1 = this->m_dataWarehouse->template GetData<MemSpace>(
                InterpTraceI1Key<TData>(block_idx));

            m_interpTraceI1Offset =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    InterpTraceI1OffsetKey<TData>(block_idx));

            m_interpEndPtI1 = this->m_dataWarehouse->template GetData<MemSpace>(
                InterpEndPtI1Key<TData>(block_idx));

            m_interpEndPtI1Offset =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    InterpEndPtI1OffsetKey<TData>(block_idx));
        }
    }

    // className - for TraceBlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in TraceBlockOperatorFactory.
    static std::unique_ptr<GetFwdBwdTracePhysBlockOp<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            GetFwdBwdTracePhysBlockOpImpl<ExecSpace, TData>>(block_idx, exp,
                                                             dataWarehouse);
    }

protected:
    LibUtilities::ShapeType m_shapeType;
    static constexpr unsigned int m_implInterleaveWidth =
        NektarSpaces::vector_width<ExecSpace, TData>::value;
    unsigned int m_nqTot;
    unsigned int m_nTraces;
    unsigned int m_dimension;
    unsigned int m_tracePts = 0;
    MemoryRegion<unsigned int> m_nqOffset;
    MemoryRegion<TData> m_wsp;
    MemoryRegion<size_t> m_traceBlockOffset;
    MemoryRegion<size_t> m_traceTotOffset;
    MemoryRegion<size_t> m_traceBlockSize;
    const unsigned int *m_locTracePhysToElmtMaps;
    const unsigned int *m_orientationMaps;
    const size_t *m_orientationMapsOffset;
    const size_t *m_locToTracePhysOffset;
    const bool *m_isLocTraceLeftAdjacent;
    const unsigned int *m_interpTraceIndex;
    const unsigned int *m_interpPoints;
    const unsigned int *m_interpTypes;
    const unsigned int *m_quadRange;
    const MultiRegions::InterpLocTraceToTrace *m_interpTrace;
    const TData *m_interpTraceI0;
    const unsigned int *m_interpTraceI0Offset;
    const TData *m_interpEndPtI0;
    const unsigned int *m_interpEndPtI0Offset;
    const TData *m_interpTraceI1;
    const unsigned int *m_interpTraceI1Offset;
    const TData *m_interpEndPtI1;
    const unsigned int *m_interpEndPtI1Offset;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &phyBlock,
                 Field<TData, FieldState::Phys> &fwd,
                 Field<TData, FieldState::Phys> &bwd) override
    {

        switch (m_dimension)
        {
            case 2:
            {
                Operator2D(phyBlock, fwd, bwd);
                break;
            }
            case 3:
            {
                Operator3D(phyBlock, fwd, bwd);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    void Operator2D(BlockAccessor<TData, FieldState::Phys> &phyBlock,
                    Field<TData, FieldState::Phys> &fwd,
                    Field<TData, FieldState::Phys> &bwd)
    {
        // Get number of elements with padding.
        const auto nelmt = phyBlock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto phyptr = phyBlock.template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = phyBlock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize  = (nelmt + blockSize - 1u) / blockSize;

        const auto nTraceBlk = fwd.GetBlocks().size();
        auto &fwdBlock       = fwd.GetBlocks()[0];
        auto &bwdBlock       = bwd.GetBlocks()[0];
        auto fwdptr          = fwdBlock.template GetPtr<MemSpace, WriteOnly>();
        auto bwdptr          = bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        size_t traceSize     = fwd.GetBlocks()[0].CompSize();

        for (unsigned int traceBlk = 1; traceBlk < nTraceBlk; ++traceBlk)
        {
            traceSize += fwd.GetBlocks()[traceBlk].CompSize();
            auto &fwdBlock = fwd.GetBlocks()[traceBlk];
            auto &bwdBlock = bwd.GetBlocks()[traceBlk];
            fwdBlock.template GetPtr<MemSpace, WriteOnly>();
            bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        }

        auto nqOffsetPtr = m_nqOffset.template GetPtr<MemSpace, ReadOnly>();

        auto traceBlockOffsetPtr =
            m_traceBlockOffset.template GetPtr<MemSpace, ReadOnly>();

        auto traceTotOffsetPtr =
            m_traceTotOffset.template GetPtr<MemSpace, ReadOnly>();

        auto traceBlockSizePtr =
            m_traceBlockSize.template GetPtr<MemSpace, ReadOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < phyBlock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                      phyBlock.GetInterleaveWidth(), nelmt,
                                      phyBlock.GetNumData(), (TData *)phyptr);

            if (this->m_fwdOnly)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (GetFwdBwdTracePhys2DKernelLauncher<true>), gridSize,
                    blockSize, 0, m_nqTot, nelmt, m_tracePts, m_nTraces,
                    nqOffsetPtr, nTraceBlk, phyBlock.GetNumComponents(), nc,
                    traceBlockSizePtr, traceBlockOffsetPtr, traceTotOffsetPtr,
                    m_locTracePhysToElmtMaps, m_orientationMaps,
                    m_orientationMapsOffset, m_locToTracePhysOffset,
                    m_isLocTraceLeftAdjacent, m_interpTraceIndex,
                    m_interpPoints, m_interpTypes, m_quadRange, m_interpTrace,
                    m_interpTraceI0, m_interpTraceI0Offset, m_interpEndPtI0,
                    m_interpEndPtI0Offset, phyptr, fwdptr, bwdptr);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (GetFwdBwdTracePhys2DKernelLauncher<false>), gridSize,
                    blockSize, 0, m_nqTot, nelmt, m_tracePts, m_nTraces,
                    nqOffsetPtr, nTraceBlk, phyBlock.GetNumComponents(), nc,
                    traceBlockSizePtr, traceBlockOffsetPtr, traceTotOffsetPtr,
                    m_locTracePhysToElmtMaps, m_orientationMaps,
                    m_orientationMapsOffset, m_locToTracePhysOffset,
                    m_isLocTraceLeftAdjacent, m_interpTraceIndex,
                    m_interpPoints, m_interpTypes, m_quadRange, m_interpTrace,
                    m_interpTraceI0, m_interpTraceI0Offset, m_interpEndPtI0,
                    m_interpEndPtI0Offset, phyptr, fwdptr, bwdptr);
            }

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, phyBlock.GetNumData(),
                                      (TData *)phyptr);

            // Increment pointers.
            phyptr += phyBlock.CompSize();
        }

        // Set trace block to phys block interleave.
        for (unsigned int traceBlk = 0; traceBlk < fwd.GetBlocks().size();
             ++traceBlk)
        {
            auto &fwdBlock = fwd.GetBlocks()[traceBlk];
            auto &bwdBlock = bwd.GetBlocks()[traceBlk];
            fwdBlock.template SetInterleaveWidth<TData>(1);
            bwdBlock.template SetInterleaveWidth<TData>(1);
        }
    }

    void Operator3D(BlockAccessor<TData, FieldState::Phys> &phyBlock,
                    Field<TData, FieldState::Phys> &fwd,
                    Field<TData, FieldState::Phys> &bwd)
    {
        // Get number of elements with padding.
        const auto nelmt = phyBlock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto phyptr = phyBlock.template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = phyBlock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize  = (nelmt + blockSize - 1u) / blockSize;

        const auto nTraceBlk = fwd.GetBlocks().size();
        auto &fwdBlock       = fwd.GetBlocks()[0];
        auto &bwdBlock       = bwd.GetBlocks()[0];
        auto fwdptr          = fwdBlock.template GetPtr<MemSpace, WriteOnly>();
        auto bwdptr          = bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        size_t traceSize     = fwd.GetBlocks()[0].CompSize();

        for (unsigned int traceBlk = 1; traceBlk < nTraceBlk; ++traceBlk)
        {
            traceSize += fwd.GetBlocks()[traceBlk].CompSize();
            auto &fwdBlock = fwd.GetBlocks()[traceBlk];
            auto &bwdBlock = bwd.GetBlocks()[traceBlk];
            fwdBlock.template GetPtr<MemSpace, WriteOnly>();
            bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        }

        constexpr unsigned int warpSize = NektarSpaces::Device::warpSize;
        const size_t nLaunchedWarps =
            static_cast<size_t>(gridSize) * (blockSize / warpSize);
        const size_t warpStride = static_cast<size_t>(m_tracePts) * warpSize;
        const size_t wspSize    = nLaunchedWarps * warpStride;

        m_wsp = MemoryRegion<TData>(wspSize);
        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        auto nqOffsetPtr = m_nqOffset.template GetPtr<MemSpace, ReadOnly>();

        auto traceBlockOffsetPtr =
            m_traceBlockOffset.template GetPtr<MemSpace, ReadOnly>();

        auto traceTotOffsetPtr =
            m_traceTotOffset.template GetPtr<MemSpace, ReadOnly>();

        auto traceBlockSizePtr =
            m_traceBlockSize.template GetPtr<MemSpace, ReadOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < phyBlock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                      phyBlock.GetInterleaveWidth(), nelmt,
                                      phyBlock.GetNumData(), (TData *)phyptr);

            if (this->m_fwdOnly)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (GetFwdBwdTracePhys3DKernelLauncher<true>), gridSize,
                    blockSize, 0, m_nqTot, nelmt, m_tracePts, m_nTraces,
                    nqOffsetPtr, nTraceBlk, phyBlock.GetNumComponents(), nc,
                    traceBlockSizePtr, traceBlockOffsetPtr, traceTotOffsetPtr,
                    m_locTracePhysToElmtMaps, m_orientationMaps,
                    m_orientationMapsOffset, m_locToTracePhysOffset,
                    m_isLocTraceLeftAdjacent, m_interpTraceIndex,
                    m_interpPoints, m_interpTypes, m_quadRange, m_interpTrace,
                    m_interpTraceI0, m_interpTraceI0Offset, m_interpTraceI1,
                    m_interpTraceI1Offset, m_interpEndPtI0,
                    m_interpEndPtI0Offset, m_interpEndPtI1,
                    m_interpEndPtI1Offset, wspptr, phyptr, fwdptr, bwdptr);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (GetFwdBwdTracePhys3DKernelLauncher<false>), gridSize,
                    blockSize, 0, m_nqTot, nelmt, m_tracePts, m_nTraces,
                    nqOffsetPtr, nTraceBlk, phyBlock.GetNumComponents(), nc,
                    traceBlockSizePtr, traceBlockOffsetPtr, traceTotOffsetPtr,
                    m_locTracePhysToElmtMaps, m_orientationMaps,
                    m_orientationMapsOffset, m_locToTracePhysOffset,
                    m_isLocTraceLeftAdjacent, m_interpTraceIndex,
                    m_interpPoints, m_interpTypes, m_quadRange, m_interpTrace,
                    m_interpTraceI0, m_interpTraceI0Offset, m_interpTraceI1,
                    m_interpTraceI1Offset, m_interpEndPtI0,
                    m_interpEndPtI0Offset, m_interpEndPtI1,
                    m_interpEndPtI1Offset, wspptr, phyptr, fwdptr, bwdptr);
            }

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, phyBlock.GetNumData(),
                                      (TData *)phyptr);

            // Increment pointers.
            phyptr += phyBlock.CompSize();
        }

        // Set trace block to phys block interleave.
        for (unsigned int traceBlk = 0; traceBlk < fwd.GetBlocks().size();
             ++traceBlk)
        {
            auto &fwdBlock = fwd.GetBlocks()[traceBlk];
            auto &bwdBlock = bwd.GetBlocks()[traceBlk];
            fwdBlock.template SetInterleaveWidth<TData>(1);
            bwdBlock.template SetInterleaveWidth<TData>(1);
        }
    }

    void v_SetTraceBlockOffset(std::vector<size_t> offset) override
    {
        this->m_traceBlockOffset =
            MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(offset);
    }

    void v_SetTraceTotOffset(std::vector<size_t> offset) override
    {
        this->m_traceTotOffset =
            MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(offset);
    }

    void v_SetTraceBlockSize(std::vector<size_t> size) override
    {
        this->m_traceBlockSize =
            MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(size);
    }
};

} // namespace Nektar::Operators::detail
