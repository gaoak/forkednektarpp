///////////////////////////////////////////////////////////////////////////////
//
// File: GetFwdBwdTracePhysSerialAVX.hpp
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

#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysBlockOp.hpp"
#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysSerialAVXKernels.hpp"

#include "Operators/Utils/UtilsKernels.hpp"
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

        m_isLocTraceLeftAdjacent =
            this->m_dataWarehouse->template GetData<MemSpace>(
                IsLocTraceLeftAdjacentKey<TData>(block_idx,
                                                 m_implInterleaveWidth));

        if (m_dimension >= 2)
        {
            m_interpTraceIndex =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    InterpTraceIndexKey<TData>(block_idx,
                                               m_implInterleaveWidth));
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
        }

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
    static constexpr unsigned int m_implInterleaveWidth = 1;

    LibUtilities::ShapeType m_shapeType;
    unsigned int m_nqTot;
    unsigned int m_nTraces;
    unsigned int m_dimension;
    unsigned int m_tracePts = 0;
    MemoryRegion<unsigned int> m_nqOffset;
    MemoryRegion<size_t> m_locToTracePhysOffset;
    const unsigned int *m_locTracePhysToElmtMaps;
    const unsigned int *m_orientationMaps;
    const size_t *m_orientationMapsOffset;
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

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &physBlock,
                 Field<TData, FieldState::Phys> &fwd,
                 Field<TData, FieldState::Phys> &bwd) override
    {

        switch (m_dimension)
        {
            case 1:
            {
                Operator1D(physBlock, fwd, bwd);
                break;
            }
            case 2:
            {
                Operator2D(physBlock, fwd, bwd);
                break;
            }
            case 3:
            {
                Operator3D(physBlock, fwd, bwd);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    void Operator1D(BlockAccessor<TData, FieldState::Phys> &physBlock,
                    Field<TData, FieldState::Phys> &fwd,
                    Field<TData, FieldState::Phys> &bwd)
    {
        // Initialize pointers.
        auto physptr = physBlock.template GetPtr<MemSpace, ReadOnly>();

        const auto nTraceBlk = fwd.GetBlocks().size();
        auto &fwdBlock       = fwd.GetBlocks()[0];
        auto &bwdBlock       = bwd.GetBlocks()[0];
        auto fwdptr          = fwdBlock.template GetPtr<MemSpace, WriteOnly>();
        auto bwdptr          = bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        size_t traceSize     = fwd.GetBlocks()[0].CompSize();

        // Synchronize memory for all blocks.
        for (unsigned int traceBlk = 1; traceBlk < nTraceBlk; ++traceBlk)
        {
            traceSize += fwd.GetBlocks()[traceBlk].CompSize();
            auto &fwdBlock = fwd.GetBlocks()[traceBlk];
            auto &bwdBlock = bwd.GetBlocks()[traceBlk];
            fwdBlock.template GetPtr<MemSpace, WriteOnly>();
            bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        }

        // Get interleave parameter.
        const auto interleaveWidth = physBlock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        auto nqOffsetPtr = m_nqOffset.template GetPtr<MemSpace, ReadOnly>();

        auto locToTracePhysOffsetPtr =
            m_locToTracePhysOffset.template GetPtr<MemSpace, ReadOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < physBlock.GetNumComponents(); ++nc)
        {
            unsigned int el = 0; // element index
            for (size_t e = 0; e < physBlock.GetNumElements(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize,
                                              m_nqTot, (TData *)physptr);
                }

                for (size_t traceId = 0; traceId < m_nTraces; ++traceId)
                {
                    if (this->m_fwdOnly)
                    {
                        GetFwdBwdTracePhys1DKernel<true, TData>(
                            el, physBlock.GetNumElements(),
                            physBlock.GetNumElementsWithPadding(), nc, traceId,
                            m_tracePts, nqOffsetPtr[traceId], m_nTraces,
                            m_locTracePhysToElmtMaps, m_orientationMaps,
                            m_orientationMapsOffset, locToTracePhysOffsetPtr,
                            m_isLocTraceLeftAdjacent, physptr, fwdptr, bwdptr);
                    }
                    else
                    {
                        GetFwdBwdTracePhys1DKernel<false, TData>(
                            el, physBlock.GetNumElements(),
                            physBlock.GetNumElementsWithPadding(), nc, traceId,
                            m_tracePts, nqOffsetPtr[traceId], m_nTraces,
                            m_locTracePhysToElmtMaps, m_orientationMaps,
                            m_orientationMapsOffset, locToTracePhysOffsetPtr,
                            m_isLocTraceLeftAdjacent, physptr, fwdptr, bwdptr);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)physptr - (width_ratio - 1) * m_nqTot);
                }

                el += m_implInterleaveWidth;
                physptr += m_nqTot * m_implInterleaveWidth;
            }

            // Increment pointers.
            physptr +=
                physBlock.CompSize() - physBlock.GetNumElements() * m_nqTot;
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

    void Operator2D(BlockAccessor<TData, FieldState::Phys> &physBlock,
                    Field<TData, FieldState::Phys> &fwd,
                    Field<TData, FieldState::Phys> &bwd)
    {
        // Initialize pointers.
        auto physptr = physBlock.template GetPtr<MemSpace, ReadOnly>();

        const auto nTraceBlk = fwd.GetBlocks().size();
        auto &fwdBlock       = fwd.GetBlocks()[0];
        auto &bwdBlock       = bwd.GetBlocks()[0];
        auto fwdptr          = fwdBlock.template GetPtr<MemSpace, WriteOnly>();
        auto bwdptr          = bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        size_t traceSize     = fwd.GetBlocks()[0].CompSize();

        // Synchronize memory for all blocks.
        for (unsigned int traceBlk = 1; traceBlk < nTraceBlk; ++traceBlk)
        {
            traceSize += fwd.GetBlocks()[traceBlk].CompSize();
            auto &fwdBlock = fwd.GetBlocks()[traceBlk];
            auto &bwdBlock = bwd.GetBlocks()[traceBlk];
            fwdBlock.template GetPtr<MemSpace, WriteOnly>();
            bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        }

        // Get interleave parameter.
        const auto interleaveWidth = physBlock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        auto nqOffsetPtr = m_nqOffset.template GetPtr<MemSpace, ReadOnly>();

        auto locToTracePhysOffsetPtr =
            m_locToTracePhysOffset.template GetPtr<MemSpace, ReadOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < physBlock.GetNumComponents(); ++nc)
        {
            unsigned int el = 0; // element index
            for (size_t e = 0; e < physBlock.GetNumElements(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize,
                                              m_nqTot, (TData *)physptr);
                }

                for (size_t traceId = 0; traceId < m_nTraces; ++traceId)
                {
                    if (this->m_fwdOnly)
                    {
                        GetFwdBwdTracePhys2DKernel<true, TData>(
                            el, physBlock.GetNumElements(),
                            physBlock.GetNumElementsWithPadding(), nc, traceId,
                            m_tracePts, nqOffsetPtr[traceId], m_nTraces,
                            m_locTracePhysToElmtMaps, m_orientationMaps,
                            m_orientationMapsOffset, locToTracePhysOffsetPtr,
                            m_isLocTraceLeftAdjacent, m_interpTraceIndex,
                            m_interpPoints, m_interpTypes, m_quadRange,
                            m_interpTrace, m_interpTraceI0,
                            m_interpTraceI0Offset, m_interpEndPtI0,
                            m_interpEndPtI0Offset, physptr, fwdptr, bwdptr);
                    }
                    else
                    {
                        GetFwdBwdTracePhys2DKernel<false, TData>(
                            el, physBlock.GetNumElements(),
                            physBlock.GetNumElementsWithPadding(), nc, traceId,
                            m_tracePts, nqOffsetPtr[traceId], m_nTraces,
                            m_locTracePhysToElmtMaps, m_orientationMaps,
                            m_orientationMapsOffset, locToTracePhysOffsetPtr,
                            m_isLocTraceLeftAdjacent, m_interpTraceIndex,
                            m_interpPoints, m_interpTypes, m_quadRange,
                            m_interpTrace, m_interpTraceI0,
                            m_interpTraceI0Offset, m_interpEndPtI0,
                            m_interpEndPtI0Offset, physptr, fwdptr, bwdptr);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)physptr - (width_ratio - 1) * m_nqTot);
                }

                el += m_implInterleaveWidth;
                physptr += m_nqTot * m_implInterleaveWidth;
            }

            // Increment pointers.
            physptr +=
                physBlock.CompSize() - physBlock.GetNumElements() * m_nqTot;
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

    void Operator3D(BlockAccessor<TData, FieldState::Phys> &physBlock,
                    Field<TData, FieldState::Phys> &fwd,
                    Field<TData, FieldState::Phys> &bwd)
    {
        // Initialize pointers.
        auto physptr = physBlock.template GetPtr<MemSpace, ReadOnly>();

        const auto nTraceBlk = fwd.GetBlocks().size();
        auto &fwdBlock       = fwd.GetBlocks()[0];
        auto &bwdBlock       = bwd.GetBlocks()[0];
        auto fwdptr          = fwdBlock.template GetPtr<MemSpace, WriteOnly>();
        auto bwdptr          = bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        size_t traceSize     = fwd.GetBlocks()[0].CompSize();

        // Synchronize memory for all blocks.
        for (unsigned int traceBlk = 1; traceBlk < nTraceBlk; ++traceBlk)
        {
            traceSize += fwd.GetBlocks()[traceBlk].CompSize();
            auto &fwdBlock = fwd.GetBlocks()[traceBlk];
            auto &bwdBlock = bwd.GetBlocks()[traceBlk];
            fwdBlock.template GetPtr<MemSpace, WriteOnly>();
            bwdBlock.template GetPtr<MemSpace, WriteOnly>();
        }

        // Get interleave parameter.
        const auto interleaveWidth = physBlock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Set to new interleave width.
        physBlock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetWorkSpace<MemSpace>(m_tracePts);

        auto nqOffsetPtr = m_nqOffset.template GetPtr<MemSpace, ReadOnly>();

        auto locToTracePhysOffsetPtr =
            m_locToTracePhysOffset.template GetPtr<MemSpace, ReadOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < physBlock.GetNumComponents(); ++nc)
        {
            unsigned int el = 0; // element index
            for (size_t e = 0; e < physBlock.GetNumElements(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize,
                                              m_nqTot, (TData *)physptr);
                }

                for (size_t traceId = 0; traceId < m_nTraces; ++traceId)
                {
                    if (this->m_fwdOnly)
                    {
                        GetFwdBwdTracePhys3DKernel<true, TData>(
                            el, physBlock.GetNumElements(),
                            physBlock.GetNumElementsWithPadding(), nc, traceId,
                            m_tracePts, nqOffsetPtr[traceId], m_nTraces,
                            m_locTracePhysToElmtMaps, m_orientationMaps,
                            m_orientationMapsOffset, locToTracePhysOffsetPtr,
                            m_isLocTraceLeftAdjacent, m_interpTraceIndex,
                            m_interpPoints, m_interpTypes, m_quadRange,
                            m_interpTrace, m_interpTraceI0,
                            m_interpTraceI0Offset, m_interpTraceI1,
                            m_interpTraceI1Offset, m_interpEndPtI0,
                            m_interpEndPtI0Offset, m_interpEndPtI1,
                            m_interpEndPtI1Offset, wspptr, physptr, fwdptr,
                            bwdptr);
                    }
                    else
                    {
                        GetFwdBwdTracePhys3DKernel<false, TData>(
                            el, physBlock.GetNumElements(),
                            physBlock.GetNumElementsWithPadding(), nc, traceId,
                            m_tracePts, nqOffsetPtr[traceId], m_nTraces,
                            m_locTracePhysToElmtMaps, m_orientationMaps,
                            m_orientationMapsOffset, locToTracePhysOffsetPtr,
                            m_isLocTraceLeftAdjacent, m_interpTraceIndex,
                            m_interpPoints, m_interpTypes, m_quadRange,
                            m_interpTrace, m_interpTraceI0,
                            m_interpTraceI0Offset, m_interpTraceI1,
                            m_interpTraceI1Offset, m_interpEndPtI0,
                            m_interpEndPtI0Offset, m_interpEndPtI1,
                            m_interpEndPtI1Offset, wspptr, physptr, fwdptr,
                            bwdptr);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)physptr - (width_ratio - 1) * m_nqTot);
                }

                el += m_implInterleaveWidth;
                physptr += m_nqTot * m_implInterleaveWidth;
            }

            // Increment pointers.
            physptr +=
                physBlock.CompSize() - physBlock.GetNumElements() * m_nqTot;
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

    void v_SetTracePhysOffset(std::vector<size_t> offset) override
    {
        this->m_locToTracePhysOffset =
            MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(offset);
    }
};

} // namespace Nektar::Operators::detail
