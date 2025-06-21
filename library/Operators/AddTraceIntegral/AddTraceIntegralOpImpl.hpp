///////////////////////////////////////////////////////////////////////////////
//
// File: AddTraceIntegralOpImpl.hpp
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

#include "Operators/AddTraceIntegral/AddTraceIntegralOp.hpp"

#include "Operators/AddTraceIntegral/AddTraceIntegralDeviceKernels.hpp"
#include "Operators/AddTraceIntegral/AddTraceIntegralSerialAVXKernels.hpp"

using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class AddTraceIntegralOpImpl : public AddTraceIntegralOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AddTraceIntegralOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : AddTraceIntegralOp<TData>(std::move(expansionList)),
          m_trace(Field<TData, FieldState::Coeff>::Create(
              GetBlockAttributes<TData>(FieldState::Coeff,
                                        expansionList->GetTrace()),
              1, ExecSpace::alignment, eDeviceOnly))
    {
        // Get Trace-to-Element Map
        auto locTraceToTraceMap = expansionList->GetLocTraceToTraceMap();
        auto &TraceCoeffsToElmtMap =
            locTraceToTraceMap->GetTraceCoeffsToElmtMap()[0];
        auto &TraceCoeffsToElmtTrace =
            locTraceToTraceMap->GetTraceCoeffsToElmtTrace()[0];
        auto &TraceCoeffsToElmtSign =
            locTraceToTraceMap->GetTraceCoeffsToElmtSign()[0];

        m_nFwdBwdCoeffs = locTraceToTraceMap->GetNFwdCoeffs() +
                          locTraceToTraceMap->GetNBwdCoeffs();

        // Split trace map per block.
        if (m_nFwdBwdCoeffs == 0)
        {
            return;
        }

        // Compute block bound.
        auto traceBlocks = GetBlockAttributes<TData>(FieldState::Coeff,
                                                     expansionList->GetTrace());
        std::vector<size_t> traceBlockBound(traceBlocks.size());
        size_t traceBound = 0;
        for (unsigned int blk = 0; blk < traceBlocks.size(); ++blk)
        {
            const auto &block = traceBlocks[blk];
            const auto ncoeff = block.GetNumData();
            const auto nelmt  = block.GetNumElements();
            traceBound += nelmt * ncoeff;
            traceBlockBound[blk] = traceBound;
        }

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);
        std::vector<size_t> blockBound(blocks.size());
        size_t bound = 0;
        for (unsigned int blk = 0; blk < blocks.size(); ++blk)
        {
            const auto &block = blocks[blk];
            const auto ncoeff = block.GetNumData();
            const auto nelmt  = block.GetNumElements();
            bound += nelmt * ncoeff;
            blockBound[blk] = bound;
        }

        std::vector<std::tuple<size_t, size_t, size_t>> traceToTraceReordered(
            m_nFwdBwdCoeffs);
        if (m_nFwdBwdCoeffs > 0)
        {
            for (size_t i = 0; i < m_nFwdBwdCoeffs; i++)
            {
                traceToTraceReordered[i] = std::make_tuple(
                    TraceCoeffsToElmtMap[i], TraceCoeffsToElmtTrace[i],
                    TraceCoeffsToElmtSign[i]);
            }
            std::sort(std::begin(traceToTraceReordered),
                      std::end(traceToTraceReordered),
                      [](std::tuple<size_t, size_t, size_t> const &t1,
                         std::tuple<size_t, size_t, size_t> const &t2) {
                          return std::get<1>(t1) < std::get<1>(t2);
                      });
        }

        // Assign map to memory region.
        std::vector<size_t> nFwdBwdCoeffsBlock(blocks.size(), 0);
        std::vector<std::vector<size_t>> traceCoeffsToElmtMapBlock(
            blocks.size());
        std::vector<std::vector<size_t>> traceCoeffsToElmtTraceBlock(
            blocks.size());
        std::vector<std::vector<int>> traceCoeffsToElmtSignBlock(blocks.size());
        m_nFwdBwdCoeffsBlock =
            std::vector<std::vector<size_t>>(traceBlocks.size());
        m_traceCoeffsToElmtMap =
            std::vector<std::vector<MemoryRegion<size_t>>>(traceBlocks.size());
        m_traceCoeffsToElmtTrace =
            std::vector<std::vector<MemoryRegion<size_t>>>(traceBlocks.size());
        m_traceCoeffsToElmtSign =
            std::vector<std::vector<MemoryRegion<int>>>(traceBlocks.size());
        unsigned int blk0 = 0, blk1 = 0;
        size_t i = 0, offset0 = 0, offset1 = 0;
        while (blk1 < traceBlocks.size())
        {
            if (i == m_nFwdBwdCoeffs ||
                std::get<1>(traceToTraceReordered[i]) >= traceBlockBound[blk1])
            {
                for (auto &nFwdBwdCoeffs : nFwdBwdCoeffsBlock)
                {
                    m_nFwdBwdCoeffsBlock[blk1].push_back(nFwdBwdCoeffs);
                    nFwdBwdCoeffs = 0;
                }
                for (auto &traceCoeffsToElmtMap : traceCoeffsToElmtMapBlock)
                {
                    m_traceCoeffsToElmtMap[blk1].push_back(
                        MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                  size_t>(
                            traceCoeffsToElmtMap, ExecSpace::alignment));
                    traceCoeffsToElmtMap.clear();
                }
                for (auto &traceCoeffsToElmtTrace : traceCoeffsToElmtTraceBlock)
                {
                    m_traceCoeffsToElmtTrace[blk1].push_back(
                        MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                  size_t>(
                            traceCoeffsToElmtTrace, ExecSpace::alignment));
                    traceCoeffsToElmtTrace.clear();
                }
                for (auto &traceCoeffsToElmtSign : traceCoeffsToElmtSignBlock)
                {
                    m_traceCoeffsToElmtSign[blk1].push_back(
                        MemoryRegion<int>::template FromVector<MemSpace, int>(
                            traceCoeffsToElmtSign, ExecSpace::alignment));
                    traceCoeffsToElmtSign.clear();
                }
                offset1 = traceBlockBound[blk1];
                blk1++;
            }
            else if (std::get<0>(traceToTraceReordered[i]) >= blockBound[blk0])
            {
                offset0 = blockBound[blk0];
                blk0++;
            }
            else
            {
                traceCoeffsToElmtMapBlock[blk0].push_back(
                    std::get<0>(traceToTraceReordered[i]) - offset0);
                traceCoeffsToElmtTraceBlock[blk0].push_back(
                    std::get<1>(traceToTraceReordered[i]) - offset1);
                traceCoeffsToElmtSignBlock[blk0].push_back(
                    std::get<2>(traceToTraceReordered[i]));
                nFwdBwdCoeffsBlock[blk0]++;
                offset0 = 0;
                blk0    = 0;
                i++;
            }
        }

        // Initialise IProductWRTBase operator.
        m_IProductWRTBaseOp = IProductWRTBaseOp<TData>::Create(
            this->m_expansionList->GetTrace(), ExecSpace::name);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<AddTraceIntegralOpImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    std::shared_ptr<IProductWRTBaseOp<TData>> m_IProductWRTBaseOp;
    Field<TData, FieldState::Coeff> m_trace;
    std::vector<std::vector<MemoryRegion<size_t>>> m_traceCoeffsToElmtMap;
    std::vector<std::vector<MemoryRegion<size_t>>> m_traceCoeffsToElmtTrace;
    std::vector<std::vector<MemoryRegion<int>>> m_traceCoeffsToElmtSign;
    std::vector<std::vector<size_t>> m_nFwdBwdCoeffsBlock;
    size_t m_nFwdBwdCoeffs;

    void v_Apply(Field<TData, FieldState::Phys> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Step 1: Inner product for trace integral.
        m_IProductWRTBaseOp->Apply(in, m_trace);

        // Step 2: Map Trace to element.
        AddTraceIntegral(out);
    }

    void AddTraceIntegral(Field<TData, FieldState::Coeff> &out)
    {
        // Return if no trace coefficient.
        if (m_nFwdBwdCoeffs == 0)
        {
            return;
        }

        for (unsigned int blk1 = 0; blk1 < m_trace.GetBlocks().size(); ++blk1)
        {
            // Initialize pointers.
            auto tracePtr =
                m_trace.GetBlocks()[blk1].template GetPtr<MemSpace, ReadOnly>();
            for (unsigned int blk0 = 0; blk0 < out.GetBlocks().size(); ++blk0)
            {
                auto nFwdBwdCoeffsBlock = m_nFwdBwdCoeffsBlock[blk1][blk0];

                if (nFwdBwdCoeffsBlock > 0)
                {
                    // Initialize pointers.
                    auto outptr = out.GetBlocks()[blk0]
                                      .template GetPtr<MemSpace, ReadWrite>();
                    auto traceCoeffsToElmtSignPtr =
                        m_traceCoeffsToElmtSign[blk1][blk0]
                            .template GetPtr<MemSpace, ReadOnly>();
                    auto traceCoeffsToElmtTracePtr =
                        m_traceCoeffsToElmtTrace[blk1][blk0]
                            .template GetPtr<MemSpace, ReadOnly>();
                    auto traceCoeffsToElmtMapPtr =
                        m_traceCoeffsToElmtMap[blk1][blk0]
                            .template GetPtr<MemSpace, ReadOnly>();

                    AddTraceIntegralKernel<ExecSpace>(
                        nFwdBwdCoeffsBlock, traceCoeffsToElmtMapPtr,
                        traceCoeffsToElmtSignPtr, traceCoeffsToElmtTracePtr,
                        tracePtr, outptr);
                }
            }
        }
    }
};

} // namespace Nektar::Operators::detail
