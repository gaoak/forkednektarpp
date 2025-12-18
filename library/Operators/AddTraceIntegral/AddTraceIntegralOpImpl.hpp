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
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/AddTraceIntegral/AddTraceIntegralDeviceKernels.hpp"
#include "Operators/AddTraceIntegral/AddTraceIntegralSerialAVXKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class AddTraceIntegralOpImpl : public AddTraceIntegralOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AddTraceIntegralOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                           const std::vector<std::string> &components)
        : AddTraceIntegralOp<TData>(expansionList, components),
          m_trace(Field<TData, FieldState::Coeff>(
              GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList->GetTrace()),
              components, 1))
    {
        m_trace.template Initialize<MemSpace>(0.0);

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

        // Compute trace block bound.
        auto traceBlockAttr = GetBlockAttributes<TData, FieldState::Coeff>(
            expansionList->GetTrace());
        std::vector<size_t> traceBlockBound(traceBlockAttr.size());
        std::vector<std::vector<size_t>> toInterleavedTraceBlock;
        size_t traceBound = 0;
        for (unsigned int blk = 0; blk < traceBlockAttr.size(); ++blk)
        {
            const auto &block = traceBlockAttr[blk];
            const auto ncoeff = block.GetNumData();
            const auto nelmt  = block.GetNumElements();
            traceBound += nelmt * ncoeff;
            traceBlockBound[blk] = traceBound;

            // Mapping to interleaved format for trace block.
            toInterleavedTraceBlock.push_back(
                std::vector<size_t>(block.CompSize()));
            auto ptr = toInterleavedTraceBlock[blk].data();
            for (size_t e = 0;
                 e < block.GetNumElmtGroups(
                         NektarSpaces::vector_width<ExecSpace, TData>::value);
                 e++)
            {
                for (unsigned int l = 0;
                     l < NektarSpaces::vector_width<ExecSpace, TData>::value;
                     l++)
                {
                    for (unsigned int i = 0; i < ncoeff; i++)
                    {
                        ptr[l * ncoeff + i] =
                            l + (e * block.GetNumData() + i) *
                                    NektarSpaces::vector_width<ExecSpace,
                                                               TData>::value;
                    }
                }
                ptr += block.GetNumData() *
                       NektarSpaces::vector_width<ExecSpace, TData>::value;
            }
        }

        // Compute block bound.
        auto blockAttr =
            GetBlockAttributes<TData, FieldState::Coeff>(expansionList);
        std::vector<size_t> blockBound(blockAttr.size());
        std::vector<std::vector<size_t>> toInterleavedBlock;
        size_t bound = 0;
        for (unsigned int blk = 0; blk < blockAttr.size(); ++blk)
        {
            const auto &block = blockAttr[blk];
            const auto ncoeff = block.GetNumData();
            const auto nelmt  = block.GetNumElements();
            bound += nelmt * ncoeff;
            blockBound[blk] = bound;

            // Mapping to interleaved format for block.
            toInterleavedBlock.push_back(std::vector<size_t>(block.CompSize()));
            auto ptr = toInterleavedBlock[blk].data();
            for (size_t e = 0;
                 e < block.GetNumElmtGroups(
                         NektarSpaces::vector_width<ExecSpace, TData>::value);
                 e++)
            {
                for (unsigned int l = 0;
                     l < NektarSpaces::vector_width<ExecSpace, TData>::value;
                     l++)
                {
                    for (unsigned int i = 0; i < ncoeff; i++)
                    {
                        ptr[l * ncoeff + i] =
                            l + (e * block.GetNumData() + i) *
                                    NektarSpaces::vector_width<ExecSpace,
                                                               TData>::value;
                    }
                }
                ptr += block.GetNumData() *
                       NektarSpaces::vector_width<ExecSpace, TData>::value;
            }
        }

        // Re-order to avoid race condition on GPUs.
        std::vector<std::tuple<size_t, size_t, int>> traceToTraceReordered(
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
                      [](std::tuple<size_t, size_t, int> const &t1,
                         std::tuple<size_t, size_t, int> const &t2) {
                          return std::get<1>(t1) < std::get<1>(t2);
                      });
        }

        // Assign map to memory region.
        std::vector<size_t> nFwdBwdCoeffsBlock(blockAttr.size(), 0);
        std::vector<std::vector<size_t>> traceCoeffsToElmtMapBlock(
            blockAttr.size());
        std::vector<std::vector<size_t>> traceCoeffsToElmtTraceBlock(
            blockAttr.size());
        std::vector<std::vector<int>> traceCoeffsToElmtSignBlock(
            blockAttr.size());
        m_nFwdBwdCoeffsBlock =
            std::vector<std::vector<size_t>>(traceBlockAttr.size());
        m_traceCoeffsToElmtMap = std::vector<std::vector<MemoryRegion<size_t>>>(
            traceBlockAttr.size());
        m_traceCoeffsToElmtTrace =
            std::vector<std::vector<MemoryRegion<size_t>>>(
                traceBlockAttr.size());
        m_traceCoeffsToElmtSign =
            std::vector<std::vector<MemoryRegion<int>>>(traceBlockAttr.size());
        unsigned int blk0 = 0, blk1 = 0;
        size_t i = 0, offset0 = 0, offset1 = 0;
        while (blk1 < traceBlockAttr.size())
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
                            traceCoeffsToElmtMap));
                    traceCoeffsToElmtMap.clear();
                }
                for (auto &traceCoeffsToElmtTrace : traceCoeffsToElmtTraceBlock)
                {
                    m_traceCoeffsToElmtTrace[blk1].push_back(
                        MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                  size_t>(
                            traceCoeffsToElmtTrace));
                    traceCoeffsToElmtTrace.clear();
                }
                for (auto &traceCoeffsToElmtSign : traceCoeffsToElmtSignBlock)
                {
                    m_traceCoeffsToElmtSign[blk1].push_back(
                        MemoryRegion<int>::template FromVector<MemSpace, int>(
                            traceCoeffsToElmtSign));
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
                // TODO: This should be update on a per block basis
                auto session        = expansionList->GetSession();
                std::string execStr = Operator<TData>::GetOpExecSpace(session);
                std::string implStr = m_IProductWRTBaseOp->GetOpImpl(
                    m_IProductWRTBaseOp->name, ExecSpace::name, session);

                if ((execStr == "AVX" && implStr == "StdMat") ||
                    (execStr == "AVX" && implStr == "SumFac") ||
                    (execStr == "Device" && implStr == "SumFac"))
                {
                    traceCoeffsToElmtMapBlock[blk0].push_back(
                        toInterleavedBlock[blk0][std::get<0>(
                                                     traceToTraceReordered[i]) -
                                                 offset0]);
                }
                else
                {
                    traceCoeffsToElmtMapBlock[blk0].push_back(
                        std::get<0>(traceToTraceReordered[i]) - offset0);
                }

                // TODO: This should be update on a per trace block basis
                if ((execStr == "AVX" && implStr == "StdMat") ||
                    (execStr == "AVX" && implStr == "SumFac") ||
                    (execStr == "Device" && implStr == "SumFac"))
                {
                    traceCoeffsToElmtTraceBlock[blk0].push_back(
                        toInterleavedTraceBlock
                            [blk1]
                            [std::get<1>(traceToTraceReordered[i]) - offset1]);
                }
                else
                {
                    traceCoeffsToElmtTraceBlock[blk0].push_back(
                        std::get<1>(traceToTraceReordered[i]) - offset1);
                }

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
            this->m_expansionList->GetTrace(), components, ExecSpace::name);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AddTraceIntegralOpImpl<ExecSpace, TData>>(
            expansionList, components);
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

            // TODO: This should be update on a per block basis
            auto session        = this->m_expansionList->GetSession();
            std::string execStr = Operator<TData>::GetOpExecSpace(session);
            std::string implStr = m_IProductWRTBaseOp->GetOpImpl(
                m_IProductWRTBaseOp->name, ExecSpace::name, session);

            // Check interleaving for trace block.
            if ((execStr == "AVX" && implStr == "StdMat") ||
                (execStr == "AVX" && implStr == "SumFac") ||
                (execStr == "Device" && implStr == "SumFac"))
            {
                auto tracePtr = m_trace.GetBlocks()[blk1]
                                    .template GetPtr<MemSpace, ReadWrite>();

                ReshapeStorage<ExecSpace>(
                    NektarSpaces::vector_width<ExecSpace, TData>::value,
                    m_trace.GetBlocks()[blk1].GetInterleaveWidth(),
                    m_trace.GetBlocks()[blk1].GetNumElementsWithPadding(),
                    m_trace.GetBlocks()[blk1].GetNumData(), tracePtr);

                m_trace.GetBlocks()[blk1].template SetInterleaveWidth<TData>(
                    NektarSpaces::vector_width<ExecSpace, TData>::value);
            }

            for (unsigned int blk0 = 0; blk0 < out.GetBlocks().size(); ++blk0)
            {
                auto nFwdBwdCoeffsBlock = m_nFwdBwdCoeffsBlock[blk1][blk0];

                if (nFwdBwdCoeffsBlock > 0)
                {
                    // Initialize pointers.
                    auto outptr = out.GetBlocks()[blk0]
                                      .template GetPtr<MemSpace, ReadWrite>();
                    auto traceCoeffsToElmtMapPtr =
                        m_traceCoeffsToElmtMap[blk1][blk0]
                            .template GetPtr<MemSpace, ReadOnly>();
                    auto traceCoeffsToElmtSignPtr =
                        m_traceCoeffsToElmtSign[blk1][blk0]
                            .template GetPtr<MemSpace, ReadOnly>();
                    auto traceCoeffsToElmtTracePtr =
                        m_traceCoeffsToElmtTrace[blk1][blk0]
                            .template GetPtr<MemSpace, ReadOnly>();

                    // TODO: This should be update on a per block basis
                    std::string execStr =
                        Operator<TData>::GetOpExecSpace(session);
                    std::string implStr = m_IProductWRTBaseOp->GetOpImpl(
                        m_IProductWRTBaseOp->name, ExecSpace::name, session);

                    // Check interleaving for output block.
                    if ((execStr == "AVX" && implStr == "StdMat") ||
                        (execStr == "AVX" && implStr == "SumFac") ||
                        (execStr == "Device" && implStr == "SumFac"))
                    {
                        ReshapeStorage<ExecSpace>(
                            NektarSpaces::vector_width<ExecSpace, TData>::value,
                            out.GetBlocks()[blk0].GetInterleaveWidth(),
                            out.GetBlocks()[blk0].GetNumElementsWithPadding(),
                            out.GetBlocks()[blk0].GetNumData(), outptr);
                    }

                    // AddTraceIntegralKernel.
                    AddTraceIntegralKernel<ExecSpace>(
                        nFwdBwdCoeffsBlock, traceCoeffsToElmtMapPtr,
                        traceCoeffsToElmtSignPtr, traceCoeffsToElmtTracePtr,
                        tracePtr, outptr);

                    // Update interleaving for output block.
                    if ((execStr == "AVX" && implStr == "StdMat") ||
                        (execStr == "AVX" && implStr == "SumFac") ||
                        (execStr == "Device" && implStr == "SumFac"))
                    {
                        out.GetBlocks()[blk0]
                            .template SetInterleaveWidth<TData>(
                                NektarSpaces::vector_width<ExecSpace,
                                                           TData>::value);
                    }
                }
            }
        }
    }
};

} // namespace Nektar::Operators::detail
