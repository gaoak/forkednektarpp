///////////////////////////////////////////////////////////////////////////////
//
// File: AddTraceIntegralImpl.hpp
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

#include "Operators/ElmtOps/OperatorIProductWRTBase.hpp"
#include "Operators/LoopExecution/LoopExecution.hpp"
#include "Operators/OperatorAddTraceIntegral.hpp"

#include "Operators/AddTraceIntegral/AddTraceIntegralCUDAKernels.cuh"
#include "Operators/AddTraceIntegral/AddTraceIntegralKokkosKernels.hpp"
#include "Operators/AddTraceIntegral/AddTraceIntegralSYCLKernels.hpp"
#include "Operators/AddTraceIntegral/AddTraceIntegralSerialAVXKernels.hpp"

using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorAddTraceIntegralImpl : public OperatorAddTraceIntegral<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorAddTraceIntegralImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAddTraceIntegral<TData>(std::move(expansionList)),
          m_trace(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              GetBlockAttributes<TData>(FieldState::Coeff,
                                        expansionList->GetTrace()),
              1, ExecSpace::alignment))
    {
        // Set mapping to skip over padding elements
        int i, j;

        i = 0, j = 0;
        Array<OneD, int> alignmentMap(expansionList->GetNcoeffs());
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);
        for (auto &block : blocks)
        {
            const auto ncoeff    = block.num_pts;
            const auto nElmts    = block.num_elements;
            const auto nPadElmts = block.num_padding_elements;

            for (unsigned int e = 0; e < nElmts; e++)
            {
                for (unsigned int n = 0; n < ncoeff; n++)
                {
                    alignmentMap[i++] = j++;
                }
            }
            j += nPadElmts * ncoeff;
        }

        i = 0, j = 0;
        Array<OneD, int> alignmentTrace(
            expansionList->GetTrace()->GetNcoeffs());
        auto traceBlocks = GetBlockAttributes<TData>(FieldState::Coeff,
                                                     expansionList->GetTrace());
        for (auto &block : traceBlocks)
        {
            const auto ncoeff    = block.num_pts;
            const auto nElmts    = block.num_elements;
            const auto nPadElmts = block.num_padding_elements;

            for (unsigned int e = 0; e < nElmts; e++)
            {
                for (unsigned int n = 0; n < ncoeff; n++)
                {
                    alignmentTrace[i++] = j++;
                }
            }
            j += nPadElmts * ncoeff;
        }

        // Get Trace-to-Element Map
        auto locTraceToTraceMap = expansionList->GetLocTraceToTraceMap();
        auto &TraceCoeffsToElmtMap =
            locTraceToTraceMap->GetTraceCoeffsToElmtMap()[0];
        auto &TraceCoeffsToElmtSign =
            locTraceToTraceMap->GetTraceCoeffsToElmtSign()[0];
        auto &TraceCoeffsToElmtTrace =
            locTraceToTraceMap->GetTraceCoeffsToElmtTrace()[0];
        Array<OneD, int> alignedTraceCoeffsToElmtMap(
            TraceCoeffsToElmtMap.size());
        Array<OneD, int> alignedTraceCoeffsToElmtTrace(
            TraceCoeffsToElmtTrace.size());

        // Compute aligned map to skip over padding elements
        for (int i = 0; i < TraceCoeffsToElmtMap.size(); i++)
        {
            alignedTraceCoeffsToElmtMap[i] =
                alignmentMap[TraceCoeffsToElmtMap[i]];
        }

        for (int i = 0; i < TraceCoeffsToElmtTrace.size(); i++)
        {
            alignedTraceCoeffsToElmtTrace[i] =
                alignmentTrace[TraceCoeffsToElmtTrace[i]];
        }

        // Assign map to memory region
        const bool device_only = true;

        m_traceCoeffsToElmtMap =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                alignedTraceCoeffsToElmtMap, ExecSpace::alignment, device_only);

        m_traceCoeffsToElmtSign =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                TraceCoeffsToElmtSign, ExecSpace::alignment, device_only);

        m_traceCoeffsToElmtTrace =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                alignedTraceCoeffsToElmtTrace, ExecSpace::alignment,
                device_only);

        // Reorder the map and sign arrays to let trace data be accessed
        // contiguously
        ReorderMap();

        // By default, deinterleave map is 0,1,2,3....
        // the map must be the size of Field, not explist or map
        size_t ncoeffs = 0;
        for (auto &block : blocks)
        {
            ncoeffs += block.block_size;
        }
        // first create an array and then copy to MemoryRegion
        Array<OneD, int> tmpArray(ncoeffs);
        for (int i = 0; i < ncoeffs; i++)
        {
            tmpArray[i] = i;
        }
        m_deInterleaveFieldMap =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                tmpArray, ExecSpace::alignment, device_only);

        // reuse Array for trace
        tmpArray = Array<OneD, int>(m_trace.GetFieldSize());
        for (int i = 0; i < m_trace.GetFieldSize(); i++)
        {
            tmpArray[i] = i;
        }
        m_deInterleaveTraceMap =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                tmpArray, ExecSpace::alignment, device_only);

        m_nFwdBwdCoeffs = locTraceToTraceMap->GetNFwdCoeffs() +
                          locTraceToTraceMap->GetNBwdCoeffs();

        // Initialise IProductWRTBase operator
        m_IProductWRTBaseOp =
            IProductWRTBase<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList->GetTrace());
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Step 1: Inner product for trace integral
        m_IProductWRTBaseOp->apply(in, m_trace);

        // interleave the map only when the input vector width is different
        // from current vector width of map
        /*if (m_trace.GetVecWidth() != m_traceVecWdith)
        {
            if (m_traceVecWdith != 1) // deinterleave first
            {
                ReshuffleMap<ExecSpace>(m_deInterleaveTraceMap,
                                        m_traceCoeffsToElmtTrace);
            }
            if (m_trace.GetVecWidth() != 1) // do interleave
            {
                MemoryRegion<int> InterleaveTraceMap =
                    MemoryRegion<int>::template create<MemSpace>(
                        m_deInterleaveTraceMap.size(), ExecSpace::alignment);

                BuildInterleaveMap<ExecSpace>(
                    m_trace.GetBlocks(), m_trace.GetVecWidth(),
                    m_deInterleaveTraceMap, InterleaveTraceMap);

                ReshuffleMap<ExecSpace>(InterleaveTraceMap,
                                        m_traceCoeffsToElmtTrace);
            }
            // update the vector width of map
            m_traceVecWdith = m_trace.GetVecWidth();
            // reorder the map and sign arrays to let trace data be accessed
            // contiguously
            ReorderMap();
        }

        if (out.GetVecWidth() != m_fieldVecWdith)
        {
            if (m_fieldVecWdith != 1) // deinterleave first
            {
                ReshuffleMap<ExecSpace>(m_deInterleaveFieldMap,
                                        m_traceCoeffsToElmtMap);
            }
            if (out.GetVecWidth() != 1) // do interleave
            {
                MemoryRegion<int> InterleaveFieldMap =
                    MemoryRegion<int>::template create<MemSpace>(
                        m_deInterleaveFieldMap.size(), ExecSpace::alignment);

                BuildInterleaveMap<ExecSpace>(
                    out.GetBlocks(), out.GetVecWidth(), m_deInterleaveFieldMap,
                    InterleaveFieldMap);

                ReshuffleMap<ExecSpace>(InterleaveFieldMap,
                                        m_traceCoeffsToElmtMap);
            }
            // update the vector width of map
            m_fieldVecWdith = out.GetVecWidth();
        }*/

        // Step 2: Map Trace to Element
        AddTraceIntegral(out);
    }

    void AddTraceIntegral(Field<TData, FieldState::Coeff> &out)
    {
        // Return if no trace coefficient.
        if (m_nFwdBwdCoeffs == 0)
        {
            return;
        }

        // Copy memory to the device, if necessary and get raw pointers.
        TData *outPtr         = out.template GetPtr<MemSpace, ReadWrite>();
        const TData *tracePtr = m_trace.template GetPtr<MemSpace, ReadOnly>();
        const int *traceCoeffsToElmtSignPtr =
            m_traceCoeffsToElmtSign.template GetPtr<MemSpace, ReadOnly>();
        const int *traceCoeffsToElmtTracePtr =
            m_traceCoeffsToElmtTrace.template GetPtr<MemSpace, ReadOnly>();
        const int *traceCoeffsToElmtMapPtr =
            m_traceCoeffsToElmtMap.template GetPtr<MemSpace, ReadOnly>();

        AddTraceIntegralKernel<ExecSpace>(
            m_nFwdBwdCoeffs, traceCoeffsToElmtMapPtr, traceCoeffsToElmtSignPtr,
            traceCoeffsToElmtTracePtr, tracePtr, outPtr);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorAddTraceIntegralImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    Field<TData, FieldState::Coeff> m_trace;
    MemoryRegion<int> m_traceCoeffsToElmtMap;
    MemoryRegion<int> m_traceCoeffsToElmtSign;
    MemoryRegion<int> m_traceCoeffsToElmtTrace;
    int m_nFwdBwdCoeffs;

    // stores the current vector width of ElmtMap / TraceMap
    int m_fieldVecWdith = 1;
    int m_traceVecWdith = 1;
    // and the Deinterleave map to restore original layout
    MemoryRegion<int> m_deInterleaveFieldMap;
    MemoryRegion<int> m_deInterleaveTraceMap;

    /// A function to order the map and sign arrays, to let trace data be
    /// accessed contiguously. This does not affect deinterleave map.
    void ReorderMap()
    {
        auto *traceCoeffsToElmtMapPtr =
            m_traceCoeffsToElmtMap.template GetPtr<MemSpace, ReadWrite>();
        auto *traceCoeffsToElmtSignPtr =
            m_traceCoeffsToElmtSign.template GetPtr<MemSpace, ReadWrite>();
        auto *traceCoeffsToElmtTracePtr =
            m_traceCoeffsToElmtTrace.template GetPtr<MemSpace, ReadWrite>();

        ReOrderMapKernel<ExecSpace>(
            m_traceCoeffsToElmtMap.size(), traceCoeffsToElmtMapPtr,
            traceCoeffsToElmtSignPtr, traceCoeffsToElmtTracePtr);
    }
};

} // namespace Nektar::Operators::detail
