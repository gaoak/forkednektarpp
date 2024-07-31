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
#include "Operators/OperatorAddTraceIntegral.hpp"

#include "Operators/AddTraceIntegral/AddTraceIntegralCUDAKernels.cuh"
#include "Operators/AddTraceIntegral/AddTraceIntegralKernels.hpp"
#include "Operators/AddTraceIntegral/AddTraceIntegralKokkosKernels.hpp"
#include "Operators/AddTraceIntegral/AddTraceIntegralSYCLKernels.hpp"

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
              GetBlockAttributes(FieldState::Coeff, expansionList->GetTrace(),
                                 vec_t::width),
              1, vec_t::alignment))
    {
        // Set mapping to skip over padding elements
        int i, j;

        i = 0, j = 0;
        Array<OneD, int> alignmentMap(expansionList->GetNcoeffs());
        auto blocks =
            GetBlockAttributes(FieldState::Coeff, expansionList, vec_t::width);
        for (auto &block : blocks)
        {
            auto const ncoeff    = block.num_pts;
            auto const nElmts    = block.num_elements;
            auto const nPadElmts = block.num_padding_elements;
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
        auto traceBlocks = GetBlockAttributes(
            FieldState::Coeff, expansionList->GetTrace(), vec_t::width);
        for (auto &block : traceBlocks)
        {
            auto const ncoeff    = block.num_pts;
            auto const nElmts    = block.num_elements;
            auto const nPadElmts = block.num_padding_elements;
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
        m_traceCoeffsToElmtMap =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                alignedTraceCoeffsToElmtMap,
                EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        m_traceCoeffsToElmtSign =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                TraceCoeffsToElmtSign,
                EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        m_traceCoeffsToElmtTrace =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                alignedTraceCoeffsToElmtTrace,
                EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

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
        const int *traceCoeffsToElmtMapPtr =
            m_traceCoeffsToElmtMap.template GetPtr<MemSpace, ReadOnly>();
        const int *traceCoeffsToElmtSignPtr =
            m_traceCoeffsToElmtSign.template GetPtr<MemSpace, ReadOnly>();
        const int *traceCoeffsToElmtTracePtr =
            m_traceCoeffsToElmtTrace.template GetPtr<MemSpace, ReadOnly>();

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
};

} // namespace Nektar::Operators::detail
