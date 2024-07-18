///////////////////////////////////////////////////////////////////////////////
//
// File: AddTraceIntegralImplShared.hpp
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

#include "Operators/AddTraceIntegral/AddTraceIntegralCUDASumFacKernels.cuh"
#include "Operators/AddTraceIntegral/AddTraceIntegralKokkosStdMatKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialStdMat.hpp"
#include "Operators/OperatorAddTraceIntegral.hpp"

using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <
    typename ExecSpace, typename Implementation, typename TData,
    typename = typename std::enable_if<
        std::is_same<ExecSpace, NektarSpaces::CUDA>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value>::type>
class OperatorAddTraceIntegralImpl : public OperatorAddTraceIntegral<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorAddTraceIntegralImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAddTraceIntegral<TData>(std::move(expansionList)),
          m_trace(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              GetBlockAttributes(FieldState::Coeff, expansionList->GetTrace())))
    {
        // Get Trace-to-Element Map
        auto locTraceToTraceMap = expansionList->GetLocTraceToTraceMap();

        m_traceCoeffsToElmtMap =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                locTraceToTraceMap->GetTraceCoeffsToElmtMap()[0],
                EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        m_traceCoeffsToElmtSign =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                locTraceToTraceMap->GetTraceCoeffsToElmtSign()[0],
                EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        m_traceCoeffsToElmtTrace =
            MemoryRegion<int>::template fromArray<MemSpace, int>(
                locTraceToTraceMap->GetTraceCoeffsToElmtTrace()[0],
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
        // Copy memory to the device, if necessary and get raw pointers.
        TData *outPtr         = out.template GetPtr<MemSpace, ReadWrite>();
        const TData *tracePtr = m_trace.template GetPtr<MemSpace, ReadOnly>();
        const int *traceCoeffsToElmtMapPtr =
            m_traceCoeffsToElmtMap.template GetPtr<MemSpace, ReadOnly>();
        const int *traceCoeffsToElmtSignPtr =
            m_traceCoeffsToElmtSign.template GetPtr<MemSpace, ReadOnly>();
        const int *traceCoeffsToElmtTracePtr =
            m_traceCoeffsToElmtTrace.template GetPtr<MemSpace, ReadOnly>();

        AddTraceIntegralKernel<ExecSpace, TData>(
            m_nFwdBwdCoeffs, 0, traceCoeffsToElmtMapPtr,
            traceCoeffsToElmtSignPtr, traceCoeffsToElmtTracePtr, tracePtr,
            outPtr);
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
