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

#include "Operators/IProductWRTBase/IProductWRTBaseSerialStdMat.hpp"
#include "Operators/OperatorAddTraceIntegral.hpp"

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
              GetBlockAttributes(FieldState::Coeff, expansionList->GetTrace())))
    {
        // Get Trace-to-Element Map
        m_locTraceToTraceMap = expansionList->GetLocTraceToTraceMap();

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

        // Step 2: LEGACY Map Trace to Element
        // TODO: Make this Field-only

        // Copy the data to from the trace field.
        Array<OneD, TData> traceArray = m_trace.toArray();
        Array<OneD, TData> outArray(this->m_expansionList->GetNcoeffs(), 0.0);

        // The legacy map (that needs to be vectorised)
        m_locTraceToTraceMap->AddTraceCoeffsToFieldCoeffs(traceArray, outArray);

        // Copy the data to the output field.
        out.template copyArray<MemSpace>(outArray);
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
    Nektar::MultiRegions::LocTraceToTraceMapSharedPtr m_locTraceToTraceMap;

    Field<TData, FieldState::Coeff> m_trace;
};
} // namespace Nektar::Operators::detail
