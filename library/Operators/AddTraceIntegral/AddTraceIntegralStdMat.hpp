///////////////////////////////////////////////////////////////////////////////
//
// File: AddTraceIntegralStdMat.hpp
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

#include "Operators/IProductWRTBase/IProductWRTBaseStdMat.hpp"
#include "Operators/OperatorAddTraceIntegral.hpp"

using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorAddTraceIntegralImpl<TData, ImplStdMat>
    : public OperatorAddTraceIntegral<TData>
{
public:
    OperatorAddTraceIntegralImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAddTraceIntegral<TData>(std::move(expansionList)),
          m_wsp_trace(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList->GetTrace())))
    {
        // Initialise Trace
        const MultiRegions::ExpListSharedPtr &traceExpansionList =
            expansionList->GetTrace();
        nTraceCoeffs = traceExpansionList->GetNcoeffs();
        nFieldCoeffs = expansionList->GetNcoeffs();

        // Get Trace-to-Element Map
        m_locTraceToTraceMap = expansionList->GetLocTraceToTraceMap();

        // Initialise IProductWRTBase operator
        m_IProductWRTBaseOp = IProductWRTBase<>::create(
            this->m_expansionList->GetTrace(), "StdMat");
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Step 1: Inner product for trace integral
        m_IProductWRTBaseOp->apply(in, m_wsp_trace);

        // Step 2: LEGACY Map Trace to Element
        // TODO: Make this Field-only
        Array<OneD, TData> inarray(nTraceCoeffs);
        Array<OneD, TData> outarray(nFieldCoeffs, 0.0);

        // Copy data from input field
        // Field -> Array
        auto *inarrptr = inarray.data();
        auto *inptr    = m_wsp_trace.GetStorage().GetCPUPtr();
        for (auto const &block : m_wsp_trace.GetBlocks())
        {
            auto nSize  = block.block_size;
            auto nElmts = block.num_elements;
            auto nmTot  = block.num_pts;

            std::copy(inptr, inptr + nElmts * nmTot, inarrptr);

            inarrptr += nElmts * nmTot;
            inptr += nSize;
        }

        // The legacy map (that needs to be vectorised)
        m_locTraceToTraceMap->AddTraceCoeffsToFieldCoeffs(inarray, outarray);

        // Copy data to output field
        // Array -> Field
        auto *outarrptr = outarray.data();
        auto *outptr    = out.GetStorage().GetCPUPtr();
        for (auto const &block : out.GetBlocks())
        {
            auto nSize  = block.block_size;
            auto nElmts = block.num_elements;
            auto nmTot  = block.num_pts;

            std::copy(outarrptr, outarrptr + nElmts * nmTot, outptr);

            outarrptr += nElmts * nmTot;
            outptr += nSize;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorAddTraceIntegralImpl<TData, ImplStdMat>>(expansionList);
    }

    static std::string className;

private:
    Field<TData, FieldState::Coeff> m_wsp_trace;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    Nektar::MultiRegions::LocTraceToTraceMapSharedPtr m_locTraceToTraceMap;
    Array<OneD, Array<OneD, int>> m_traceCoeffsToElmtMap;
    Array<OneD, Array<OneD, int>> m_traceCoeffsToElmtSign;
    int nTraceCoeffs;
    int nFieldCoeffs;
};
} // namespace Nektar::Operators::detail
