///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassImpl.hpp
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
// Description: Implementation of the elemental inverse mass operator for the
// standard matrix approach.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <StdRegions/StdExpansion.h>

#include "Operators/OperatorMultiplyByElmtInvMass.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorMultiplyByElmtInvMassImpl
    : public OperatorMultiplyByElmtInvMass<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorMultiplyByElmtInvMassImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMultiplyByElmtInvMass<TData>(expansionList)
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // TODO: Make this Field-only
        auto ncoeffs = this->m_expansionList->GetNcoeffs();

        // Copy the data to from the input field.
        Array<OneD, TData> inArray = in.toArray();
        Array<OneD, TData> outArray(ncoeffs, 0.0);

        // Copy from Explist
        MultiRegions::GlobalMatrixKey mkey(StdRegions::eInvMass);
        const DNekScalBlkMatSharedPtr &InvMass =
            this->m_expansionList->GetBlockMatrix(mkey);

        NekVector<TData> inVec(ncoeffs, inArray, eWrapper);
        NekVector<TData> outVec(ncoeffs, outArray, eWrapper);

        outVec = (*InvMass) * inVec;

        // Copy the data to the output field.
        out.template copyArray<MemSpace>(outArray);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorMultiplyByElmtInvMassImpl<
            ExecSpace, Implementation, TData>>(expansionList);
    }
};

} // namespace Nektar::Operators::detail
