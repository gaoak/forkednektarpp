///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassStdMat.hpp
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

// standard matrix implementation
template <typename TData>
class OperatorMultiplyByElmtInvMassImpl<TData, ImplStdMat>
    : public OperatorMultiplyByElmtInvMass<TData>
{
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
        Array<OneD, TData> inarray(ncoeffs);
        Array<OneD, TData> outarray(ncoeffs, 0.0);

        // Copy data from input field
        // Field -> Array
        auto *inarrptr = inarray.data();
        auto *inptr    = in.GetStorage().GetCPUPtr();
        for (auto const &block : in.GetBlocks())
        {
            auto nSize  = block.block_size;
            auto nElmts = block.num_elements;
            auto nmTot  = block.num_pts;

            std::copy(inptr, inptr + nElmts * nmTot, inarrptr);

            inarrptr += nElmts * nmTot;
            inptr += nSize;
        }

        // Copy from Explist
        MultiRegions::GlobalMatrixKey mkey(StdRegions::eInvMass);
        const DNekScalBlkMatSharedPtr &InvMass =
            this->m_expansionList->GetBlockMatrix(mkey);

        NekVector<TData> invec(ncoeffs, inarray, eWrapper);
        NekVector<TData> outvec(ncoeffs, outarray, eWrapper);

        outvec = (*InvMass) * invec;

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
            OperatorMultiplyByElmtInvMassImpl<TData, ImplStdMat>>(
            expansionList);
    }

    static std::string className;
};

} // namespace Nektar::Operators::detail
