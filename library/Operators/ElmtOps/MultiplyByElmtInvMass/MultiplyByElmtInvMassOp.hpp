///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassOp.hpp
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

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassBlockOp.hpp"

namespace Nektar::Operators
{

// MultiplyByElmtInvMass base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class MultiplyByElmtInvMassOp
    : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>;

public:
    static std::shared_ptr<MultiplyByElmtInvMassOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto op = ElmtOp<FieldState::Coeff, FieldState::Coeff,
                         TData>::template Create<MultiplyByElmtInvMassOp,
                                                 MultiplyByElmtInvMassBlockOp>(
            expansionList, execStr, implStr);

        // Loop over the blocks.
        auto blockAttr =
            GetBlockAttributes<TData, FieldState::Coeff>(expansionList);

        for (unsigned int blk = 0; blk < op->m_blockOp.size(); ++blk)
        {
            std::vector<TData> dmat;
            const auto exp_idx = GetCollection(expansionList, blk)
                                     .GetExpVector()[0]
                                     ->GetElmtId();
            const auto exp   = expansionList->GetExp(exp_idx);
            const auto nmTot = exp->GetNcoeffs();
            const auto deformed =
                exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
            const auto nelmt    = blockAttr[blk].GetNumElements();
            const auto nelmtPad = blockAttr[blk].GetNumElementsWithPadding();

            if (deformed)
            {
                dmat.resize(nelmtPad * nmTot * nmTot);
                auto dmatptr = dmat.data();
                for (size_t e = 0; e < nelmt; ++e)
                {
                    const auto exp = expansionList->GetExp(exp_idx + e);
                    const auto &InvMass =
                        exp->GetLocMatrix(StdRegions::eInvMass);
                    std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot, dmatptr);
                    dmatptr += nmTot * nmTot;
                }
            }

            op->m_blockOp[blk]->SetInvMassMatrix(dmat);
        }

        return op;
    }

    static inline const std::string name = "MultiplyByElmtInvMass";

protected:
    std::vector<std::shared_ptr<MultiplyByElmtInvMassBlockOp<TData>>> m_blockOp;

    MultiplyByElmtInvMassOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList)
    {
    }

    ~MultiplyByElmtInvMassOp() override = default;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, outblock);
        }
    }
};

} // namespace Nektar::Operators
