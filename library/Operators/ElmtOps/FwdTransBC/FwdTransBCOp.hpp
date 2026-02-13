///////////////////////////////////////////////////////////////////////////////
//
// File: FwdTransBCOp.hpp
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

#include "LocalRegions/MatrixKey.h"
#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/FwdTransBC/FwdTransBCBlockOp.hpp"

namespace Nektar::Operators
{

// FwdTransBC base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class FwdTransBCOp : public ElmtOp<FieldState::Phys, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Coeff, TData>;

public:
    static std::shared_ptr<FwdTransBCOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto op =
            ElmtOp<FieldState::Phys, FieldState::Coeff, TData>::template Create<
                FwdTransBCOp, FwdTransBCBlockOp>(expansionList, components,
                                                 execStr, implStr);

        /// Fetch inverse interior mass matrices
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
            const auto nmInt = exp->GetNcoeffs() - exp->NumBndryCoeffs();
            const auto deformed =
                exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
            const auto nelmt     = blockAttr[blk].GetNumElements();
            const auto nelmtPad  = blockAttr[blk].GetNumElementsWithPadding();
            const auto shapeType = exp->DetShapeType();

            if (deformed)
            {
                dmat.resize(nelmtPad * nmInt * nmInt);
                auto dmatptr = dmat.data();
                for (size_t e = 0; e < nelmt; ++e)
                {
                    const auto expe = expansionList->GetExp(exp_idx + e);

                    // Get statically condensed matrix system for interior
                    // matrix
                    LocalRegions::MatrixKey masskey(StdRegions::eMass,
                                                    shapeType, *expe);
                    const auto &InvIntMass =
                        expe->GetLocStaticCondMatrix(masskey)->GetBlock(1, 1);
                    std::copy_n(InvIntMass->GetRawPtr(), nmInt * nmInt,
                                dmatptr);
                    dmatptr += nmInt * nmInt;
                }
            }

            op->m_blockOp[blk]->SetInvMassMatrix(dmat);
        }

        return op;
    }

    static inline const std::string name = "FwdTransBC";

protected:
    std::vector<std::shared_ptr<FwdTransBCBlockOp<TData>>> m_blockOp;

    FwdTransBCOp(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Coeff, TData>(expansionList,
                                                             components)
    {
    }

    ~FwdTransBCOp() override = default;

    void v_Apply(Field<TData, FieldState::Phys> &in,
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
