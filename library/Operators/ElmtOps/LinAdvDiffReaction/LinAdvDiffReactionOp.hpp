///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionOp.hpp
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

#include "Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionBlockOp.hpp"

namespace Nektar::Operators
{

// LinAdvDiffReaction base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class LinAdvDiffReactionOp
    : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>;

public:
    static std::shared_ptr<LinAdvDiffReactionOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<LinAdvDiffReactionOp, LinAdvDiffReactionBlockOp>(
                expansionList, execStr, implStr);
    }

    static inline const std::string name = "LinAdvDiffReaction";

    void SetLambda(TData lambda)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetLambda(lambda);
        }
        m_isSetLambda = true;
    }

    void SetAdvVel(const unsigned int nVel, const Array<OneD, NekDouble> &Vel)
    {
        v_SetAdvVel(nVel, Vel);
        m_isSetAdvVel = true;
    }

protected:
    bool m_isSetLambda = false;
    bool m_isSetAdvVel = false;
    Field<TData, FieldState::Phys> m_advVel;
    std::vector<std::shared_ptr<LinAdvDiffReactionBlockOp<TData>>> m_blockOp;

    LinAdvDiffReactionOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList)
    {
    }

    ~LinAdvDiffReactionOp() override = default;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        ASSERTL1(m_isSetLambda,
                 "m_lambda has not been set."
                 "Set the value with SetLambda() before calling Apply().");

        ASSERTL1(m_isSetAdvVel,
                 "m_advVel has not been set."
                 "Set the value with SetAdvVel() before calling Apply().");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, outblock);
        }
    }

    virtual void v_SetAdvVel(const unsigned int nVel,
                             const Array<OneD, NekDouble> &Vel) = 0;
};

} // namespace Nektar::Operators
