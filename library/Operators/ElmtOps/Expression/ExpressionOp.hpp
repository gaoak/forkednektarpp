///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionOp.hpp
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

#include <boost/algorithm/string.hpp>

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/Expression/ExpressionBlockOp.hpp"

namespace Nektar::Operators
{

// Expression base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class ExpressionOp : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    static std::shared_ptr<ExpressionOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto op =
            ElmtOp<FieldState::Phys, FieldState::Phys, TData>::template Create<
                ExpressionOp, ExpressionBlockOp>(expansionList, components,
                                                 execStr, implStr);

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < op->m_blockOp.size(); ++blk)
        {
            // Default time set to 0.0
            op->m_blockOp[blk]->SetTime(0.0);

            // Default scale set to 1.0
            op->m_blockOp[blk]->SetScale(1.0);
        }
        return op;
    }

    static inline const std::string name = "Expression";

    void SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs)
    {
        // Check number of EVARS for new expressions
        std::vector<unsigned int> numEvars;
        for (unsigned int ne = 0; ne < exprs.size(); ++ne)
        {
            numEvars.push_back(this->getNumberEvars(exprs[ne]));
        }

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetExpressions(exprs);
            this->m_blockOp[blk]->SetNumEvars(numEvars);
        }

        m_isExpressionsDefined = true;
    }

    void SetTime(const TData &time)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetTime(time);
        }
    }

    void SetScale(const TData &scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetScale(scale);
        }
    }

    unsigned int getNumberEvars(
        const LibUtilities::EquationSharedPtr &expression)
    {
        // Extract the number of expression variables (EVARS).
        // Note that we use four by default: x, y, z, t
        std::vector<std::string> vars;
        auto variableList = expression->GetVlist();
        boost::split(vars, variableList, boost::is_any_of(", "));
        return vars.size();
    }

protected:
    bool m_isExpressionsDefined = false;
    std::vector<std::shared_ptr<ExpressionBlockOp<TData>>> m_blockOp;

    ExpressionOp(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~ExpressionOp() override = default;

    void v_Apply(Field<TData, FieldState::Phys> &in,
                 Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        ASSERTL1(this->m_isExpressionsDefined,
                 "No expressions defined for this Operator. Define with "
                 "SetExpressions().")

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
