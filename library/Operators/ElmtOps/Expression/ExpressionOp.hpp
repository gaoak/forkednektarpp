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

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/Expression/ExpressionBlockOp.hpp"

namespace Nektar::Operators
{

// Expression base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class ExpressionOp : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    static std::shared_ptr<ExpressionOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string &exprStr = "", const std::string &execStr = "",
        const std::string &implStr = "")
    {
        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? session->GetCmdLineArgument<std::string>("opExecSpace")
                : execStr;
        std::string implStr0 =
            (implStr == "") ? session->GetCmdLineArgument<std::string>("opImpl")
                            : implStr;

        auto op = Operator<TData>::template Create<ExpressionOp<TData>>(
            expansionList, execStr0);

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);

        // Read expression for each component
        auto nvariables = session->GetVariables().size();
        std::vector<LibUtilities::EquationSharedPtr> expressions;
        for (unsigned int nvar = 0; nvar < nvariables; ++nvar)
        {
            expressions.push_back(
                expansionList->GetSession()->GetFunction(exprStr, nvar));
        }

        // Loop over the blocks.
        for (auto &block : blocks)
        {
            op->m_blockOp.push_back(ExpressionBlockOp<TData>::Create(
                expansionList->GetExp(block.GetExpIdx()),
                expansionList->GetDataWarehouseSharedPtr(), execStr0,
                implStr0));

            op->m_blockOp.back()->SetExpressions(expressions);
        }

        return op;
    }

    static inline const std::string name = "Expression";

protected:
    std::vector<std::shared_ptr<ExpressionBlockOp<TData>>> m_blockOp;

    ExpressionOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList)
    {
    }

    ~ExpressionOp() override = default;

    void v_Apply(Field<TData, FieldState::Phys> &in,
                 Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, outblock);
        }
    }
};

} // namespace Nektar::Operators
