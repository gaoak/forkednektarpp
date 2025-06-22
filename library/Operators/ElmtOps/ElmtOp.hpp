///////////////////////////////////////////////////////////////////////////////
//
// File: ElmtOp.hpp
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

#include "Operators/Common/Operator.hpp"
#include "Operators/ElmtOps/BlockOperator.hpp"

namespace Nektar::Operators
{

template <FieldState TFieldIn, FieldState TFieldOut, typename TData>
class ElmtOp : public Operator<TData>
{
public:
    template <template <typename> typename TOperator,
              template <typename> typename TBlockOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? session->GetCmdLineArgument<std::string>("opExecSpace")
                : execStr;
        std::string implStr0 =
            (implStr == "") ? session->GetCmdLineArgument<std::string>("opImpl")
                            : implStr;

        auto op = Operator<TData>::template Create<TOperator>(expansionList,
                                                              execStr0);

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);

        // Loop over the blocks.
        for (auto &block : blocks)
        {
            op->m_blockOp.push_back(TBlockOperator<TData>::Create(
                expansionList->GetExp(block.GetExpIdx()),
                expansionList->GetDataWarehouseSharedPtr(), execStr0,
                implStr0));
        }

        return op;
    }

    void Apply(Field<TData, TFieldIn> &in, Field<TData, TFieldOut> &out)
    {
        v_Apply(in, out);
    }

    void operator()(Field<TData, TFieldIn> &in, Field<TData, TFieldOut> &out)
    {
        this->v_Apply(in, out);
    }

protected:
    ~ElmtOp() override = default;

    ElmtOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void v_Apply(Field<TData, TFieldIn> &in,
                         Field<TData, TFieldOut> &out) = 0;
};

} // namespace Nektar::Operators
