///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseOp.hpp
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

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp"

namespace Nektar::Operators
{

// IProductWRTDerivBase base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class IProductWRTDerivBaseOp
    : public ElmtOp<FieldState::Phys, FieldState::Coeff, TData>
{
public:
    static std::shared_ptr<IProductWRTDerivBaseOp<TData>> Create(
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

        auto op =
            Operator<TData>::template Create<IProductWRTDerivBaseOp<TData>>(
                expansionList, execStr0);

        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);

        // Loop over the blocks.
        for (auto &block : blocks)
        {
            op->m_blockOp.push_back(IProductWRTDerivBaseBlockOp<TData>::Create(
                expansionList->GetExp(block.GetExpIdx()),
                expansionList->GetDataWarehouseSharedPtr(), execStr0,
                implStr0));
        }

        return op;
    }

    static inline const std::string name = "IProductWRTDerivBase";

    void SetAppend(bool append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

protected:
    std::vector<std::shared_ptr<IProductWRTDerivBaseBlockOp<TData>>> m_blockOp;

    IProductWRTDerivBaseOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : ElmtOp<FieldState::Phys, FieldState::Coeff, TData>(expansionList)
    {
    }

    ~IProductWRTDerivBaseOp() override = default;

    void v_Apply(Field<TData, FieldState::Phys> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(out.GetNumComponents() ==
                     in.GetNumComponents() /
                         this->m_expansionList->GetCoordim(0),
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
