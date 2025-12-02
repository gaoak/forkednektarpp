///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionBlockOp.hpp
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

#include "Operators/ElmtOps/BlockOperator.hpp"

namespace Nektar::Operators
{

template <typename TData>
class LinAdvDiffReactionBlockOp : public BlockOperator<TData>
{
public:
    static std::shared_ptr<LinAdvDiffReactionBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse, std::string execStr,
        std::string implStr)
    {
        return BlockOperator<TData>::template Create<LinAdvDiffReactionBlockOp>(
            block_idx, exp, dataWarehouse, execStr, implStr);
    }

    static inline const std::string name = "BlockLinAdvDiffReaction";

    void SetLambda(TData lambda)
    {
        this->v_SetLambda(lambda);
    }

    void SetAdvVel(const unsigned int nVel, BlockAccessor<TData> &Vel)
    {
        v_SetAdvVel(nVel, Vel);
    }

protected:
    TData m_lambda;

    LinAdvDiffReactionBlockOp(const unsigned int block_idx,
                              const LocalRegions::ExpansionSharedPtr &exp,
                              NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(block_idx, exp, dataWarehouse)
    {
    }

    ~LinAdvDiffReactionBlockOp() override = default;

    virtual void v_SetLambda(const TData &lambda) = 0;

    virtual void v_SetAdvVel(const unsigned int nVel,
                             BlockAccessor<TData> &Vel) = 0;
};

} // namespace Nektar::Operators
