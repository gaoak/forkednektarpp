///////////////////////////////////////////////////////////////////////////////
//
// File: FwdTransBCBlockOp.hpp
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

#include "Operators/Common/BlockOperator.hpp"
#include "Operators/ElmtOps/ElmtBlockOp.hpp"

namespace Nektar::Operators
{

template <typename TData> class FwdTransBCBlockOp : public BlockOperator<TData>
{
public:
    ~FwdTransBCBlockOp() override = default;

    static std::shared_ptr<FwdTransBCBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse, const std::string &execStr)
    {
        return BlockOperator<TData>::template Create<FwdTransBCBlockOp>(
            block_idx, exp, dataWarehouse, execStr);
    }

    static inline const std::string name = "BlockFwdTransBC";

    void Apply(MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
               MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

    void operator()(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

    void SetInvMassMatrix(std::vector<TData> &dmat)
    {
        v_SetInvMassMatrix(dmat);
    }

protected:
    FwdTransBCBlockOp(const unsigned int block_idx,
                      const LocalRegions::ExpansionSharedPtr &exp,
                      NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(block_idx, exp, dataWarehouse)
    {
    }

    virtual void v_Apply(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock) = 0;

    virtual void v_SetInvMassMatrix(std::vector<TData> &dmat) = 0;
};

} // namespace Nektar::Operators
