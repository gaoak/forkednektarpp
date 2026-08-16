///////////////////////////////////////////////////////////////////////////////
//
// File: GetFwdBwdTracePhysBlockOp.hpp
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
#include <MultiRegions/DataWarehouse/TraceDataWarehouse.hpp>

namespace Nektar::Operators
{

template <typename TData>
class GetFwdBwdTracePhysBlockOp : public BlockOperator<TData>
{
public:
    ~GetFwdBwdTracePhysBlockOp() override = default;

    static std::shared_ptr<GetFwdBwdTracePhysBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr)
    {
        return BlockOperator<TData>::template Create<GetFwdBwdTracePhysBlockOp>(
            block_idx, exp, dataWarehouse, execStr);
    }

    static inline const std::string name = "BlockGetFwdBwdTracePhys";

    void Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &phyBlock,
               LibUtilities::Field<TData, FieldState::Phys> &fwd,
               LibUtilities::Field<TData, FieldState::Phys> &bwd)
    {
        this->v_Apply(phyBlock, fwd, bwd);
    }

    void operator()(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &phyBlock,
        LibUtilities::Field<TData, FieldState::Phys> &fwd,
        LibUtilities::Field<TData, FieldState::Phys> &bwd)
    {
        this->v_Apply(phyBlock, fwd, bwd);
    }

    void SetTracePhysOffset(std::vector<size_t> offset)
    {
        v_SetTracePhysOffset(offset);
    }

    void SetFwdOnly(bool fwdOnly)
    {
        this->m_fwdOnly = fwdOnly;
    }

protected:
    bool m_fwdOnly = false;

    GetFwdBwdTracePhysBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(block_idx, exp, dataWarehouse)
    {
    }

    virtual void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &phyBlock,
        LibUtilities::Field<TData, FieldState::Phys> &fwd,
        LibUtilities::Field<TData, FieldState::Phys> &bwd) = 0;

    virtual void v_SetTracePhysOffset(std::vector<size_t> offset) = 0;
};

} // namespace Nektar::Operators
