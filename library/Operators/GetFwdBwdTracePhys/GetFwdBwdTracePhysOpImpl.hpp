///////////////////////////////////////////////////////////////////////////////
//
// File: GetFwdBwdTracePhysOpImpl.hpp
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

#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysOp.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class GetFwdBwdTracePhysOpImpl : public GetFwdBwdTracePhysOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    GetFwdBwdTracePhysOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : GetFwdBwdTracePhysOp<TData>(expansionList, components)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<GetFwdBwdTracePhysOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

    void v_Apply(Field<TData, FieldState::Phys> &phys,
                 Field<TData, FieldState::Phys> &fwd,
                 Field<TData, FieldState::Phys> &bwd) override
    {
        ASSERTL1(fwd.GetNumComponents() == bwd.GetNumComponents(),
                 "Number of input and output components differ");

        fwd.template Initialize<MemSpace>(0.0);
        bwd.template Initialize<MemSpace>(0.0);

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &phyBlock = phys.GetBlocks()[blk];
            this->m_blockOp[blk]->Apply(phyBlock, fwd, bwd);
        }
    }
};

} // namespace Nektar::Operators::detail
