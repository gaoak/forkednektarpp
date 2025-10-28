///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionOpImpl.hpp
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

#include "Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionOp.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class LinAdvDiffReactionOpImpl : public LinAdvDiffReactionOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    LinAdvDiffReactionOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : LinAdvDiffReactionOp<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<LinAdvDiffReactionOpImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    void v_SetAdvVel(const unsigned int nVel,
                     const Array<OneD, NekDouble> &Vel) override
    {
        // Set up a physBlockAttributes which will be
        std::vector<BlockAttributes> physBlockAttributes =
            GetBlockAttributes<TData>(FieldState::Phys, this->m_expansionList,
                                      1);

        this->m_advVel = Field<TData, FieldState::Phys>(
            "Advection Field", physBlockAttributes, nVel, 1,
            ExecSpace::alignment);

        this->m_advVel.template CopyArray<NektarSpaces::HostSpace>(Vel);

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAdvVel(nVel,
                                            this->m_advVel.GetBlocks()[blk]);
        }
    }
};

} // namespace Nektar::Operators::detail
