///////////////////////////////////////////////////////////////////////////////
//
// File: MaxStdVelocityOpImpl.hpp
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
// Description: MaxStdVelocity operator implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/Communication/Comm.h"
#include "SolverCore/CFL/MaxStdVelocity/MaxStdVelocityOp.hpp"

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class MaxStdVelocityOpImpl : public MaxStdVelocityOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    MaxStdVelocityOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : MaxStdVelocityOp<TData>(expansionList, components)
    {
        // Everything Apply() needs is taken once here, so the apply path
        // does not consult the expansion list.
        m_rowComm = expansionList->GetComm()->GetRowComm();
    }

    static std::string className;

    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<MaxStdVelocityOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    LibUtilities::CommSharedPtr m_rowComm;

    // The input carries one velocity component per coordinate direction and,
    // when the sound speed factor is set, a wave speed after them; anything
    // beyond that is ignored, so the same field can be passed with the
    // weight on and off. The contract is fixed where the solver builds the
    // field and is not re-checked here.
    TData v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) override
    {
        ASSERTL1(in.GetNumHomoModes() == 1,
                 "MaxStdVelocity is not implemented for homogeneous "
                 "expansions.");

        const auto nBlocks = in.GetBlocks().size();

        this->m_data.template Initialize<MemSpace>(TData{0});

        for (unsigned int blk = 0; blk < nBlocks; ++blk)
        {
            auto &inblock = in.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, this->m_data);
        }

        // Weight each block by its own order before combining, which is what
        // lets a variable order mesh reduce to a single number.
        auto dataptr =
            this->m_data.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        TData invTimeScale = TData(0);
        for (unsigned int blk = 0; blk < nBlocks; ++blk)
        {
            invTimeScale =
                std::max(invTimeScale, dataptr[blk] * this->m_orderWeight[blk]);
        }

        // The timestep is set from the whole mesh, so every rank needs the
        // same answer.
        m_rowComm->AllReduce(invTimeScale, Nektar::LibUtilities::ReduceMax);

        return invTimeScale;
    }
};

} // namespace Nektar::SolverCore::detail
