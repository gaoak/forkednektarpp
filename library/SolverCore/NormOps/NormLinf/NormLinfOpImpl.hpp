///////////////////////////////////////////////////////////////////////////////
//
// File: NormLinfOpImpl.hpp
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

#include <SolverCore/NormOps/NormLinf/NormLinfOp.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class NormLinfOpImpl : public NormLinfOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NormLinfOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : NormLinfOp<TData>(expansionList, components)
    {
        if (expansionList->GetExpType() == MultiRegions::e3DH1D)
        {
            m_numPlanes = expansionList->GetZIDs().size();
        }

        // Layout: [ norm_c0, ..., norm_c{n-1} ]; the block ops take the
        // maximum over the local planes.
        this->m_norm = LibUtilities::MemoryRegion<TData>(
            "NormLinf", components.size(), eHostPinned);
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory.
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<NormLinfOpImpl<ExecSpace, TData>>(expansionList,
                                                                  components);
    }

protected:
    /// Number of local planes, or 1 without a homogeneous direction.
    unsigned int m_numPlanes = 1;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) override
    {
        ASSERTL1(in.GetNumComponents() == this->m_components.size(),
                 "NormLinfOp field component count mismatch.");
        ASSERTL1(in.GetNumHomoModes() == this->m_numPlanes,
                 "NormLinfOp field plane count mismatch.");
        ASSERTL1(in.GetBlocks().size() == this->m_blockOp.size(),
                 "NormLinfOp field block count changed after configuration.");

        // Reset norms.
        this->m_norm.template Initialize<MemSpace>(TData{0});

        // Loop all blocks.
        for (unsigned int block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto &inblock = in.GetBlocks()[block_idx];
            this->m_blockOp[block_idx]->Apply(inblock, this->m_norm);
        }

        // Take the maximum over the spatial ranks, which exclude those of other
        // time slices, to every one or to rank zero alone.
        auto comm = this->m_expansionList->GetComm()->GetSpaceComm();
        if (this->m_rootOnly)
        {
            comm->template Reduce<MemSpace>(this->m_norm,
                                            LibUtilities::ReduceMax, 0);
        }
        else
        {
            comm->template AllReduce<MemSpace>(this->m_norm,
                                               LibUtilities::ReduceMax);
        }
    }
};

} // namespace Nektar::SolverCore::detail
