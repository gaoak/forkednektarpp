///////////////////////////////////////////////////////////////////////////////
//
// File: MeanOpImpl.hpp
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
// Description: Device-portable domain average of a physical-space field.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SolverCore/Mean/MeanOp.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class MeanOpImpl : public MeanOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    MeanOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
               const std::vector<std::string> &components)
        : MeanOp<TData>(expansionList, components)
    {
        if (expansionList->GetExpType() == MultiRegions::e3DH1D)
        {
            m_numPlanes = expansionList->GetZIDs().size();
        }

        // Layout: [ integral_c0, ..., integral_c{n-1} ]; GetMeans() divides
        // the integrals by the volume.
        this->m_result = LibUtilities::MemoryRegion<TData>(
            "Mean result", components.size(), eHostPinned);
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory.
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<MeanOpImpl<ExecSpace, TData>>(expansionList,
                                                              components);
    }

protected:
    /// Number of local planes, or 1 without a homogeneous direction.
    unsigned int m_numPlanes = 1;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) override
    {
        ASSERTL1(in.GetNumComponents() == this->m_components.size(),
                 "MeanOp field component count mismatch.");
        ASSERTL1(in.GetNumHomoModes() == this->m_numPlanes,
                 "MeanOp field plane count mismatch.");
        ASSERTL1(in.GetBlocks().size() == this->m_blockOp.size(),
                 "MeanOp field block count changed after configuration.");

        // The volume is computed here, where every rank takes part, so that
        // GetVolume() needs no collective afterwards.
        v_ComputeVolume();

        // The block operators accumulate on their own streams, between the
        // zeroing of the result and its reduction on the null stream. With a
        // homogeneous direction they sum the local planes of each component
        // with their z quadrature weights.
        this->m_result.template Initialize<MemSpace>(TData{0});

        // Loop all blocks.
        for (unsigned int block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto &inblock = in.GetBlocks()[block_idx];
            this->m_blockOp[block_idx]->Apply(inblock, this->m_result);
        }

        // Sum over the spatial ranks, which exclude those of other time
        // slices, to every one or to rank zero alone.
        auto comm = this->m_expansionList->GetComm()->GetSpaceComm();
        if (this->m_rootOnly)
        {
            comm->template Reduce<MemSpace>(this->m_result,
                                            LibUtilities::ReduceSum, 0);
        }
        else
        {
            comm->template AllReduce<MemSpace>(this->m_result,
                                               LibUtilities::ReduceSum);
        }
    }

    /// The volume of the mesh, summed over the blocks and across the ranks
    /// sharing the planes, which hold one copy of the mesh, and extended over
    /// the homogeneous length.
    void v_ComputeVolume() override
    {
        if (!this->m_volumeComputed)
        {
            LibUtilities::MemoryRegion<TData> volume("Mean volume", 1,
                                                     eHostPinned);
            volume.template Initialize<MemSpace>(TData{0});

            for (auto &blockOp : this->m_blockOp)
            {
                blockOp->Volume(volume);
            }

            auto comm = this->m_expansionList->GetComm();
            comm->GetRowComm()->template AllReduce<MemSpace>(
                volume, LibUtilities::ReduceSum);

            this->m_volumeComputed = true;
            this->m_volume =
                volume.template GetPtr<NektarSpaces::HostSpace, ReadOnly>()[0] *
                this->m_homoLen;
            ASSERTL1(this->m_volume > 0.0,
                     "MeanOp encountered a non-positive domain volume.");
        }
    }
};

} // namespace Nektar::SolverCore::detail
