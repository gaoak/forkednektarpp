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
        // Layout: [ integral_c0, ..., integral_c{n-1} ]; GetMeans() divides
        // the integrals by the volume.
        this->m_result = LibUtilities::MemoryRegion<TData>(
            "Mean result", components.size(), eHostPinned);
    }

    static std::string className;

    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<MeanOpImpl<ExecSpace, TData>>(expansionList,
                                                              components);
    }

protected:
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &field) override
    {
        ASSERTL1(field.GetNumHomoModes() == 1,
                 "Mean does not support homogeneous expansions.");
        ASSERTL1(field.GetNumComponents() == this->m_components.size(),
                 "Mean field component count mismatch.");
        ASSERTL1(field.GetBlocks().size() == this->m_blockOp.size(),
                 "Mean field block count changed after configuration.");

        ComputeVolume();

        // The block operators accumulate on their own streams, between the
        // zeroing of the result and its reduction on the null stream.
        this->m_result.template Initialize<MemSpace>(TData{0});

        for (unsigned int block = 0; block < field.GetBlocks().size(); ++block)
        {
            this->m_blockOp[block]->Apply(field.GetBlocks()[block],
                                          this->m_result);
        }

        auto comm = this->m_expansionList->GetComm();
        comm->template Reduce<MemSpace>(this->m_result, LibUtilities::ReduceSum,
                                        0);
    }

    /// The volume of the mesh, summed over the blocks and across the ranks.
    void ComputeVolume()
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
            comm->template AllReduce<MemSpace>(volume, LibUtilities::ReduceSum);

            this->m_volume =
                volume.template GetPtr<NektarSpaces::HostSpace, ReadOnly>()[0];

            ASSERTL0(this->m_volume > TData(0),
                     "Mean encountered a non-positive domain volume.");
            this->m_volumeComputed = true;
        }
    }
};

} // namespace Nektar::SolverCore::detail
