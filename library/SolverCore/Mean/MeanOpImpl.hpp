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

        // Layout: [ c0_p0, ..., c0_p{m-1}, c1_p0, ... ] over the m local
        // planes; v_Apply() sums the planes into m_result.
        if (!this->m_planeWeights.empty())
        {
            this->m_plane = LibUtilities::MemoryRegion<TData>(
                "Mean plane", components.size() * this->m_planeWeights.size(),
                eHostPinned);
        }
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
                 "Mean field component count mismatch.");
        ASSERTL1(in.GetNumHomoModes() == m_numPlanes,
                 "Mean field plane count mismatch.");
        ASSERTL1(in.GetBlocks().size() == this->m_blockOp.size(),
                 "Mean field block count changed after configuration.");

        const auto numComp = in.GetNumComponents();
        const auto numHomo = in.GetNumHomoModes();

        v_ComputeVolume();

        // The block ops integrate each plane separately; without a
        // homogeneous direction they write straight into the result.
        auto &plane = (numHomo == 1) ? this->m_result : this->m_plane;

        // The block operators accumulate on their own streams, between the
        // zeroing of the result and its reduction on the null stream.
        plane.template Initialize<MemSpace>(TData{0});

        // Loop all blocks.
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            auto &inblock = in.GetBlocks()[blk];
            this->m_blockOp[blk]->Apply(inblock, plane);
        }

        // Communicate norms.
        auto comm = this->m_expansionList->GetComm();
        comm->GetRowComm()->template AllReduce<MemSpace>(
            plane, LibUtilities::ReduceSum);

        if (numHomo == 1)
        {
            return;
        }

        // Sum the planes with their z quadrature weights, then over the
        // ranks holding the other planes.
        auto planeptr =
            plane.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto resultptr =
            this->m_result
                .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int nc = 0; nc < numComp; ++nc)
        {
            resultptr[nc] = 0.0;
            for (unsigned int p = 0; p < numHomo; ++p)
            {
                resultptr[nc] +=
                    this->m_planeWeights[p] * planeptr[nc * numHomo + p];
            }
        }

        comm->GetColumnComm()->template AllReduce<NektarSpaces::HostSpace>(
            this->m_result, LibUtilities::ReduceSum);
    }

    /// The volume of the mesh, summed over the blocks and across the ranks,
    /// and extended over the homogeneous length.
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
                     "Mean encountered a non-positive domain volume.");
        }
    }
};

} // namespace Nektar::SolverCore::detail
