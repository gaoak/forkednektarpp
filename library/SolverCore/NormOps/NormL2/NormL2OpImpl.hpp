///////////////////////////////////////////////////////////////////////////////
//
// File: NormL2OpImpl.hpp
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

#include <SolverCore/NormOps/NormL2/NormL2Op.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class NormL2OpImpl : public NormL2Op<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NormL2OpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : NormL2Op<TData>(expansionList, components)
    {
        if (expansionList->GetExpType() == MultiRegions::e3DH1D)
        {
            m_numPlanes = expansionList->GetZIDs().size();
        }

        // Layout: [ norm_c0, ..., norm_c{n-1} ], the squared norms until
        // GetNorms() takes their square roots.
        this->m_norm = LibUtilities::MemoryRegion<TData>(
            "NormL2", components.size(), eHostPinned);
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory.
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<NormL2OpImpl<ExecSpace, TData>>(expansionList,
                                                                components);
    }

protected:
    /// Number of local planes, or 1 without a homogeneous direction.
    unsigned int m_numPlanes = 1;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) override
    {
        ASSERTL1(in.GetNumComponents() == this->m_components.size(),
                 "NormL2 field component count mismatch.");
        ASSERTL1(in.GetNumHomoModes() == this->m_numPlanes,
                 "NormL2 field plane count mismatch.");
        ASSERTL1(in.GetBlocks().size() == this->m_blockOp.size(),
                 "NormL2 field block count changed after configuration.");

        // Reset norms.
        this->m_norm.template Initialize<MemSpace>(TData{0});

        // Loop all blocks. With a homogeneous direction they sum the local
        // planes of each component with their z quadrature weights.
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            auto &inblock = in.GetBlocks()[blk];
            this->m_blockOp[blk]->Apply(inblock, this->m_norm);
        }

        // Sum over the ranks sharing the planes, then over the ranks holding
        // the other planes.
        auto comm = this->m_expansionList->GetComm();
        comm->GetRowComm()->template AllReduce<MemSpace>(
            this->m_norm, LibUtilities::ReduceSum);
        if (this->m_homogeneous)
        {
            comm->GetColumnComm()->template AllReduce<MemSpace>(
                this->m_norm, LibUtilities::ReduceSum);
        }
    }

    /// The volume of the mesh, summed over the blocks and across the ranks,
    /// and extended over the homogeneous length.
    void v_ComputeVolume() override
    {
        if (!this->m_volumeComputed)
        {
            LibUtilities::MemoryRegion<TData> volume("NormL2 volume", 1,
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
                     "NormL2Op encountered a non-positive volume.");
        }
    }
};

} // namespace Nektar::SolverCore::detail
