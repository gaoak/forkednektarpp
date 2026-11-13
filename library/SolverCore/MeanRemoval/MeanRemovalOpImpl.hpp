///////////////////////////////////////////////////////////////////////////////
//
// File: MeanRemovalOpImpl.hpp
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

#include <SolverCore/MeanRemoval/MeanRemovalOp.hpp>

#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <SolverCore/GlobalLinSysOps/LinearSystems/LinearSystemOp.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class MeanRemovalOpImpl : public MeanRemovalOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    MeanRemovalOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components)
        : MeanRemovalOp<TData>(expansionList, components)
    {
        if (expansionList->GetExpType() == MultiRegions::e3DH1D)
        {
            m_numPlanes = expansionList->GetZIDs().size();
        }

        // Layout: [ integral_c0, ..., integral_c{n-1} ].
        this->m_domainIntegral = LibUtilities::MemoryRegion<TData>(
            "MeanRemoval", components.size(), eHostPinned);

        // Layout: [ surface_c0, ..., surface_c{n-1} ]: the surface integrals
        // of the Neumann values of the system set by SetNeumannBalance(),
        // zero without one.
        this->m_surfaceIntegral = LibUtilities::MemoryRegion<TData>(
            "MeanRemoval surface", components.size(), eHostPinned);

        // Layout: [ c0_p0, ..., c0_p{m-1}, c1_p0, ... ] over the m local
        // planes; the planes are summed into m_domainIntegral and
        // m_surfaceIntegral.
        if (!this->m_planeWeights.empty())
        {
            this->m_domainIntegralPlane = LibUtilities::MemoryRegion<TData>(
                "MeanRemoval plane",
                components.size() * this->m_planeWeights.size(), eHostPinned);
            this->m_surfaceIntegralPlane = LibUtilities::MemoryRegion<TData>(
                "MeanRemoval surface plane",
                components.size() * this->m_planeWeights.size(), eHostPinned);
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory.
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<MeanRemovalOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    /// Number of local planes, or 1 without a homogeneous direction.
    unsigned int m_numPlanes = 1;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &inout) override
    {
        ASSERTL1(inout.GetNumComponents() == this->m_components.size(),
                 "MeanRemoval field component count mismatch.");
        ASSERTL1(inout.GetNumHomoModes() == m_numPlanes,
                 "MeanRemoval field plane count mismatch.");
        ASSERTL1(inout.GetBlocks().size() == this->m_blockOp.size(),
                 "MeanRemoval field block count changed after configuration.");

        const auto numComp = inout.GetNumComponents();
        const auto numHomo = inout.GetNumHomoModes();

        v_ComputeVolume();
        v_ComputeNeumannSurfaceIntegral();

        // The block ops integrate each plane separately; without a
        // homogeneous direction they write straight into m_domainIntegral.
        auto &plane = (numHomo == 1) ? this->m_domainIntegral
                                     : this->m_domainIntegralPlane;

        plane.template Initialize<MemSpace>(TData{0});

        // Loop all blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            auto &inblock = inout.GetBlocks()[blk];
            this->m_blockOp[blk]->Apply(inblock, plane);
        }

        auto comm = this->m_expansionList->GetComm();
        comm->GetRowComm()->template AllReduce<MemSpace>(
            plane, LibUtilities::ReduceSum);

        if (numHomo > 1)
        {
            // Sum the planes with their z quadrature weights, then over the
            // ranks holding the other planes.
            const unsigned int numPlanes = this->m_planeWeights.size();
            auto planeptr =
                plane.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto resultptr =
                this->m_domainIntegral
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int nc = 0; nc < this->m_components.size(); ++nc)
            {
                resultptr[nc] = 0.0;
                for (unsigned int p = 0; p < numPlanes; ++p)
                {
                    resultptr[nc] +=
                        this->m_planeWeights[p] * planeptr[nc * numPlanes + p];
                }
            }

            comm->GetColumnComm()->template AllReduce<NektarSpaces::HostSpace>(
                this->m_domainIntegral, LibUtilities::ReduceSum);
        }

        auto integral = this->m_domainIntegral.ToVector();
        auto surface  = this->m_surfaceIntegral.ToVector();

        // The same constant is removed from every plane of a component.
        std::vector<TData> negMean(numComp * numHomo);
        for (unsigned int nc = 0; nc < numComp; ++nc)
        {
            for (unsigned int p = 0; p < numHomo; ++p)
            {
                negMean[nc * numHomo + p] =
                    -(integral[nc] - surface[nc]) / this->m_volume;
            }
        }

        Math::addScalar<ExecSpace>(negMean, inout, inout);
    }

    /// The surface integral of the Neumann values of each component,
    /// summed across the ranks and, with a homogeneous direction, over the
    /// planes, into m_surfaceIntegral.
    void v_ComputeNeumannSurfaceIntegral() override
    {
        const bool homo = !this->m_planeWeights.empty();
        auto &plane =
            homo ? this->m_surfaceIntegralPlane : this->m_surfaceIntegral;

        plane.template Initialize<MemSpace>(TData{0});

        if (this->m_neumannSystem)
        {
            this->m_neumannSystem->AddNeumannSurfaceIntegral(plane, 0);
        }

        auto comm = this->m_expansionList->GetComm();
        comm->GetRowComm()->template AllReduce<MemSpace>(
            plane, LibUtilities::ReduceSum);

        if (!homo)
        {
            return;
        }

        // Sum the planes with their z quadrature weights, then over the
        // ranks holding the other planes.
        const unsigned int numPlanes = this->m_planeWeights.size();
        auto planeptr =
            plane.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto resultptr =
            this->m_surfaceIntegral
                .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int nc = 0; nc < this->m_components.size(); ++nc)
        {
            resultptr[nc] = 0.0;
            for (unsigned int p = 0; p < numPlanes; ++p)
            {
                resultptr[nc] +=
                    this->m_planeWeights[p] * planeptr[nc * numPlanes + p];
            }
        }

        comm->GetColumnComm()->template AllReduce<NektarSpaces::HostSpace>(
            this->m_surfaceIntegral, LibUtilities::ReduceSum);
    }

    /// The volume of the mesh, summed over the blocks and across the ranks,
    /// and extended over the homogeneous length.
    void v_ComputeVolume() override
    {
        if (!this->m_volumeComputed)
        {
            LibUtilities::MemoryRegion<TData> volume("MeanRemoval volume", 1,
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
                     "MeanRemovalOp encountered a non-positive volume.");
        }
    }
};

} // namespace Nektar::SolverCore::detail
