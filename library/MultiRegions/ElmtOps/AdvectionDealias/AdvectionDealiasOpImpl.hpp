///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasOpImpl.hpp
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

#include <vector>

#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <MultiRegions/ExpListHomogeneous1D.h>

#include <MultiRegions/ElmtOps/AdvectionDealias/AdvectionDealiasOp.hpp>
#include <MultiRegions/ElmtOps/DerivZOpImpl.hpp>
#include <MultiRegions/ElmtOps/PhysGalerkinProject1DScaled/PhysGalerkinProject1DScaledOp.hpp>
#include <MultiRegions/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledOp.hpp>

namespace Nektar::MultiRegions::detail
{

template <typename ExecSpace, typename TData>
class AdvectionDealiasOpImpl : public AdvectionDealiasOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AdvectionDealiasOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                           const std::vector<std::string> &components)
        : AdvectionDealiasOp<TData>(expansionList, components)
    {
        // The 3/2 rule is applied in the plane only, and the advection along
        // z rides it exactly as the other two directions do: du/dz is taken
        // on the native grid, interpolated to the fine grid, multiplied by
        // the through-plane velocity there and projected back. The Galerkin
        // projection is linear, so projecting this term on its own and
        // adding it gives the same answer as summing it into the block
        // operators' fine-grid product first.
        //
        // The homogeneous direction itself is not padded; that is the
        // solver's separate DEALIASING switch, which this does not do.
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);
        // The planes this rank holds, not the homogeneous basis' point
        // count: with npz > 1 the direction is split over the column
        // communicator while the basis still reports the global total.
        const unsigned int nhomo =
            homoExpList
                ? static_cast<unsigned int>(homoExpList->GetZIDs().size())
                : 1u;

        if (nhomo <= 1)
        {
            return;
        }

        m_zOp = std::make_shared<
            DerivZOpImpl<ExecSpace, TData, DerivZLayout::Identity,
                         DerivZOrder::First, false>>(expansionList);

        m_interpOp =
            PhysInterp1DScaledOp<TData>::Create(expansionList, components);
        m_interpOp->SetScaleFactor(m_dealiasScale);

        m_projectOp = PhysGalerkinProject1DScaledOp<TData>::Create(
            expansionList, components);
        m_projectOp->SetScaleFactor(m_dealiasScale);
        // The block operators have already written the plane part, so this
        // term goes on top of it.
        m_projectOp->SetAppend(true);

        const unsigned int nComp = static_cast<unsigned int>(components.size());
        const auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList);
        const auto fineBlockAttr =
            MultiRegions::GetScaledBlockAttributes<TData, FieldState::Phys>(
                expansionList, m_dealiasScale);

        m_dudz = LibUtilities::Field<TData, FieldState::Phys>(
            "AdvectionDealiasDuDz", blockAttr, nComp, nhomo);
        m_dudzFine = LibUtilities::Field<TData, FieldState::Phys>(
            "AdvectionDealiasDuDzFine", fineBlockAttr, nComp, nhomo);
        // Three velocity components on a 3DH1 expansion, of which the z pass
        // reads the last.
        m_advVelFine = LibUtilities::Field<TData, FieldState::Phys>(
            "AdvectionDealiasAdvVelFine", fineBlockAttr, 3u, nhomo);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AdvectionDealiasOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    /// The 3/2 rule, matching the fine point counts the block operators
    /// fetch their interpolation and projection matrices for.
    static constexpr TData m_dealiasScale = TData(1.5);

    /// Component for component: the z-derivative of each advected variable
    /// on the native grid, which then rides the in-plane interpolation.
    std::shared_ptr<DerivZOpImpl<ExecSpace, TData, DerivZLayout::Identity,
                                 DerivZOrder::First, false>>
        m_zOp;
    std::shared_ptr<PhysInterp1DScaledOp<TData>> m_interpOp;
    std::shared_ptr<PhysGalerkinProject1DScaledOp<TData>> m_projectOp;

    LibUtilities::Field<TData, FieldState::Phys> m_dudz;
    LibUtilities::Field<TData, FieldState::Phys> m_dudzFine;
    /// The advection velocity on the fine grid. It does not change between
    /// applications, so it is interpolated once in SetAdvVel().
    LibUtilities::Field<TData, FieldState::Phys> m_advVelFine;

    TData m_scale = 1.0;

    void v_SetScaleFFT(const TData &scale) override
    {
        m_scale = scale;
    }

    void v_SetAdvVelFFT(
        LibUtilities::Field<TData, FieldState::Phys> &advVel) override
    {
        if (!m_zOp)
        {
            return;
        }

        // Apply() takes whole Fields, so the whole velocity is interpolated
        // and the z pass reads the component it wants out of the result.
        m_interpOp->Apply(advVel, m_advVelFine);
    }

    void v_ApplyFFT(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        if (in.GetNumHomoModes() <= 1)
        {
            return;
        }

        // m_zOp is built only for a multi-plane 3DH1 expansion, so a
        // homogeneous field without one is 3DH2, which is not wired at all.
        // The guard is L0: without it the z advection would simply be
        // missing and the result would come back silently wrong in a
        // release build.
        ASSERTL0(m_zOp, "AdvectionDealiasOp does not support 3DH2 "
                        "configurations.");

        // du/dz on the native grid, then up to the fine grid to meet the
        // velocity there.
        m_zOp->Launch(in, m_dudz);
        m_interpOp->Apply(m_dudz, m_dudzFine);

        // scale * w du/dz, the one through-plane velocity component against
        // every advected variable.
        MultiplyByAdvVelZ();

        // Project back, on top of the plane part.
        m_projectOp->Apply(m_dudzFine, out);
    }

    /// Multiply every component of m_dudzFine by scale times the advection
    /// velocity along the homogeneous direction, which is the last component
    /// of m_advVelFine and is shared by them all.
    void MultiplyByAdvVelZ()
    {
        for (unsigned int blk = 0; blk < m_dudzFine.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &dudzblock = m_dudzFine.GetBlocks()[blk];
            auto &velblock  = m_advVelFine.GetBlocks()[blk];

            // The fine velocity takes the interleave the advection velocity
            // had when it was set, which the xy backend leaves in its own
            // format, so it is realigned with du/dz before being read.
            const auto interleaveWidth = dudzblock.GetInterleaveWidth();
            if (velblock.GetInterleaveWidth() != interleaveWidth)
            {
                auto velPtr =
                    velblock.template GetPtr<MemSpace, ReadWrite>(streamID);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, velblock.GetInterleaveWidth(),
                    velblock.GetNumElementsWithPadding() *
                        velblock.GetNumComponents() *
                        velblock.GetNumHomoModes(),
                    velblock.GetNumData(), velPtr, streamID);
                velblock.template SetInterleaveWidth<TData>(interleaveWidth);
            }

            const size_t nsize =
                dudzblock.CompSize() * dudzblock.GetNumHomoModes();
            const TData *velPtr =
                velblock.template GetPtr<MemSpace, ReadOnly>(streamID) +
                2u * velblock.CompSize() * velblock.GetNumHomoModes();
            TData *dudzPtr =
                dudzblock.template GetPtr<MemSpace, ReadWrite>(streamID);

            for (unsigned int c = 0; c < m_dudzFine.GetNumComponents(); ++c)
            {
                TData *dst = dudzPtr + c * nsize;
                Math::mulKernel<ExecSpace>(nsize, velPtr, dst, dst, streamID);
                Math::mulKernel<ExecSpace>(nsize, m_scale, dst, dst, streamID);
            }
        }
    }
};

} // namespace Nektar::MultiRegions::detail
