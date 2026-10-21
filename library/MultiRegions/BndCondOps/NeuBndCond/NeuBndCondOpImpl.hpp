///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondOpImpl.hpp
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

#include <algorithm>
#include <string>
#include <vector>

#include <LibUtilities/BasicUtils/StringUtils.hpp>

#include <MultiRegions/ContField.h>

#include <MultiRegions/BndCondOps/Common/CGBndCondKernels.hpp>
#include <MultiRegions/BndCondOps/Common/CGBndCondMapBuilder.hpp>
#include <MultiRegions/BndCondOps/NeuBndCond/NeuBndCondKernels.hpp>
#include <MultiRegions/BndCondOps/NeuBndCond/NeuBndCondOp.hpp>

#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/Expression/ExpressionOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>

using namespace Nektar;

namespace Nektar::MultiRegions::detail
{

template <typename ExecSpace, typename TData>
class NeuBndCondOpImpl : public NeuBndCondOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NeuBndCondOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : NeuBndCondOp<TData>(expansionList, components)
    {
        BuildRegions();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<NeuBndCondOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    bool m_anySignChange         = false;
    size_t m_numBndCoeffCompSize = 0;
    std::vector<bool> m_signChange;

    /// Stream the gather runs on, the first block's. It orders it against
    /// the block streams with events.
    unsigned int m_opStreamID = 1;
    /// Streams of the region and domain blocks, which write the compact
    /// array and read the gathered boundary values.
    std::vector<unsigned int> m_updateStreams;

    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_wsp_phys;
    std::vector<LibUtilities::Field<TData, FieldState::Coeff>> m_wsp_coeffs;
    std::vector<std::shared_ptr<ExpressionOp<TData>>> m_expressionOps;
    std::vector<std::shared_ptr<IProductWRTBaseOp<TData>>> m_iprodOps;
    std::vector<size_t> m_neuCoeffOffsets;
    std::vector<size_t> m_neuNumCoeffs;
    std::vector<bool> m_neuIsPoint;

    std::vector<LibUtilities::MemoryRegion<size_t>> m_map;
    std::vector<LibUtilities::MemoryRegion<TData>> m_sign;
    std::vector<LibUtilities::MemoryRegion<TData>> m_bndCoeff;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_bndCoeffSrc;
    LibUtilities::MemoryRegion<TData> m_compactBndCoeff;
    std::vector<std::vector<size_t>> m_compOffsets;
    std::vector<std::vector<size_t>> m_compCounts;

    // Local coefficient targets written by more than one boundary trace
    // piece of the same element (see CGBndCondCoeffMaps::groupTarget) --
    // summed locally and written with one non-atomic add in v_Apply()
    // instead of racing an atomic_add per contribution.
    std::vector<LibUtilities::MemoryRegion<size_t>> m_groupTarget;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_groupOffset;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_groupSrc;
    std::vector<LibUtilities::MemoryRegion<TData>> m_groupSign;
    std::vector<LibUtilities::MemoryRegion<TData>> m_groupValue;
    std::vector<std::vector<size_t>> m_compGroupOffsets;
    std::vector<std::vector<size_t>> m_compGroupCounts;

    // Total number of matched boundary trace coefficients, for
    // v_GetNumBndDofs() -- see where this is set in BuildRegions().
    size_t m_totalBndDofs = 0;

    // Total number of grouped (duplicate-target) entries across every
    // block/component -- see where this is set in BuildRegions().
    size_t m_totalGroupCount = 0;

    // User-defined types of the Neumann conditions another operator
    // applies, set by ExcludeUserDefinedType().
    std::vector<std::string> m_excludedUserDefinedTypes;

    // Whether this operator applies @p bc: a Neumann condition whose
    // user-defined type is not excluded.
    bool IsApplied(const SpatialDomains::BoundaryConditionShPtr &bc) const
    {
        return std::dynamic_pointer_cast<
                   SpatialDomains::NeumannBoundaryCondition>(bc) &&
               std::none_of(m_excludedUserDefinedTypes.begin(),
                            m_excludedUserDefinedTypes.end(),
                            [&bc](const std::string &type) {
                                return StringUtils::IEquals(
                                    bc->GetUserDefined(), type);
                            });
    }

    // Build the fields, operators and maps of the regions with a Neumann
    // condition this operator applies (IsApplied()), and evaluate it.
    void BuildRegions()
    {
        auto session       = this->m_expansionList->GetSession();
        auto graph         = this->m_expansionList->GetGraph();
        unsigned int nComp = this->m_components.size();

        // Create boundary conditions
        const SpatialDomains::BoundaryConditions bcs(session, graph);
        const SpatialDomains::BoundaryRegionCollection &bregions =
            bcs.GetBoundaryRegions();
        const SpatialDomains::BoundaryConditionCollection &bconditions =
            bcs.GetBoundaryConditions();

        // Loop all boundary regions to create Fields and Operators, matching
        // components with a Neumann condition this operator applies. Everything
        // shared with DirBndCondOpImpl's identical loop lives in
        // BuildCGBndCondRegions(); onMatchedRegion() below builds the one piece
        // that is genuinely Neumann-specific -- the IProductWRTBaseOp (or, for
        // 0D point regions, a plain copy) that turns the evaluated boundary
        // values into boundary coefficients.
        auto regions = BuildCGBndCondRegions<ExecSpace, TData>(
            this->m_expansionList, this->m_components, bregions, bconditions,
            "Neumann BC", this->m_wsp_phys, this->m_wsp_coeffs,
            this->m_expressionOps, this->m_hasTimeDependentBndCoeffs,
            [this](const SpatialDomains::BoundaryConditionShPtr &bc) {
                return IsApplied(bc);
            },
            [&](const MultiRegions::ExpListSharedPtr &bcExpList) {
                m_neuNumCoeffs.push_back(bcExpList->GetNcoeffs());
                m_neuIsPoint.push_back(bcExpList->GetShapeDimension() == 0);

                // Note we do a copy for 0D (points). Keep this vector
                // aligned with Neumann boundary regions.
                if (!m_neuIsPoint.back())
                {
                    this->m_iprodOps.push_back(IProductWRTBaseOp<TData>::Create(
                        bcExpList, this->m_components, ExecSpace::name));
                }
                else
                {
                    this->m_iprodOps.push_back(nullptr);
                }
            });

        std::vector<std::vector<bool>> &isNeumannByRegion =
            regions.isTypeByRegion;
        std::vector<size_t> &numBcExpCoeffs = regions.numBcExpCoeffs;
        m_numBndCoeffCompSize               = regions.numBndCoeffCompSize;

        // Return if no Neumann boundary coefficients on any rank. Must be a
        // global reduction rather than the purely local
        // m_numBndCoeffCompSize: BuildCGBndCondCoeffMaps() below may build
        // a component's assembly map, which is collective over the row
        // communicator. If one rank's local partition has zero Neumann dofs
        // while another rank's has some, a per-rank-only check makes some
        // ranks skip that collective construction while others enter it,
        // deadlocking (see the identical fix/comment in
        // DirBndCondOpImpl's constructor).
        size_t hasAnyBndCoeff = m_numBndCoeffCompSize;
        session->GetComm()->GetRowComm()->AllReduce(hasAnyBndCoeff,
                                                    LibUtilities::ReduceMax);
        if (hasAnyBndCoeff == 0)
        {
            return;
        }

        // Collect the compact Neumann-to-full-boundary coefficient index map,
        // bucketed by execution block. Neumann has no extra per-component
        // bookkeeping beyond the shared map/sign construction, so the
        // per-component hook is a no-op.
        m_neuCoeffOffsets.clear();
        size_t bndcnt = 0;
        for (unsigned int i = 0; i < bregions.size(); ++i)
        {
            if (std::any_of(isNeumannByRegion[i].begin(),
                            isNeumannByRegion[i].end(),
                            [](bool b) { return b; }))
            {
                m_neuCoeffOffsets.push_back(bndcnt);
                bndcnt += numBcExpCoeffs[i];
            }
        }

        auto maps = BuildCGBndCondCoeffMaps<ExecSpace, TData>(
            this->m_expansionList, this->m_components, bregions,
            isNeumannByRegion, numBcExpCoeffs, m_numBndCoeffCompSize,
            [](unsigned int, const MultiRegions::AssemblyMapCGSharedPtr &,
               const auto &, const auto &) {});

        // v_GetNumBndDofs() reports the total number of matched boundary
        // trace coefficients -- capture that from the full map before
        // switching to the unique/group split below, so it stays independent
        // of how entries get folded for scatter-ADD.
        m_totalBndDofs = 0;
        for (auto &counts : maps.compCounts)
        {
            for (auto count : counts)
            {
                m_totalBndDofs += count;
            }
        }

        // Total number of grouped (duplicate-target) entries across every
        // block/component -- lets v_Apply() skip fetching the group
        // pointers entirely when zero, mirroring DirBndCondOpImpl's
        // m_maxLocalDup/m_maxSREntries early-out. On meshes where no
        // boundary element shares an edge/vertex between two boundary
        // trace pieces (e.g. an axis-aligned box), this is always zero.
        m_totalGroupCount = 0;
        for (auto &counts : maps.compGroupCounts)
        {
            for (auto count : counts)
            {
                m_totalGroupCount += count;
            }
        }

        // Neumann is a scatter-ADD, so it consumes the unique-target subset
        // of the shared map (plus the duplicate-target groups below)
        // instead of the full map -- see CGBndCondCoeffMaps::uniqueMap for
        // why summing the full map directly would double-count a duplicated
        // target's contribution.
        m_anySignChange    = maps.anySignChange;
        m_signChange       = std::move(maps.signChange);
        m_map              = std::move(maps.uniqueMap);
        m_sign             = std::move(maps.uniqueSign);
        m_bndCoeff         = std::move(maps.uniqueBndCoeff);
        m_bndCoeffSrc      = std::move(maps.uniqueBndCoeffSrc);
        m_compOffsets      = std::move(maps.compUniqueOffsets);
        m_compCounts       = std::move(maps.compUniqueCounts);
        m_groupTarget      = std::move(maps.groupTarget);
        m_groupOffset      = std::move(maps.groupOffset);
        m_groupSrc         = std::move(maps.groupSrc);
        m_groupSign        = std::move(maps.groupSign);
        m_groupValue       = std::move(maps.groupValue);
        m_compGroupOffsets = std::move(maps.compGroupOffsets);
        m_compGroupCounts  = std::move(maps.compGroupCounts);
        // Populated in full by v_UpdateBndCoeffs() every call, so it does
        // not need a zeroed starting value here (only ever written via
        // WriteOnly device access, never read before being written).
        m_compactBndCoeff =
            LibUtilities::MemoryRegion<TData>(nComp * m_numBndCoeffCompSize);
        // Allocated here, on the null stream, so the region block streams
        // that write it never race the allocation.
        m_compactBndCoeff.template GetPtr<MemSpace, WriteOnly>();

        // The region blocks write the compact array and the domain blocks
        // read the values gathered from it.
        size_t nUpdateStreams = m_bndCoeff.size();
        for (auto &wsp : m_wsp_coeffs)
        {
            nUpdateStreams = std::max(nUpdateStreams, wsp.GetBlocks().size());
        }
        for (unsigned int id = 1; id <= nUpdateStreams; ++id)
        {
            m_updateStreams.push_back(id);
        }

        this->UpdateBndCoeffs(static_cast<TData>(0.0));
    }

    // Leave the regions with user-defined type @p userDefinedType out,
    // rebuilding the operator if any rank applies one of them.
    void v_ExcludeUserDefinedType(const std::string &userDefinedType) override
    {
        auto session = this->m_expansionList->GetSession();
        const SpatialDomains::BoundaryConditions bcs(
            session, this->m_expansionList->GetGraph());
        const auto &bconditions = bcs.GetBoundaryConditions();

        size_t hasExcludedRegion = 0;
        for (const auto &[regionId, region] : bcs.GetBoundaryRegions())
        {
            const auto &conditionMap = *bconditions.at(regionId);
            for (const auto &component : this->m_components)
            {
                const auto &bc = conditionMap.at(component);
                if (IsApplied(bc) &&
                    StringUtils::IEquals(bc->GetUserDefined(), userDefinedType))
                {
                    hasExcludedRegion = 1;
                }
            }
        }
        m_excludedUserDefinedTypes.push_back(userDefinedType);

        // Collective, as BuildRegions() is: a rank holding none of the
        // regions rebuilds with the others.
        session->GetComm()->GetRowComm()->AllReduce(hasExcludedRegion,
                                                    LibUtilities::ReduceMax);
        if (hasExcludedRegion == 0)
        {
            return;
        }

        ClearRegions();
        BuildRegions();
    }

    // Return the operator to its state before BuildRegions().
    void ClearRegions()
    {
        this->m_bndCoeffsInitialised      = false;
        this->m_hasTimeDependentBndCoeffs = false;

        m_anySignChange       = false;
        m_numBndCoeffCompSize = 0;
        m_totalBndDofs        = 0;
        m_totalGroupCount     = 0;
        m_signChange.clear();
        m_updateStreams.clear();
        m_wsp_phys.clear();
        m_wsp_coeffs.clear();
        m_expressionOps.clear();
        this->m_iprodOps.clear();
        m_neuCoeffOffsets.clear();
        m_neuNumCoeffs.clear();
        m_neuIsPoint.clear();
        m_map.clear();
        m_sign.clear();
        m_bndCoeff.clear();
        m_bndCoeffSrc.clear();
        m_compactBndCoeff = LibUtilities::MemoryRegion<TData>();
        m_compOffsets.clear();
        m_compCounts.clear();
        m_groupTarget.clear();
        m_groupOffset.clear();
        m_groupSrc.clear();
        m_groupSign.clear();
        m_groupValue.clear();
        m_compGroupOffsets.clear();
        m_compGroupCounts.clear();
    }

    void v_UpdateBndCoeffs(const TData &time) override
    {
        if (m_numBndCoeffCompSize == 0)
        {
            return;
        }

        const unsigned int nComp = this->m_components.size();

        // Get a device pointer to the compact array: entirely on-device, no
        // host round-trip. No zeroing needed first -- m_neuCoeffOffsets/
        // m_neuNumCoeffs tile [0, m_numBndCoeffCompSize) exactly, and the
        // copy loop below writes every position for every component
        // (regardless of whether that component actually has a Neumann
        // condition there -- non-matching components get zero from
        // ExpressionOp's masked-out, zeroed wsp_phys, not a skipped write),
        // so every element gets overwritten unconditionally on every call.
        auto compactPtr =
            m_compactBndCoeff.template GetPtr<MemSpace, WriteOnly>(
                m_opStreamID);

        for (unsigned int iNeu = 0; iNeu < m_expressionOps.size(); ++iNeu)
        {
            const size_t nBndExpCoeff = m_neuNumCoeffs[iNeu];
            const size_t bndOffset    = m_neuCoeffOffsets[iNeu];

            // Zero wsp_phys and wsp_coeff
            Math::zero<ExecSpace>(m_wsp_phys[iNeu]);
            Math::zero<ExecSpace>(m_wsp_coeffs[iNeu]);

            // Update time
            m_expressionOps[iNeu]->SetTime(time);

            // Evaluate Neumann expression
            m_expressionOps[iNeu]->Apply(m_wsp_phys[iNeu], m_wsp_phys[iNeu]);

            // Evaluate inner product of Neumann BC
            if (!m_neuIsPoint[iNeu])
            {
                m_iprodOps[iNeu]->Apply(m_wsp_phys[iNeu], m_wsp_coeffs[iNeu]);
            }
            else
            {
                // A point region has a single block.
                const unsigned int blk      = 0;
                const unsigned int streamID = blk + 1;

                auto physptr =
                    m_wsp_phys[iNeu]
                        .GetBlocks()[blk]
                        .template GetPtr<MemSpace, ReadOnly>(streamID);
                auto coeffptr =
                    m_wsp_coeffs[iNeu]
                        .GetBlocks()[blk]
                        .template GetPtr<MemSpace, WriteOnly>(streamID);
                for (size_t nc = 0; nc < nComp; nc++)
                {
                    // Note use copyKernel instead of copy because of
                    // FieldState mismatch Phys != Coeff. In 0D, they are
                    // equivalent hence the copy is correct.
                    Math::copyKernel<ExecSpace>(nBndExpCoeff, physptr, coeffptr,
                                                streamID);
                    physptr += m_wsp_phys[iNeu].GetBlocks()[blk].CompSize();
                    coeffptr += m_wsp_coeffs[iNeu].GetBlocks()[blk].CompSize();
                }
            }

            // Copy m_wsp_coeffs[iNeu] into the compact array, component by
            // component, block by block (mirrors Field::ToVector()'s own
            // block-concatenation but writes straight into device memory
            // instead of a host std::vector).
            size_t regionOffset = 0;
            for (unsigned int wblk = 0;
                 wblk < m_wsp_coeffs[iNeu].GetBlocks().size(); ++wblk)
            {
                const unsigned int streamID = wblk + 1;

                auto &wspBlk = m_wsp_coeffs[iNeu].GetBlocks()[wblk];
                auto srcPtr =
                    wspBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
                const auto nSize = wspBlk.CompSize();
                const auto blockLen =
                    wspBlk.GetNumElements() * wspBlk.GetNumData();

                for (unsigned int nc = 0; nc < nComp; ++nc)
                {
                    Math::copyKernel<ExecSpace>(blockLen, srcPtr + nc * nSize,
                                                compactPtr +
                                                    nc * m_numBndCoeffCompSize +
                                                    bndOffset + regionOffset,
                                                streamID);
                }
                regionOffset += blockLen;
            }

            ASSERTL1(regionOffset == nBndExpCoeff,
                     "Unexpected boundary coefficient vector size.");
        }

        // The gather reads entries written on any region block's stream and
        // overwrites values the domain block streams last read.
        JoinStreams<ExecSpace>(m_opStreamID, m_updateStreams);

        for (unsigned int blk = 0; blk < m_bndCoeff.size(); ++blk)
        {
            auto idxPtr =
                m_bndCoeffSrc[blk].template GetPtr<MemSpace, ReadOnly>(
                    m_opStreamID);
            auto dstPtr = m_bndCoeff[blk].template GetPtr<MemSpace, WriteOnly>(
                m_opStreamID);

            CGBndCondGatherKernel<ExecSpace>(m_bndCoeffSrc[blk].size(), idxPtr,
                                             compactPtr, dstPtr, m_opStreamID);

            // Gathered unconditionally (nsize may be 0, in which case the
            // kernel is a no-op) so that m_groupValue[blk] is always marked
            // valid before v_Apply() reads it -- mirrors m_bndCoeff above.
            auto groupIdxPtr =
                m_groupSrc[blk].template GetPtr<MemSpace, ReadOnly>(
                    m_opStreamID);
            auto groupDstPtr =
                m_groupValue[blk].template GetPtr<MemSpace, WriteOnly>(
                    m_opStreamID);

            CGBndCondGatherKernel<ExecSpace>(m_groupSrc[blk].size(),
                                             groupIdxPtr, compactPtr,
                                             groupDstPtr, m_opStreamID);
        }

        // Order the work queued next on the block streams after the gather.
        ReleaseStreams<ExecSpace>(m_opStreamID, m_updateStreams);
    }

    size_t v_GetNumBndDofs() const override
    {
        return m_totalBndDofs;
    }

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &inout) override
    {
        // Return if no Neumann boundary condition.
        if (m_numBndCoeffCompSize == 0)
        {
            return;
        }

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &inoutBlk   = inout.GetBlocks()[blk];
            auto inoutWidth  = inoutBlk.GetInterleaveWidth();
            unsigned blksize = inoutBlk.CompSize();

            // Initialize pointers.
            auto inoutPtr =
                inoutBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto mapPtrBlock =
                m_map[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
            auto bndcoeffPtrBlock =
                m_bndCoeff[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
            const TData *signPtrBlock =
                m_anySignChange
                    ? m_sign[blk].template GetPtr<MemSpace, ReadOnly>(streamID)
                    : nullptr;

            // Pointers for the (rare) grouped/duplicate-target entries --
            // see NeuBndCondGroupKernel(). groupOffsetPtrBlock/
            // groupTargetPtrBlock are indexed by group, groupValuePtrBlock/
            // groupSignPtrBlock by (flat) contribution. Only fetched when
            // m_totalGroupCount > 0: on meshes with no duplicate-target
            // entries at all (e.g. an axis-aligned box, where boundary
            // trace pieces never share a vertex/edge within the same
            // element), every m_compGroupCounts[blk][nc] below is zero, so
            // these four GetPtr() calls would otherwise be pure overhead
            // paid on every block, every Apply() call, for nothing.
            const size_t *groupOffsetPtrBlock = nullptr;
            const size_t *groupTargetPtrBlock = nullptr;
            const TData *groupValuePtrBlock   = nullptr;
            const TData *groupSignPtrBlock    = nullptr;
            if (m_totalGroupCount > 0)
            {
                groupOffsetPtrBlock =
                    m_groupOffset[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID);
                groupTargetPtrBlock =
                    m_groupTarget[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID);
                groupValuePtrBlock =
                    m_groupValue[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID);
                if (m_anySignChange)
                {
                    groupSignPtrBlock =
                        m_groupSign[blk].template GetPtr<MemSpace, ReadOnly>(
                            streamID);
                }
            }

            // if block is interlaced deInterleave block since currently
            // mapping set up assuming serial alignment. Reshape once for
            // the whole block (all components) rather than per-component.
            LibUtilities::ReshapeStorage<ExecSpace>(
                1u, inoutWidth,
                inoutBlk.GetNumElementsWithPadding() * inout.GetNumComponents(),
                inoutBlk.GetNumData(), inoutPtr, streamID);

            // Add weak boundary condition forcing.
            for (unsigned nc = 0; nc < inout.GetNumComponents(); ++nc)
            {
                auto nbndCoeffBlk = m_compCounts[blk][nc];
                if (nbndCoeffBlk > 0)
                {
                    const size_t offset = m_compOffsets[blk][nc];
                    auto mapPtr         = mapPtrBlock + offset;
                    auto bndcoeffPtr    = bndcoeffPtrBlock + offset;
                    const TData *signPtr =
                        m_signChange[nc] ? signPtrBlock + offset : nullptr;

                    if (m_signChange[nc])
                    {
                        NeuBndCondKernel<ExecSpace>(
                            nbndCoeffBlk, signPtr, mapPtr, bndcoeffPtr,
                            inoutPtr + nc * blksize, streamID);
                    }
                    else
                    {
                        NeuBndCondKernel<ExecSpace>(
                            nbndCoeffBlk, mapPtr, bndcoeffPtr,
                            inoutPtr + nc * blksize, streamID);
                    }
                }

                auto nGroupsBlk = m_compGroupCounts[blk][nc];
                if (nGroupsBlk == 0)
                {
                    continue;
                }

                const size_t groupOffset = m_compGroupOffsets[blk][nc];
                auto groupOffsetPtr      = groupOffsetPtrBlock + groupOffset;
                auto groupTargetPtr      = groupTargetPtrBlock + groupOffset;

                if (m_signChange[nc])
                {
                    NeuBndCondGroupKernel<ExecSpace>(
                        nGroupsBlk, groupOffsetPtr, groupTargetPtr,
                        groupSignPtrBlock, groupValuePtrBlock,
                        inoutPtr + nc * blksize, streamID);
                }
                else
                {
                    NeuBndCondGroupKernel<ExecSpace>(
                        nGroupsBlk, groupOffsetPtr, groupTargetPtr,
                        groupValuePtrBlock, inoutPtr + nc * blksize, streamID);
                }
            }

            // Reshape back, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                inoutWidth, 1u,
                inoutBlk.GetNumElementsWithPadding() * inout.GetNumComponents(),
                inoutBlk.GetNumData(), inoutPtr, streamID);
        }
    }
};

} // namespace Nektar::MultiRegions::detail
