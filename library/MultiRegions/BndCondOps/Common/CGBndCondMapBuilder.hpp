///////////////////////////////////////////////////////////////////////////////
//
// File: CGBndCondMapBuilder.hpp
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
// Description: Shared boundary-coefficient map construction for continuous
// Galerkin (CG) boundary condition operators (Dirichlet lifting, Neumann weak
// forcing). Both operators need to map a compact per-boundary-condition-type
// coefficient array onto the (block-partitioned) domain coefficient array,
// bucketed by device/host execution block and split by sign-changed dofs.
// This builder captures that shared bucketing/sorting logic so it is
// implemented and tested in exactly one place.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include <MultiRegions/ContField.h>

#include "LibUtilities/BasicUtils/Field/Field.hpp"
#include <MultiRegions/ElmtOps/Expression/ExpressionOp.hpp>

namespace Nektar::MultiRegions::detail
{

// Result of BuildCGBndCondRegions(): boundary-region bookkeeping shared by
// the Dirichlet and Neumann constructors, needed afterwards to build the
// compact-to-domain coefficient maps via BuildCGBndCondCoeffMaps().
template <typename TData> struct CGBndCondRegions
{
    // isTypeByRegion[i][nc]: whether component nc of boundary region i has
    // the boundary condition type this operator cares about.
    std::vector<std::vector<bool>> isTypeByRegion;

    // Number of boundary expansion coefficients for *every* region (matching
    // or not), matching BuildCGBndCondCoeffMaps()'s numBcExpCoeffs, since the
    // running offset there must account for skipped regions too.
    std::vector<size_t> numBcExpCoeffs;

    // Total number of matching boundary coefficients across all regions, for
    // a single component.
    size_t numBndCoeffCompSize = 0;
};

// Shared per-boundary-region setup for a CG boundary condition operator
// (Dirichlet lifting, Neumann weak forcing): for every boundary region with
// at least one matching component, builds the boundary expansion list, the
// phys/coeff workspace fields, and the ExpressionOp that evaluates the
// boundary condition equations into them. What counts as a "match" and what
// type-specific operator turns the evaluated boundary values into boundary
// coefficients (FwdTransBCOp, IProductWRTBaseOp, ...) is supplied by the
// caller, since that is the one part that genuinely differs between
// Dirichlet and Neumann.
//
// - isMatch(bc) returns whether a BoundaryCondition is the type this
//   operator cares about (e.g. a dynamic_pointer_cast check).
// - onMatchedRegion(bcExpList) is invoked once per matching region, right
//   after the shared workspace/ExpressionOp setup for that region (so
//   ordering matches wspPhys/wspCoeffs/expressionOps), letting the caller
//   build its own type-specific projection operator and any bookkeeping that
//   depends on bcExpList (e.g. Dirichlet's m_dirNumCoeffs, Neumann's
//   m_neuNumCoeffs/m_neuIsPoint).
template <typename ExecSpace, typename TData, typename IsMatchFn,
          typename OnMatchedRegionFn>
CGBndCondRegions<TData> BuildCGBndCondRegions(
    const MultiRegions::ExpListSharedPtr &expansionList,
    const std::vector<std::string> &components,
    const SpatialDomains::BoundaryRegionCollection &bregions,
    const SpatialDomains::BoundaryConditionCollection &bconditions,
    const std::string &wspLabel,
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> &wspPhys,
    std::vector<LibUtilities::Field<TData, FieldState::Coeff>> &wspCoeffs,
    std::vector<std::shared_ptr<ExpressionOp<TData>>> &expressionOps,
    bool &hasTimeDependentBndCoeffs, IsMatchFn &&isMatch,
    OnMatchedRegionFn &&onMatchedRegion)
{
    using MemSpace           = typename ExecSpace::memory_space;
    const unsigned int nComp = components.size();

    auto session       = expansionList->GetSession();
    auto graph         = expansionList->GetGraph();
    unsigned int nhomo = 1;
    session->LoadParameter("HomModesZ", nhomo, 1);

    CGBndCondRegions<TData> result;

    for (auto &it : bregions)
    {
        auto regionId       = it.first;
        auto collectionIter = bconditions.find(regionId);

        ASSERTL1(collectionIter != bconditions.end(),
                 "Unable to locate collection " + std::to_string(regionId));

        const SpatialDomains::BoundaryConditionMapShPtr bndCondMap =
            (*collectionIter).second;

        // Note use just the first component as we are just looking for the
        // boundary expansion list
        auto conditionMapIter = bndCondMap->find(components[0]);

        ASSERTL1(conditionMapIter != bndCondMap->end(),
                 "Unable to locate condition map.");

        SpatialDomains::BoundaryConditionShPtr bc = (*conditionMapIter).second;

        // Create boundary condition expansion
        auto bcExpList =
            MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                session, *(it.second), graph, true, components[0], false,
                bc->GetComm(), Collections::eNoImpType);

        // Set data warehouse for device support operators
        bcExpList->SetDataWarehouse();

        // Save number of coefficients
        result.numBcExpCoeffs.push_back(bcExpList->GetNcoeffs());

        // Check if any component has a matching condition
        bool hasMatch = false;
        std::vector<bool> isMatchComp(nComp);
        for (unsigned int nc = 0; nc < nComp; nc++)
        {
            auto cndMapIter = bndCondMap->find(components[nc]);
            bc              = (*cndMapIter).second;
            bool tmp        = isMatch(bc);

            isMatchComp[nc] = tmp;
            hasMatch        = hasMatch || tmp;
            hasTimeDependentBndCoeffs =
                hasTimeDependentBndCoeffs || (tmp && bc->IsTimeDependent());
        }

        result.isTypeByRegion.push_back(isMatchComp);

        // Skip, if no matching condition in this boundary region
        if (!hasMatch)
        {
            continue;
        }

        // Create fields for evaluating BCs into
        auto blocks_phys =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                bcExpList);
        wspPhys.push_back(LibUtilities::Field<TData, FieldState::Phys>(
            wspLabel + " phys", blocks_phys, nComp, nhomo));
        auto blocks_coeffs =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                bcExpList);
        wspCoeffs.push_back(LibUtilities::Field<TData, FieldState::Coeff>(
            wspLabel + " coeff", blocks_coeffs, nComp, nhomo));

        // Compute number of boundary coefficients.
        for (unsigned int blk = 0; blk < blocks_coeffs.size(); ++blk)
        {
            const auto ncoeff = blocks_coeffs[blk].GetNumData();
            const auto nelmt  = blocks_coeffs[blk].GetNumElements();
            result.numBndCoeffCompSize += nelmt * ncoeff;
        }

        // Initialize memory regions
        wspPhys.back().template Initialize<MemSpace>(0.0);
        wspCoeffs.back().template Initialize<MemSpace>(0.0);

        // Create operators for this boundary condition
        expressionOps.push_back(
            ExpressionOp<TData>::Create(bcExpList, components));
        expressionOps.back()->SetComponentMask(isMatchComp);

        // Gather equations for each field/component
        std::vector<LibUtilities::EquationSharedPtr> listOfEquations;
        for (unsigned int nc = 0; nc < nComp; nc++)
        {
            auto cndMapIter = bndCondMap->find(components[nc]);
            bc              = (*cndMapIter).second;
            listOfEquations.push_back(bc->GetEquation());
        }

        // Set boundary conditions for each component in this boundary region
        expressionOps.back()->SetExpressions(listOfEquations);

        onMatchedRegion(bcExpList);
    }

    return result;
}

// Result of BuildCGBndCondCoeffMaps(): everything needed by v_Apply()/
// v_UpdateBndCoeffs() to scatter a compact per-component boundary coefficient
// array into the (block-partitioned) domain coefficient array.
template <typename TData> struct CGBndCondCoeffMaps
{
    bool anySignChange = false;
    std::vector<bool> signChange;

    // One entry per execution block, per matching boundary trace coefficient
    // -- including any local coefficient target written by more than one
    // trace piece (a domain corner where two matching-type edges of the same
    // element meet). bndCoeffSrc holds, for each block, the device-resident
    // gather index used to populate bndCoeff[blk] directly from the compact
    // per-component boundary coefficient array (see CGBndCondGatherKernel)
    // -- kept device-resident so v_UpdateBndCoeffs() never needs to
    // round-trip through host memory.
    //
    // Safe as-is for a "set" consumer (Dirichlet), where two trace pieces
    // writing the same target are (at worst) a harmless last-write-wins race
    // -- both are independent evaluations of the same boundary condition at
    // the same point. A scatter-ADD consumer (Neumann) must NOT use this
    // array directly, since summing every entry would double-count a
    // duplicated target's contribution; use uniqueMap + groupTarget below
    // instead, which partition this same set of entries so every
    // contribution is counted exactly once.
    std::vector<LibUtilities::MemoryRegion<size_t>> map;
    std::vector<LibUtilities::MemoryRegion<TData>> sign;
    std::vector<LibUtilities::MemoryRegion<TData>> bndCoeff;
    std::vector<LibUtilities::MemoryRegion<size_t>> bndCoeffSrc;

    // The unique-target subset of map/sign/bndCoeff/bndCoeffSrc above --
    // every target here is written by exactly one entry, so a scatter-ADD
    // consumer can add it with a plain, non-atomic store. Duplicate-target
    // entries excluded here are folded into the "grouped" arrays below
    // instead, one group per duplicated target: group g's contributing
    // entries are the CSR slice [groupOffset[blk][g], groupOffset[blk][g +
    // 1]) into groupSrc/groupSign/groupValue (groupOffset[blk] therefore has
    // numGroups + 1 entries). Summing a group locally and writing it with one
    // more non-atomic store accounts for the remaining (rare) duplicated
    // contributions -- together, uniqueMap and the groups cover every entry
    // in map exactly once, without ever racing an atomic_add.
    std::vector<LibUtilities::MemoryRegion<size_t>> uniqueMap;
    std::vector<LibUtilities::MemoryRegion<TData>> uniqueSign;
    std::vector<LibUtilities::MemoryRegion<TData>> uniqueBndCoeff;
    std::vector<LibUtilities::MemoryRegion<size_t>> uniqueBndCoeffSrc;

    std::vector<LibUtilities::MemoryRegion<size_t>> groupTarget;
    std::vector<LibUtilities::MemoryRegion<size_t>> groupOffset;
    std::vector<LibUtilities::MemoryRegion<size_t>> groupSrc;
    std::vector<LibUtilities::MemoryRegion<TData>> groupSign;
    std::vector<LibUtilities::MemoryRegion<TData>> groupValue;

    // Indexed [block][component].
    std::vector<std::vector<size_t>> compOffsets;
    std::vector<std::vector<size_t>> compCounts;
    std::vector<std::vector<size_t>> compUniqueOffsets;
    std::vector<std::vector<size_t>> compUniqueCounts;
    // Indexed [block][component], counting groups (not underlying entries).
    std::vector<std::vector<size_t>> compGroupOffsets;
    std::vector<std::vector<size_t>> compGroupCounts;
};

// Build the compact-boundary-coefficient-to-domain-coefficient maps for a CG
// boundary condition operator (Dirichlet or Neumann).
//
// - isTypeByRegion[i][nc] indicates whether component nc of boundary region i
//   has the boundary condition type this operator cares about.
// - numBcExpCoeffs[i] is the number of boundary expansion coefficients for
//   boundary region i (for *every* region, matching or not, since the
//   running offset must account for skipped regions too).
// - numBndCoeffCompSize is the total number of matching boundary coefficients
//   across all regions, for a single component.
// - perComponent(nc, assmbMap, domainBlocks, blockBound) is invoked once per
//   field component after the shared per-component map/sign bucketing has
//   been computed, so that callers needing extra information from the same
//   assembly map (e.g. Dirichlet's local-dir-dofs and parallel-dir-sign
//   bookkeeping) do not need to reconstruct it.
template <typename ExecSpace, typename TData, typename PerComponentFn>
CGBndCondCoeffMaps<TData> BuildCGBndCondCoeffMaps(
    const MultiRegions::ExpListSharedPtr &expansionList,
    const std::vector<std::string> &components,
    const SpatialDomains::BoundaryRegionCollection &bregions,
    const std::vector<std::vector<bool>> &isTypeByRegion,
    const std::vector<size_t> &numBcExpCoeffs, size_t numBndCoeffCompSize,
    PerComponentFn &&perComponent)
{
    using MemSpace           = typename ExecSpace::memory_space;
    const unsigned int nComp = components.size();

    CGBndCondCoeffMaps<TData> result;

    auto domainBlocks =
        MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
            expansionList);
    std::vector<size_t> blockBound(domainBlocks.size());
    size_t bound = 0;
    for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
    {
        const auto &block = domainBlocks[blk];
        const auto ncoeff = block.GetNumData();
        const auto nelmt  = block.GetNumElements();
        bound += nelmt * ncoeff;
        blockBound[blk] = bound;
    }

    // Collect the compact type-to-full-boundary coefficient index map for
    // one component. Boundary coefficients themselves are dynamic and are
    // filled in by the caller's v_UpdateBndCoeffs().
    std::vector<size_t> index(numBndCoeffCompSize);
    std::vector<std::vector<size_t>> typeIndexByComp(nComp);
    size_t bndcnt = 0, cnt = 0;
    for (unsigned int i = 0; i < bregions.size(); ++i)
    {
        auto nBndExpCoeff = numBcExpCoeffs[i];

        if (std::any_of(isTypeByRegion[i].begin(), isTypeByRegion[i].end(),
                        [](bool b) { return b; }))
        {
            for (size_t j = 0; j < nBndExpCoeff; ++j)
            {
                index[bndcnt + j] = cnt + j;
            }
            for (unsigned int nc = 0; nc < nComp; ++nc)
            {
                if (isTypeByRegion[i][nc])
                {
                    for (size_t j = 0; j < nBndExpCoeff; ++j)
                    {
                        typeIndexByComp[nc].push_back(bndcnt + j);
                    }
                }
            }
            bndcnt += nBndExpCoeff;
        }
        cnt += nBndExpCoeff;
    }

    ASSERTL1(bndcnt == numBndCoeffCompSize,
             "The component size does not match the number of coefficients "
             "for all boundary conditions of this type.")

    result.signChange.resize(nComp, false);
    result.compOffsets.assign(domainBlocks.size(),
                              std::vector<size_t>(nComp, 0));
    result.compCounts.assign(domainBlocks.size(),
                             std::vector<size_t>(nComp, 0));
    result.compUniqueOffsets.assign(domainBlocks.size(),
                                    std::vector<size_t>(nComp, 0));
    result.compUniqueCounts.assign(domainBlocks.size(),
                                   std::vector<size_t>(nComp, 0));
    result.compGroupOffsets.assign(domainBlocks.size(),
                                   std::vector<size_t>(nComp, 0));
    result.compGroupCounts.assign(domainBlocks.size(),
                                  std::vector<size_t>(nComp, 0));

    // Full set of entries (Dirichlet's view) -- every matching boundary
    // trace coefficient, including duplicated targets.
    std::vector<std::vector<size_t>> mapBlockByBlk(domainBlocks.size());
    std::vector<std::vector<TData>> signBlockByBlk(domainBlocks.size());
    std::vector<std::vector<size_t>> bndCoeffSrcBlockByBlk(domainBlocks.size());

    // The same entries, partitioned into a unique-target subset and
    // duplicate-target groups (Neumann's view) -- see
    // CGBndCondCoeffMaps::uniqueMap/groupTarget for what these hold and why.
    std::vector<std::vector<size_t>> uniqueMapBlockByBlk(domainBlocks.size());
    std::vector<std::vector<TData>> uniqueSignBlockByBlk(domainBlocks.size());
    std::vector<std::vector<size_t>> uniqueBndCoeffSrcBlockByBlk(
        domainBlocks.size());

    std::vector<std::vector<size_t>> groupTargetByBlk(domainBlocks.size());
    std::vector<std::vector<size_t>> groupOffsetByBlk(domainBlocks.size());
    std::vector<std::vector<size_t>> groupSrcByBlk(domainBlocks.size());
    std::vector<std::vector<TData>> groupSignByBlk(domainBlocks.size());

    auto session = expansionList->GetSession();
    auto graph   = expansionList->GetGraph();
    auto expContField =
        std::dynamic_pointer_cast<MultiRegions::ContField>(expansionList);

    for (unsigned int nc = 0; nc < nComp; ++nc)
    {
        MultiRegions::AssemblyMapCGSharedPtr assmbMap;
        if (expContField &&
            expContField->GetLocalToGlobalMap()->GetVariable() ==
                components[nc])
        {
            assmbMap = expContField->GetLocalToGlobalMap();
        }
        else
        {
            MultiRegions::ContField compfield(session, graph, components[nc],
                                              true, false,
                                              Collections::eNoCollection);
            assmbMap = compfield.GetLocalToGlobalMap();
        }
        auto &sign            = assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map             = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();
        result.signChange[nc] = assmbMap->GetSignChange();

        std::vector<std::tuple<size_t, size_t>> mapReordered;
        mapReordered.reserve(typeIndexByComp[nc].size());
        for (size_t idx : typeIndexByComp[nc])
        {
            mapReordered.push_back(std::make_tuple(idx, map[index[idx]]));
        }
        std::sort(mapReordered.begin(), mapReordered.end(),
                  [](std::tuple<size_t, size_t> const &t1,
                     std::tuple<size_t, size_t> const &t2) {
                      return std::tie(std::get<1>(t1), std::get<0>(t1)) <
                             std::tie(std::get<1>(t2), std::get<0>(t2));
                  });

        std::vector<std::vector<size_t>> blockIndices(domainBlocks.size());
        size_t cursor = 0;
        for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
        {
            while (cursor < mapReordered.size() &&
                   std::get<1>(mapReordered[cursor]) < blockBound[blk])
            {
                blockIndices[blk].push_back(cursor);
                cursor++;
            }
        }

        for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
        {
            const size_t offset     = (blk == 0) ? 0 : blockBound[blk - 1];
            const size_t compOffset = mapBlockByBlk[blk].size();
            const size_t compUniqueOffset     = uniqueMapBlockByBlk[blk].size();
            const size_t compGroupOffset      = groupTargetByBlk[blk].size();
            result.compOffsets[blk][nc]       = compOffset;
            result.compUniqueOffsets[blk][nc] = compUniqueOffset;
            result.compGroupOffsets[blk][nc]  = compGroupOffset;

            const auto &blkIdx = blockIndices[blk];
            mapBlockByBlk[blk].reserve(compOffset + blkIdx.size());
            signBlockByBlk[blk].reserve(compOffset + blkIdx.size());
            bndCoeffSrcBlockByBlk[blk].reserve(compOffset + blkIdx.size());
            uniqueMapBlockByBlk[blk].reserve(compUniqueOffset + blkIdx.size());
            uniqueSignBlockByBlk[blk].reserve(compUniqueOffset + blkIdx.size());
            uniqueBndCoeffSrcBlockByBlk[blk].reserve(compUniqueOffset +
                                                     blkIdx.size());

            // blkIdx is sorted by target local coefficient index, so any
            // duplicate targets -- two boundary trace pieces of the same
            // element sharing a vertex/edge -- form a contiguous run. Every
            // entry always goes into the full map/sign/bndCoeffSrc (below);
            // a run of length 1 (the overwhelming majority) additionally
            // goes into the unique-target arrays, while longer runs are
            // folded into one "group" entry instead, so a scatter-ADD
            // consumer can sum their contributions locally and write them
            // once, rather than racing an atomic_add.
            size_t pos = 0;
            while (pos < blkIdx.size())
            {
                const size_t target = std::get<1>(mapReordered[blkIdx[pos]]);
                size_t runEnd       = pos + 1;
                while (runEnd < blkIdx.size() &&
                       std::get<1>(mapReordered[blkIdx[runEnd]]) == target)
                {
                    ++runEnd;
                }

                for (size_t r = pos; r < runEnd; ++r)
                {
                    const size_t idx = blkIdx[r];
                    mapBlockByBlk[blk].push_back(target - offset);
                    signBlockByBlk[blk].push_back(
                        result.signChange[nc]
                            ? sign[index[std::get<0>(mapReordered[idx])]]
                            : static_cast<TData>(1));
                    bndCoeffSrcBlockByBlk[blk].push_back(
                        std::get<0>(mapReordered[idx]) +
                        nc * numBndCoeffCompSize);
                }

                if (runEnd - pos == 1)
                {
                    const size_t idx = blkIdx[pos];
                    uniqueMapBlockByBlk[blk].push_back(target - offset);
                    uniqueSignBlockByBlk[blk].push_back(
                        result.signChange[nc]
                            ? sign[index[std::get<0>(mapReordered[idx])]]
                            : static_cast<TData>(1));
                    uniqueBndCoeffSrcBlockByBlk[blk].push_back(
                        std::get<0>(mapReordered[idx]) +
                        nc * numBndCoeffCompSize);
                }
                else
                {
                    groupTargetByBlk[blk].push_back(target - offset);
                    groupOffsetByBlk[blk].push_back(groupSrcByBlk[blk].size());
                    for (size_t r = pos; r < runEnd; ++r)
                    {
                        const size_t idx = blkIdx[r];
                        groupSignByBlk[blk].push_back(
                            result.signChange[nc]
                                ? sign[index[std::get<0>(mapReordered[idx])]]
                                : static_cast<TData>(1));
                        groupSrcByBlk[blk].push_back(
                            std::get<0>(mapReordered[idx]) +
                            nc * numBndCoeffCompSize);
                    }
                }

                pos = runEnd;
            }

            result.compCounts[blk][nc] = mapBlockByBlk[blk].size() - compOffset;
            result.compUniqueCounts[blk][nc] =
                uniqueMapBlockByBlk[blk].size() - compUniqueOffset;
            result.compGroupCounts[blk][nc] =
                groupTargetByBlk[blk].size() - compGroupOffset;
        }
        result.anySignChange = result.anySignChange || result.signChange[nc];

        perComponent(nc, assmbMap, domainBlocks, blockBound);
    }

    result.map.reserve(domainBlocks.size());
    result.bndCoeff.reserve(domainBlocks.size());
    result.bndCoeffSrc.reserve(domainBlocks.size());
    result.uniqueMap.reserve(domainBlocks.size());
    result.uniqueBndCoeff.reserve(domainBlocks.size());
    result.uniqueBndCoeffSrc.reserve(domainBlocks.size());
    result.groupTarget.reserve(domainBlocks.size());
    result.groupOffset.reserve(domainBlocks.size());
    result.groupSrc.reserve(domainBlocks.size());
    result.groupValue.reserve(domainBlocks.size());
    if (result.anySignChange)
    {
        result.sign.reserve(domainBlocks.size());
        result.uniqueSign.reserve(domainBlocks.size());
        result.groupSign.reserve(domainBlocks.size());
    }
    for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
    {
        ASSERTL1(mapBlockByBlk[blk].size() == bndCoeffSrcBlockByBlk[blk].size(),
                 "Mismatch between map and boundary coefficient sizes.");
        result.map.push_back(
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                    size_t>(
                mapBlockByBlk[blk]));
        // bndCoeff[blk] is populated in full by CGBndCondGatherKernel on
        // every UpdateBndCoeffs() call, so it does not need a zeroed
        // starting value here.
        result.bndCoeff.emplace_back(bndCoeffSrcBlockByBlk[blk].size());
        result.bndCoeffSrc.push_back(
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                    size_t>(
                bndCoeffSrcBlockByBlk[blk]));
        if (result.anySignChange)
        {
            result.sign.push_back(
                LibUtilities::MemoryRegion<TData>::template FromVector<MemSpace,
                                                                       TData>(
                    signBlockByBlk[blk]));
        }

        result.uniqueMap.push_back(
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                    size_t>(
                uniqueMapBlockByBlk[blk]));
        // uniqueBndCoeff[blk] is populated in full by CGBndCondGatherKernel
        // on every UpdateBndCoeffs() call, same as bndCoeff above.
        result.uniqueBndCoeff.emplace_back(
            uniqueBndCoeffSrcBlockByBlk[blk].size());
        result.uniqueBndCoeffSrc.push_back(
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                    size_t>(
                uniqueBndCoeffSrcBlockByBlk[blk]));
        if (result.anySignChange)
        {
            result.uniqueSign.push_back(
                LibUtilities::MemoryRegion<TData>::template FromVector<MemSpace,
                                                                       TData>(
                    uniqueSignBlockByBlk[blk]));
        }

        // Close off the group CSR with a trailing sentinel (the total
        // number of grouped contributions), so group g's slice is always
        // [groupOffset[g], groupOffset[g + 1]).
        groupOffsetByBlk[blk].push_back(groupSrcByBlk[blk].size());

        result.groupTarget.push_back(
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                    size_t>(
                groupTargetByBlk[blk]));
        result.groupOffset.push_back(
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                    size_t>(
                groupOffsetByBlk[blk]));
        result.groupSrc.push_back(
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                    size_t>(
                groupSrcByBlk[blk]));
        // groupValue[blk] is gathered in full by CGBndCondGatherKernel on
        // every UpdateBndCoeffs() call, same as bndCoeff above.
        result.groupValue.emplace_back(groupSrcByBlk[blk].size());
        if (result.anySignChange)
        {
            result.groupSign.push_back(
                LibUtilities::MemoryRegion<TData>::template FromVector<MemSpace,
                                                                       TData>(
                    groupSignByBlk[blk]));
        }
    }

    return result;
}

} // namespace Nektar::MultiRegions::detail
