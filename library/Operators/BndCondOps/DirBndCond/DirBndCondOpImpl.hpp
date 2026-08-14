///////////////////////////////////////////////////////////////////////////////
//
// File: DirBndCondOpImpl.hpp
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

#include <memory>
#include <unordered_map>
// AssemblyCommCG.h uses std::unordered_set/std::unordered_map without
// including the corresponding headers itself; include them here first so it
// does not depend on whatever an enclosing translation unit happens to have
// pulled in.
#include <unordered_set>
#include <vector>

#include <MultiRegions/AssemblyMap/AssemblyCommCG.h>
#include <MultiRegions/ContField.h>

#include "Operators/BndCondOps/Common/CGBndCondKernels.hpp"
#include "Operators/BndCondOps/Common/CGBndCondMapBuilder.hpp"
#include "Operators/BndCondOps/DirBndCond/DirBndCondKernels.hpp"
#include "Operators/BndCondOps/DirBndCond/DirBndCondOp.hpp"
#include "Operators/BndCondOps/FwdTransBC/FwdTransBCOp.hpp"

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "MultiRegions/Field/Math.hpp"
#include "Operators/ElmtOps/Expression/ExpressionOp.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class DirBndCondOpImpl : public DirBndCondOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DirBndCondOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : DirBndCondOp<TData>(expansionList, components)
    {
        auto session       = this->m_expansionList->GetSession();
        auto graph         = this->m_expansionList->GetGraph();
        unsigned int nComp = components.size();

        // Create boundary conditions
        const SpatialDomains::BoundaryConditions bcs(session, graph);
        const SpatialDomains::BoundaryRegionCollection &bregions =
            bcs.GetBoundaryRegions();
        const SpatialDomains::BoundaryConditionCollection &bconditions =
            bcs.GetBoundaryConditions();

        // Loop all boundary regions to create Fields and Operators, matching
        // components with a Dirichlet condition. Everything shared with
        // NeuBndCondOpImpl's identical loop lives in BuildCGBndCondRegions();
        // onMatchedRegion() below builds the one piece that is genuinely
        // Dirichlet-specific -- the FwdTransBCOp that lifts the evaluated
        // boundary values into boundary coefficients.
        auto regions = BuildCGBndCondRegions<ExecSpace, TData>(
            this->m_expansionList, components, bregions, bconditions,
            "Dirichlet BC", this->m_wsp_phys, this->m_wsp_coeffs,
            this->m_expressionOps, this->m_hasTimeDependentBndCoeffs,
            [](const SpatialDomains::BoundaryConditionShPtr &bc) {
                return (bool)std::dynamic_pointer_cast<
                    SpatialDomains::DirichletBoundaryCondition>(bc);
            },
            [&](const MultiRegions::ExpListSharedPtr &bcExpList) {
                m_dirNumCoeffs.push_back(bcExpList->GetNcoeffs());
                this->m_fwdTransBCOps.push_back(FwdTransBCOp<TData>::Create(
                    bcExpList, components, ExecSpace::name));
            });

        std::vector<std::vector<bool>> &isDirichletByRegion =
            regions.isTypeByRegion;
        std::vector<size_t> &numBcExpCoeffs = regions.numBcExpCoeffs;
        m_numBndCoeffCompSize               = regions.numBndCoeffCompSize;

        // Return if no Dirichlet boundary coefficients on any rank. This
        // must be a global reduction rather than the purely local
        // m_numBndCoeffCompSize: BuildCGBndCondCoeffMaps() below constructs
        // a fresh ContField per component, which is collective over the row
        // communicator, and SetUpUniversalDirComm() below builds each
        // component's AssemblyCommCG, whose shared-id discovery is itself a
        // ring exchange every rank must enter. If one rank's local partition
        // has zero
        // Dirichlet dofs while another rank's has some (entirely possible,
        // e.g. Dirichlet boundaries confined to one end of a partitioned 1D
        // mesh), a per-rank-only check makes some ranks skip these
        // collectives while others enter them, deadlocking.
        size_t hasAnyBndCoeff = m_numBndCoeffCompSize;
        session->GetComm()->GetRowComm()->AllReduce(hasAnyBndCoeff,
                                                    LibUtilities::ReduceMax);
        m_hasAnyBndCoeff = hasAnyBndCoeff > 0;

        if (!m_hasAnyBndCoeff)
        {
            return;
        }

        // Collect the compact Dirichlet-to-full-boundary coefficient offset
        // per matching region (needed by v_UpdateBndCoeffs()).
        m_dirCoeffOffsets.clear();
        {
            size_t bndcnt = 0;
            for (unsigned int i = 0; i < bregions.size(); ++i)
            {
                if (std::any_of(isDirichletByRegion[i].begin(),
                                isDirichletByRegion[i].end(),
                                [](bool b) { return b; }))
                {
                    m_dirCoeffOffsets.push_back(bndcnt);
                    bndcnt += numBcExpCoeffs[i];
                }
            }
        }

        auto domainBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                this->m_expansionList);

        // Stride of the compact per-component coefficient layout, computed
        // exactly the way Field::ToArray() computes it (real data only, no
        // per-block padding) so the flatten/unflatten below is layout
        // compatible with the local coefficient numbering the assembly maps
        // use.
        m_flatCompSize = 0;
        for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
        {
            m_flatCompSize += domainBlocks[blk].GetNumData() *
                              domainBlocks[blk].GetNumElements();
        }

        m_rowComm    = session->GetComm()->GetRowComm();
        m_isParallel = m_rowComm->GetSize() > 1;
        m_dirComm.resize(nComp);

        m_nParDirBndSignSize.resize(nComp, 0);
        m_localDirSize.resize(nComp, 0);
        m_parDirOffsets.assign(domainBlocks.size(),
                               std::vector<size_t>(nComp, 0));
        m_parDirCounts.assign(domainBlocks.size(),
                              std::vector<size_t>(nComp, 0));

        // Temporary accumulators for the local-dir-dofs bookkeeping (only
        // ever populated for meshes with duplicated local Dirichlet dofs,
        // e.g. collapsed-coordinate or periodic-adjacent elements). These
        // stay local to the constructor; only the non-empty (blk1,blk0)
        // pairs are kept in m_locBlockPairs afterwards.
        std::vector<std::vector<int>> parDirBlockByBlk(domainBlocks.size());
        std::vector<std::vector<std::vector<size_t>>> locid0BlockByPair(
            domainBlocks.size(),
            std::vector<std::vector<size_t>>(domainBlocks.size()));
        std::vector<std::vector<std::vector<size_t>>> locid1BlockByPair(
            domainBlocks.size(),
            std::vector<std::vector<size_t>>(domainBlocks.size()));
        std::vector<std::vector<std::vector<TData>>> locsignBlockByPair(
            domainBlocks.size(),
            std::vector<std::vector<TData>>(domainBlocks.size()));
        std::vector<std::vector<std::vector<size_t>>> locOffsetsTmp(
            domainBlocks.size(),
            std::vector<std::vector<size_t>>(domainBlocks.size(),
                                             std::vector<size_t>(nComp, 0)));
        std::vector<std::vector<std::vector<size_t>>> locCountsTmp(
            domainBlocks.size(),
            std::vector<std::vector<size_t>>(domainBlocks.size(),
                                             std::vector<size_t>(nComp, 0)));

        auto maps = BuildCGBndCondCoeffMaps<ExecSpace, TData>(
            this->m_expansionList, components, bregions, isDirichletByRegion,
            numBcExpCoeffs, m_numBndCoeffCompSize,
            [&](unsigned int nc,
                const MultiRegions::AssemblyMapCGSharedPtr &assmbMap,
                const auto &blocks, const auto &blkBound) {
                // Build this component's device-resident universal Dirichlet
                // assembly. Each component gets its own communicator: with
                // mixed boundary conditions, different components can have
                // Dirichlet dofs in different places, so their cross-rank
                // topologies are not interchangeable.
                SetUpUniversalDirComm(nc, assmbMap);

                auto &parallelDirBndSign = assmbMap->GetParallelDirBndSign();

                // local dir dofs.
                m_localDirSize[nc] = assmbMap->GetCopyLocalDirDofs().size();
                std::vector<std::tuple<size_t, size_t, TData>> locReordered(
                    m_localDirSize[nc]);
                if (m_localDirSize[nc] > 0)
                {
                    size_t cnt = 0;
                    for (auto &it : assmbMap->GetCopyLocalDirDofs())
                    {
                        locReordered[cnt] = it;
                        cnt++;
                    }
                    std::sort(
                        std::begin(locReordered), std::end(locReordered),
                        [](std::tuple<size_t, size_t, TData> const &t1,
                           std::tuple<size_t, size_t, TData> const &t2) {
                            return std::tie(std::get<1>(t1), std::get<0>(t1)) <
                                   std::tie(std::get<1>(t2), std::get<0>(t2));
                        });

                    std::vector<size_t> nLocCoeffBlock(blocks.size(), 0);
                    std::vector<std::vector<size_t>> locid0Block(blocks.size());
                    std::vector<std::vector<size_t>> locid1Block(blocks.size());
                    std::vector<std::vector<TData>> locsignBlock(blocks.size());
                    unsigned int blk0 = 0, blk1 = 0;
                    size_t offset0 = 0, offset1 = 0;
                    unsigned int i = 0;
                    while (blk1 < blocks.size())
                    {
                        if (i == m_localDirSize[nc] ||
                            std::get<1>(locReordered[i]) >= blkBound[blk1])
                        {
                            for (size_t b0 = 0; b0 < blocks.size(); ++b0)
                            {
                                const size_t compOffset =
                                    locid0BlockByPair[blk1][b0].size();
                                const size_t compCount = locid0Block[b0].size();
                                locOffsetsTmp[blk1][b0][nc] = compOffset;
                                locCountsTmp[blk1][b0][nc]  = compCount;

                                auto &dst0 = locid0BlockByPair[blk1][b0];
                                auto &dst1 = locid1BlockByPair[blk1][b0];
                                auto &dsts = locsignBlockByPair[blk1][b0];
                                dst0.insert(dst0.end(), locid0Block[b0].begin(),
                                            locid0Block[b0].end());
                                dst1.insert(dst1.end(), locid1Block[b0].begin(),
                                            locid1Block[b0].end());
                                dsts.insert(dsts.end(),
                                            locsignBlock[b0].begin(),
                                            locsignBlock[b0].end());
                                locid0Block[b0].clear();
                                locid1Block[b0].clear();
                                locsignBlock[b0].clear();
                                nLocCoeffBlock[b0] = 0;
                            }
                            offset1 = blkBound[blk1];
                            blk1++;
                        }
                        else if (std::get<0>(locReordered[i]) >= blkBound[blk0])
                        {
                            offset0 = blkBound[blk0];
                            blk0++;
                        }
                        else
                        {
                            locid0Block[blk0].push_back(
                                std::get<0>(locReordered[i]) - offset0);
                            locid1Block[blk0].push_back(
                                std::get<1>(locReordered[i]) - offset1);
                            locsignBlock[blk0].push_back(
                                std::get<2>(locReordered[i]));
                            nLocCoeffBlock[blk0]++;
                            offset0 = 0;
                            blk0    = 0;
                            i++;
                        }
                    }
                }

                // Parallel dir sign.
                m_nParDirBndSignSize[nc] = parallelDirBndSign.size();
                std::vector<int> parDirBndSignReordered(
                    m_nParDirBndSignSize[nc]);
                if (m_nParDirBndSignSize[nc] > 0)
                {
                    size_t cnt = 0;
                    for (auto &it : parallelDirBndSign)
                    {
                        parDirBndSignReordered[cnt] = it;
                        cnt++;
                    }
                    std::sort(std::begin(parDirBndSignReordered),
                              std::end(parDirBndSignReordered),
                              [](const size_t t1, const size_t t2) {
                                  return t1 < t2;
                              });
                }

                if (m_nParDirBndSignSize[nc] > 0)
                {
                    std::vector<int> parDirBndSignBlock;
                    unsigned int i = 0, blk = 0, offset = 0;
                    while (blk < blocks.size())
                    {
                        if (i == m_nParDirBndSignSize[nc] ||
                            parDirBndSignReordered[i] >= blkBound[blk])
                        {
                            const size_t compOffset =
                                parDirBlockByBlk[blk].size();
                            const size_t compCount = parDirBndSignBlock.size();
                            m_parDirOffsets[blk][nc] = compOffset;
                            m_parDirCounts[blk][nc]  = compCount;

                            auto &dst = parDirBlockByBlk[blk];
                            dst.insert(dst.end(), parDirBndSignBlock.begin(),
                                       parDirBndSignBlock.end());
                            parDirBndSignBlock.clear();
                            offset = blkBound[blk];
                            blk++;
                        }
                        else
                        {
                            parDirBndSignBlock.push_back(
                                parDirBndSignReordered[i] - offset);
                            i++;
                        }
                    }
                }
            });

        m_anySignChange = maps.anySignChange;
        m_signChange    = std::move(maps.signChange);
        m_map           = std::move(maps.map);
        m_sign          = std::move(maps.sign);
        m_bndCoeff      = std::move(maps.bndCoeff);
        m_bndCoeffSrc   = std::move(maps.bndCoeffSrc);
        m_compOffsets   = std::move(maps.compOffsets);
        m_compCounts    = std::move(maps.compCounts);
        // Populated in full by v_UpdateBndCoeffs() every call, so it does
        // not need a zeroed starting value here (only ever written via
        // WriteOnly device access, never read before being written).
        m_compactBndCoeff =
            LibUtilities::MemoryRegion<TData>(nComp * m_numBndCoeffCompSize);

        // Staging for the device-resident universal Dirichlet assembly.
        // Fully overwritten by the flatten at the start of every assembly,
        // but initialized here so the first ReadWrite access is valid.
        m_flatBuf = LibUtilities::MemoryRegion<TData>(nComp * m_flatCompSize);
        m_flatBuf.template Initialize<MemSpace>(0.0);

        m_maxLocalDup  = 0;
        m_maxSREntries = 0;
        for (unsigned int nc = 0; nc < nComp; ++nc)
        {
            m_maxLocalDup  = std::max(m_maxLocalDup, m_dirComm[nc].nLocalDup);
            m_maxSREntries = std::max(m_maxSREntries, m_dirComm[nc].nSREntries);
        }
        if (m_maxLocalDup > 0)
        {
            m_tmpBuf = LibUtilities::MemoryRegion<TData>(m_maxLocalDup);
            m_tmpBuf.template Initialize<MemSpace>(0.0);
        }

        m_parDirBndSign.reserve(domainBlocks.size());
        for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
        {
            m_parDirBndSign.push_back(
                LibUtilities::MemoryRegion<int>::template FromVector<MemSpace,
                                                                     int>(
                    parDirBlockByBlk[blk]));
        }

        this->UpdateBndCoeffs(static_cast<TData>(0.0));

        // Keep only the (blk1,blk0) execution-block pairs that actually have
        // duplicated local Dirichlet dofs to copy.
        m_locBlockPairs.clear();
        for (unsigned int blk1 = 0; blk1 < domainBlocks.size(); ++blk1)
        {
            for (unsigned int blk0 = 0; blk0 < domainBlocks.size(); ++blk0)
            {
                if (locid0BlockByPair[blk1][blk0].empty())
                {
                    continue;
                }

                LocalDirBlockPair pair;
                pair.blk1        = blk1;
                pair.blk0        = blk0;
                pair.compOffsets = locOffsetsTmp[blk1][blk0];
                pair.compCounts  = locCountsTmp[blk1][blk0];
                pair.locid0 =
                    LibUtilities::MemoryRegion<size_t>::template FromVector<
                        MemSpace, size_t>(locid0BlockByPair[blk1][blk0]);
                pair.locid1 =
                    LibUtilities::MemoryRegion<size_t>::template FromVector<
                        MemSpace, size_t>(locid1BlockByPair[blk1][blk0]);
                pair.locsign =
                    LibUtilities::MemoryRegion<TData>::template FromVector<
                        MemSpace, TData>(locsignBlockByPair[blk1][blk0]);
                m_locBlockPairs.push_back(std::move(pair));
            }
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<DirBndCondOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    bool m_anySignChange         = false;
    size_t m_numBndCoeffCompSize = 0;
    // Whether any rank (not just this one) has local Dirichlet boundary
    // coefficients. Guards the collective calls in v_Apply() (see
    // constructor comment); the purely-local m_numBndCoeffCompSize remains
    // safe to use elsewhere since it only ever skips genuinely empty local
    // work.
    bool m_hasAnyBndCoeff = false;
    std::vector<bool> m_signChange;

    // ---------------------------------------------------------------------
    // Device-resident universal Dirichlet assembly.
    //
    // Replaces AssemblyMap::UniversalAbsMaxBnd() -- a host-only
    // Gs::Gather(..., Gs::gs_amax, m_dirBndGsh) -- and, with it, the D2H
    // copy of the whole coefficient array, the host MPI call per component,
    // and the H2D copy back that v_Apply() used to perform on every call.
    //
    // One instance per component: with mixed boundary conditions different
    // components can have Dirichlet dofs in different places, so their
    // communication topologies are not interchangeable (this is exactly why
    // the legacy code needed one UniversalAbsMaxBnd() call per component).
    struct DirUniversalComm
    {
        // Duplicated local copies of the same universal Dirichlet dof:
        // locIdx[k] is a local coefficient index and repIdx[k] is the index
        // chosen to represent its universal id. Resolving these is required
        // even on a single rank -- it is the local half of what the legacy
        // Gs gather did, and it has to happen before the copy-local-dir
        // processing further down v_Apply().
        LibUtilities::MemoryRegion<size_t> locIdx;
        LibUtilities::MemoryRegion<size_t> repIdx;
        size_t nLocalDup = 0;

        // Cross-rank exchange. Only built when running on more than one
        // rank; nSREntries is 0 when this rank shares no Dirichlet dof with
        // any neighbour.
        std::unique_ptr<MultiRegions::AssemblyCommCG<TData>> comm;
        LibUtilities::MemoryRegion<size_t> srEntries;
        LibUtilities::MemoryRegion<TData> sendBuffer;
        LibUtilities::MemoryRegion<TData> recvBuffer;
        size_t nSREntries = 0;
    };
    std::vector<DirUniversalComm> m_dirComm;

    LibUtilities::CommSharedPtr m_rowComm;
    bool m_isParallel = false;

    // Number of local coefficients per component, i.e. the stride of the
    // compact per-component layout that Field::ToArray() would produce.
    size_t m_flatCompSize = 0;
    // Compact nComp * m_flatCompSize staging buffer that the field blocks
    // are flattened into (device-to-device) for the assembly, and unflattened
    // from afterwards.
    LibUtilities::MemoryRegion<TData> m_flatBuf;
    // Scratch for the gather half of the local duplicate reduce/broadcast.
    LibUtilities::MemoryRegion<TData> m_tmpBuf;
    size_t m_maxLocalDup = 0;
    // Max over components of nSREntries -- together with m_maxLocalDup, lets
    // UniversalAbsMaxDirBnd() early-out when no component has anything to
    // reconcile (see its use there).
    size_t m_maxSREntries = 0;

    std::vector<MultiRegions::Field<TData, FieldState::Phys>> m_wsp_phys;
    std::vector<MultiRegions::Field<TData, FieldState::Coeff>> m_wsp_coeffs;
    std::vector<std::shared_ptr<ExpressionOp<TData>>> m_expressionOps;
    std::vector<std::shared_ptr<FwdTransBCOp<TData>>> m_fwdTransBCOps;
    std::vector<size_t> m_dirCoeffOffsets;
    std::vector<size_t> m_dirNumCoeffs;

    std::vector<LibUtilities::MemoryRegion<size_t>> m_map;
    std::vector<LibUtilities::MemoryRegion<TData>> m_sign;
    std::vector<LibUtilities::MemoryRegion<TData>> m_bndCoeff;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_bndCoeffSrc;
    LibUtilities::MemoryRegion<TData> m_compactBndCoeff;
    std::vector<std::vector<size_t>> m_compOffsets;
    std::vector<std::vector<size_t>> m_compCounts;

    std::vector<size_t> m_nParDirBndSignSize;
    std::vector<size_t> m_localDirSize;

    std::vector<LibUtilities::MemoryRegion<int>> m_parDirBndSign;
    std::vector<std::vector<size_t>> m_parDirOffsets;
    std::vector<std::vector<size_t>> m_parDirCounts;

    // Duplicated local Dirichlet dofs are only present with e.g. collapsed-
    // coordinate or periodic-adjacent elements, so most meshes never
    // populate this at all. Store only the (blk1,blk0) execution-block
    // pairs that actually have entries rather than a dense
    // nBlocks*nBlocks*nComp matrix.
    struct LocalDirBlockPair
    {
        unsigned int blk1;
        unsigned int blk0;
        LibUtilities::MemoryRegion<size_t> locid0;
        LibUtilities::MemoryRegion<size_t> locid1;
        LibUtilities::MemoryRegion<TData> locsign;
        std::vector<size_t> compOffsets;
        std::vector<size_t> compCounts;
    };
    std::vector<LocalDirBlockPair> m_locBlockPairs;

    void v_UpdateBndCoeffs(const TData &time) override
    {
        // Guard on global (not local) participation: even when this rank
        // has no local Dirichlet dofs, BuildCGBndCondCoeffMaps() still
        // allocated a (zero-sized) m_bndCoeff[blk]/m_map[blk]/etc. entry for
        // every domain block on this rank, since some other rank may have
        // local dofs there. v_Apply() reads m_bndCoeff[blk] unconditionally,
        // so it must be brought to a valid (if trivially empty) state below
        // -- via the WriteOnly GetPtr in the loop over m_bndCoeff -- rather
        // than skipped, or MemoryRegion's read will find it uninitialized.
        if (!m_hasAnyBndCoeff)
        {
            return;
        }

        const unsigned int nComp = this->m_components.size();

        // Get a device pointer to the compact array: entirely on-device, no
        // host round-trip. No zeroing needed first -- m_dirCoeffOffsets/
        // m_dirNumCoeffs tile [0, m_numBndCoeffCompSize) exactly, and the
        // copy loop below writes every position for every component
        // (regardless of whether that component actually has a Dirichlet
        // condition there -- non-matching components get zero from
        // ExpressionOp's masked-out, zeroed wsp_phys, not a skipped write),
        // so every element gets overwritten unconditionally on every call.
        auto compactPtr =
            m_compactBndCoeff.template GetPtr<MemSpace, WriteOnly>();

        for (unsigned int iDir = 0; iDir < m_expressionOps.size(); ++iDir)
        {
            const size_t bndOffset = m_dirCoeffOffsets[iDir];

            // Zero wsp_phys and wsp_coeff
            Math::zero<ExecSpace>(m_wsp_phys[iDir]);
            Math::zero<ExecSpace>(m_wsp_coeffs[iDir]);

            // Update time
            m_expressionOps[iDir]->SetTime(time);

            // Evaluate Dirichlet expression and project onto boundary
            m_expressionOps[iDir]->Apply(m_wsp_phys[iDir], m_wsp_phys[iDir]);
            m_fwdTransBCOps[iDir]->Apply(m_wsp_phys[iDir], m_wsp_coeffs[iDir]);

            // Copy m_wsp_coeffs[iDir] into the compact array, component by
            // component, block by block (mirrors Field::ToVector()'s own
            // block-concatenation but writes straight into device memory
            // instead of a host std::vector).
            size_t regionOffset = 0;
            for (unsigned int wblk = 0;
                 wblk < m_wsp_coeffs[iDir].GetBlocks().size(); ++wblk)
            {
                auto &wspBlk     = m_wsp_coeffs[iDir].GetBlocks()[wblk];
                auto srcPtr      = wspBlk.template GetPtr<MemSpace, ReadOnly>();
                const auto nSize = wspBlk.CompSize();
                const auto blockLen =
                    wspBlk.GetNumElements() * wspBlk.GetNumData();

                for (unsigned int nc = 0; nc < nComp; ++nc)
                {
                    Math::copyKernel<ExecSpace>(blockLen, srcPtr + nc * nSize,
                                                compactPtr +
                                                    nc * m_numBndCoeffCompSize +
                                                    bndOffset + regionOffset,
                                                0);
                }
                regionOffset += blockLen;
            }

            ASSERTL1(regionOffset == m_dirNumCoeffs[iDir],
                     "Unexpected boundary coefficient vector size.");
        }

        for (unsigned int blk = 0; blk < m_bndCoeff.size(); ++blk)
        {
            auto idxPtr =
                m_bndCoeffSrc[blk].template GetPtr<MemSpace, ReadOnly>();
            auto dstPtr =
                m_bndCoeff[blk].template GetPtr<MemSpace, WriteOnly>();

            CGBndCondGatherKernel<ExecSpace>(m_bndCoeffSrc[blk].size(), idxPtr,
                                             compactPtr, dstPtr, 0);
        }
    }

    size_t v_GetNumBndDofs() const override
    {
        size_t total = 0;
        for (auto &counts : m_compCounts)
        {
            for (auto count : counts)
            {
                total += count;
            }
        }
        return total;
    }

    // Build one component's device-resident universal Dirichlet assembly.
    //
    // Collective over the row communicator when running in parallel (the
    // AssemblyCommCG constructor performs a ring discovery in which every
    // rank must take part), so this is only ever reached from the
    // constructor, under the same global m_hasAnyBndCoeff guard as the rest
    // of the collective setup.
    void SetUpUniversalDirComm(
        unsigned int nc, const MultiRegions::AssemblyMapCGSharedPtr &assmbMap)
    {
        auto &dc = m_dirComm[nc];

        // The exact per-local-coefficient universal id mask that the legacy
        // m_dirBndGsh handle is built from.
        //
        // Deliberately *not* re-derived from GetGlobalToUniversalMap() and
        // GetNumGlobalDirBndCoeffs(): that would cover every Dirichlet dof,
        // whereas this mask covers only those that need reconciling across
        // partitions (a dof no rank could fill from its own boundary
        // expansion). Only the narrow set has had its local sign conventions
        // aligned -- GetParallelDirBndSign(), which drives
        // FlipParallelDirBndSign() around the assembly, is populated from
        // this very mask -- and an absolute-maximum combine over dofs whose
        // signs still disagree between ranks would select an arbitrary sign.
        const Array<OneD, long> &paraDirBnd = assmbMap->GetParaDirBnd();

        ASSERTL1(static_cast<size_t>(paraDirBnd.size()) == m_flatCompSize,
                 "Dirichlet universal id mask size does not match the local "
                 "coefficient count.");

        // Group local coefficients by universal id. A group's representative
        // is its *last* local index, matching AssemblyCommCG's own choice in
        // BuildSendRecvMaps() (m_uid_to_index[uid] = i over ascending i), so
        // that the local indices it later hands back through GetSREntries()
        // are exactly these representatives.
        std::unordered_map<long, size_t> repOfUid, countOfUid;
        for (size_t i = 0; i < static_cast<size_t>(paraDirBnd.size()); ++i)
        {
            if (paraDirBnd[i] == 0)
            {
                continue;
            }
            repOfUid[paraDirBnd[i]] = i;
            countOfUid[paraDirBnd[i]]++;
        }

        std::vector<size_t> locIdx, repIdx;
        for (size_t i = 0; i < static_cast<size_t>(paraDirBnd.size()); ++i)
        {
            const long uid = paraDirBnd[i];
            if (uid == 0 || countOfUid[uid] < 2 || repOfUid[uid] == i)
            {
                continue;
            }
            locIdx.push_back(i);
            repIdx.push_back(repOfUid[uid]);
        }

        dc.nLocalDup = locIdx.size();
        if (dc.nLocalDup > 0)
        {
            dc.locIdx = LibUtilities::MemoryRegion<size_t>::template FromVector<
                MemSpace, size_t>(locIdx);
            dc.repIdx = LibUtilities::MemoryRegion<size_t>::template FromVector<
                MemSpace, size_t>(repIdx);
        }

        if (!m_isParallel)
        {
            return;
        }

        // Constructed on every rank, including ranks whose own mask is
        // entirely zero: the ring discovery inside is collective.
        dc.comm = std::make_unique<MultiRegions::AssemblyCommCG<TData>>(
            m_rowComm, paraDirBnd);

        const std::vector<size_t> &sr = dc.comm->GetSREntries();
        dc.nSREntries                 = sr.size();

#ifndef NDEBUG
        // The pack/unpack steps index the flattened coefficient array
        // directly with these entries, which is only correct if each one is
        // the representative of its universal id group.
        for (auto idx : sr)
        {
            ASSERTL1(idx < static_cast<size_t>(paraDirBnd.size()) &&
                         paraDirBnd[idx] != 0 &&
                         repOfUid[paraDirBnd[idx]] == idx,
                     "AssemblyCommCG send/recv entry is not the "
                     "representative local index for its universal id.");
        }
#endif

        // No Dirichlet dof shared with any neighbour: nothing to exchange.
        // Safe to skip unilaterally -- the shared-id discovery is symmetric,
        // so no other rank is expecting a message from this one.
        if (dc.nSREntries == 0)
        {
            return;
        }

        dc.srEntries =
            LibUtilities::MemoryRegion<size_t>::template FromVector<MemSpace,
                                                                    size_t>(sr);
        dc.sendBuffer =
            LibUtilities::MemoryRegion<TData>(dc.nSREntries, eHostPinned);
        dc.recvBuffer =
            LibUtilities::MemoryRegion<TData>(dc.nSREntries, eHostPinned);

        // Persistent requests are bound to the raw buffer pointers here, so
        // they must be the pointers MPI will actually read from and write to
        // for the lifetime of this operator: device pointers when the
        // communicator is GPU aware, pinned host pointers otherwise.
        TData *sendPtr, *recvPtr;
        if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
            m_rowComm->IsGPUAware())
        {
            dc.sendBuffer.template Initialize<NektarSpaces::DeviceSpace>(0.0);
            dc.recvBuffer.template Initialize<NektarSpaces::DeviceSpace>(0.0);
            sendPtr =
                (TData *)dc.sendBuffer
                    .template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
            recvPtr =
                (TData *)dc.recvBuffer
                    .template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
        }
        else
        {
            dc.sendBuffer.template Initialize<NektarSpaces::HostSpace>(0.0);
            dc.recvBuffer.template Initialize<NektarSpaces::HostSpace>(0.0);
            sendPtr = (TData *)dc.sendBuffer
                          .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            recvPtr = (TData *)dc.recvBuffer
                          .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        }

        dc.comm->InitSendRecvComms(dc.nSREntries, sendPtr, recvPtr, 1);
    }

    // Move data between the per-block field storage and the compact
    // per-component staging array, which is laid out exactly as
    // Field::ToArray() would lay it out -- but device to device, with no
    // host staging.
    void FlattenBlocks(MultiRegions::Field<TData, FieldState::Coeff> &inout,
                       TData *flatPtr, const bool toFlat)
    {
        const unsigned nComp = inout.GetNumComponents();
        size_t blockOffset   = 0;

        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            auto &blkRef     = inout.GetBlocks()[blk];
            auto blkPtr      = blkRef.template GetPtr<MemSpace, ReadWrite>();
            const auto nSize = blkRef.CompSize();
            const auto len   = blkRef.GetNumElements() * blkRef.GetNumData();

            for (unsigned nc = 0; nc < nComp; ++nc)
            {
                TData *const flat = flatPtr + nc * m_flatCompSize + blockOffset;
                TData *const blkc = blkPtr + nc * nSize;

                if (toFlat)
                {
                    Math::copyKernel<ExecSpace>(len, blkc, flat, 0);
                }
                else
                {
                    Math::copyKernel<ExecSpace>(len, flat, blkc, 0);
                }
            }

            blockOffset += len;
        }

        ASSERTL1(blockOffset == m_flatCompSize,
                 "Flattened block extent does not match the local coefficient "
                 "count.");
    }

    // Device-resident equivalent of the per-component
    // AssemblyMap::UniversalAbsMaxBnd() calls this operator used to make: an
    // absolute-maximum gather-scatter over the Dirichlet coefficients that
    // straddle a partition boundary, plus their duplicated local copies,
    // using GPU-aware MPI where the communicator supports it.
    void UniversalAbsMaxDirBnd(
        MultiRegions::Field<TData, FieldState::Coeff> &inout)
    {
        const unsigned nComp = inout.GetNumComponents();

        ASSERTL1(nComp == m_dirComm.size(),
                 "Field component count does not match the number of "
                 "Dirichlet assembly maps.");

        // Everything above this point ran on the per-block streams; the
        // flatten reads every block and the MPI step needs those values
        // settled. This sync stays unconditional (not folded into the
        // early-out below) since v_Apply()'s later cross-block
        // local-dir-dof copy (m_locBlockPairs) also relies on it: it reads
        // one block's data while writing another's, on that other block's
        // stream, so it needs every block's writes visible regardless of
        // whether this function itself has anything to reconcile.
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
            {
                nekStreamSynchronize(blk + 1);
            }
        }

        // Nothing to reconcile: no component has a locally duplicated
        // Dirichlet dof, and no component has a dof shared with another
        // rank (m_maxLocalDup/m_maxSREntries are maxes over components,
        // fixed by mesh topology at construction time). Skip the
        // full-domain flatten/unflatten and all the gather/unpack kernels
        // below -- covers both serial runs and parallel runs where this
        // rank's local partition happens to touch no shared Dirichlet dof.
        if (m_maxLocalDup == 0 && m_maxSREntries == 0)
        {
            return;
        }

        auto flatPtr  = m_flatBuf.template GetPtr<MemSpace, ReadWrite>();
        TData *tmpPtr = m_maxLocalDup > 0
                            ? m_tmpBuf.template GetPtr<MemSpace, ReadWrite>()
                            : nullptr;

        FlattenBlocks(inout, flatPtr, true);

        // 1. Fold duplicated local copies of a shared Dirichlet dof into
        //    that dof's representative. Needed on a single rank too: this is
        //    the local half of what Gs::Gather() used to do, and it has to
        //    happen before the copy-local-dir processing in v_Apply().
        for (unsigned nc = 0; nc < nComp; ++nc)
        {
            auto &dc = m_dirComm[nc];
            if (dc.nLocalDup == 0)
            {
                continue;
            }

            CGBndCondGatherKernel<ExecSpace>(
                dc.nLocalDup, dc.locIdx.template GetPtr<MemSpace, ReadOnly>(),
                flatPtr + nc * m_flatCompSize, tmpPtr, 0);
            CGBndCondUnpackAbsMaxKernel<ExecSpace>(
                dc.nLocalDup, dc.repIdx.template GetPtr<MemSpace, ReadOnly>(),
                tmpPtr, flatPtr + nc * m_flatCompSize, 0);
        }

        if (m_isParallel)
        {
            // 2. Pack each component's representative values and post the
            //    exchange. All components are packed and posted before any
            //    is waited on, so the per-component exchanges overlap rather
            //    than running as nComp serialized MPI rounds.
            for (unsigned nc = 0; nc < nComp; ++nc)
            {
                auto &dc = m_dirComm[nc];
                if (dc.nSREntries == 0)
                {
                    continue;
                }

                CGBndCondGatherKernel<ExecSpace>(
                    dc.nSREntries,
                    dc.srEntries.template GetPtr<MemSpace, ReadOnly>(),
                    flatPtr + nc * m_flatCompSize,
                    dc.sendBuffer.template GetPtr<MemSpace, WriteOnly>(), 0);
            }

            // Stage the send buffers where MPI will read them from.
            if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
                !m_rowComm->IsGPUAware())
            {
                // Without GPU-aware MPI the data must be copied down to the
                // pinned host buffer first. Only the send buffer moves, not
                // the field.
                for (unsigned nc = 0; nc < nComp; ++nc)
                {
                    if (m_dirComm[nc].nSREntries > 0)
                    {
                        m_dirComm[nc]
                            .sendBuffer.template GetPtr<NektarSpaces::HostSpace,
                                                        ReadOnly>();
                    }
                }
            }
            else if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
                     m_rowComm->IsGPUAware())
            {
                // MPI reads the device buffer directly, so just make sure
                // the packing kernels have completed.
                nekStreamSynchronize(0);
            }

            for (unsigned nc = 0; nc < nComp; ++nc)
            {
                if (m_dirComm[nc].nSREntries > 0)
                {
                    m_dirComm[nc].comm->BeginComm();
                }
            }

            // Prime the receive buffers in the space MPI will deliver into,
            // so the subsequent read in the operator's memory space picks up
            // the new data (and stages it up from the host if needed).
            for (unsigned nc = 0; nc < nComp; ++nc)
            {
                auto &dc = m_dirComm[nc];
                if (dc.nSREntries == 0)
                {
                    continue;
                }

                if (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace> &&
                    m_rowComm->IsGPUAware())
                {
                    dc.recvBuffer.template GetPtr<NektarSpaces::DeviceSpace,
                                                  WriteOnly>();
                }
                else
                {
                    dc.recvBuffer
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                }
            }

            for (unsigned nc = 0; nc < nComp; ++nc)
            {
                if (m_dirComm[nc].nSREntries > 0)
                {
                    m_dirComm[nc].comm->EndComm();
                }
            }

            // 3. Fold what the neighbours sent into the representatives. A
            //    dof shared by three or more ranks contributes one entry per
            //    neighbour, so the same index can appear more than once and
            //    the combine has to be a real reduction.
            for (unsigned nc = 0; nc < nComp; ++nc)
            {
                auto &dc = m_dirComm[nc];
                if (dc.nSREntries == 0)
                {
                    continue;
                }

                CGBndCondUnpackAbsMaxKernel<ExecSpace>(
                    dc.nSREntries,
                    dc.srEntries.template GetPtr<MemSpace, ReadOnly>(),
                    dc.recvBuffer.template GetPtr<MemSpace, ReadOnly>(),
                    flatPtr + nc * m_flatCompSize, 0);
            }
        }

        // 4. Broadcast each representative's resolved value back over its
        //    duplicated local copies, so every local copy of a shared
        //    Dirichlet dof ends up holding the same value -- which is what
        //    the legacy gather-scatter left behind.
        for (unsigned nc = 0; nc < nComp; ++nc)
        {
            auto &dc = m_dirComm[nc];
            if (dc.nLocalDup == 0)
            {
                continue;
            }

            CGBndCondGatherKernel<ExecSpace>(
                dc.nLocalDup, dc.repIdx.template GetPtr<MemSpace, ReadOnly>(),
                flatPtr + nc * m_flatCompSize, tmpPtr, 0);
            // outptr[mapPtr[i]] = inptr[i]
            DirBndCondKernel<ExecSpace>(
                dc.nLocalDup, dc.locIdx.template GetPtr<MemSpace, ReadOnly>(),
                tmpPtr, flatPtr + nc * m_flatCompSize, 0);
        }

        FlattenBlocks(inout, flatPtr, false);
    }

    // Flip the sign of local Dirichlet coefficients that are shared with a
    // Dirichlet boundary on another partition. Called once before and once
    // after the universal max-magnitude gather in v_Apply(), mirroring
    // ContField::v_ImposeDirichletConditions().
    void FlipParallelDirBndSign(
        MultiRegions::Field<TData, FieldState::Coeff> &inout)
    {
        for (unsigned nc = 0; nc < inout.GetNumComponents(); ++nc)
        {
            if (m_nParDirBndSignSize[nc] == 0)
            {
                continue;
            }

            for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
            {
                const unsigned int streamID = blk + 1;

                auto nParDirBndSignBlock = m_parDirCounts[blk][nc];
                if (nParDirBndSignBlock == 0)
                {
                    continue;
                }

                auto inoutptr =
                    inout.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(
                        streamID);
                auto parDirBndSignPtr =
                    m_parDirBndSign[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID) +
                    m_parDirOffsets[blk][nc];
                auto blksize = inout.GetBlocks()[blk].CompSize();
                ParallelDirBndSignKernel<ExecSpace>(
                    nParDirBndSignBlock, parDirBndSignPtr,
                    inoutptr + nc * blksize, streamID);
            }
        }
    }

    void v_Apply(MultiRegions::Field<TData, FieldState::Coeff> &inout) override
    {
        // Return if no Dirichlet boundary condition on any rank -- must
        // match the constructor's global check, since the universal
        // assembly below exchanges with every neighbouring rank that shares
        // a Dirichlet dof, regardless of whether this rank has any local
        // Dirichlet dofs of its own.
        if (!m_hasAnyBndCoeff)
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

            // if block is interlaced deInterleave block since currently mapping
            // set up assuming serial alignment
            ReshapeStorage<ExecSpace>(
                1u, inoutWidth,
                inoutBlk.GetNumElementsWithPadding() * inout.GetNumComponents(),
                inoutBlk.GetNumData(), inoutPtr, streamID);

            // Add Dirichlet boundary conditions.
            for (unsigned nc = 0; nc < inout.GetNumComponents(); ++nc)
            {
                auto nbndCoeffBlk = m_compCounts[blk][nc];
                if (nbndCoeffBlk == 0)
                {
                    continue;
                }

                const size_t offset = m_compOffsets[blk][nc];
                auto mapPtr         = mapPtrBlock + offset;
                auto bndcoeffPtr    = bndcoeffPtrBlock + offset;
                const TData *signPtr =
                    m_signChange[nc] ? signPtrBlock + offset : nullptr;

                if (m_signChange[nc])
                {
                    DirBndCondKernel<ExecSpace>(
                        nbndCoeffBlk, signPtr, mapPtr, bndcoeffPtr,
                        inoutPtr + nc * blksize, streamID);
                }
                else
                {
                    DirBndCondKernel<ExecSpace>(
                        nbndCoeffBlk, mapPtr, bndcoeffPtr,
                        inoutPtr + nc * blksize, streamID);
                }
            }
        }

        FlipParallelDirBndSign(inout);

        // Universal max-magnitude assembly of the Dirichlet coefficients.
        // Fully device resident: no ToArray()/CopyArray() round-trip of the
        // coefficient array and no host-side Gs gather. This is also
        // required in serial, to resolve duplicated local Dirichlet boundary
        // coefficients before copy-local-dir processing, mirroring
        // ContField::v_ImposeDirichletConditions(). Each component is
        // assembled with its own communication topology: with mixed boundary
        // conditions, different components can have Dirichlet dofs in
        // different places, so their topologies are not interchangeable.
        UniversalAbsMaxDirBnd(inout);

        // Flip the sign back now that the max-magnitude gather has resolved
        // the duplicated Dirichlet coefficients.
        FlipParallelDirBndSign(inout);

        // Copy duplicated local Dirichlet dofs. Most meshes have none at
        // all, in which case m_locBlockPairs is empty and this is a no-op.
        for (auto &pair : m_locBlockPairs)
        {
            const unsigned int blk1 = pair.blk1;
            const unsigned int blk0 = pair.blk0;
            auto blksize1           = inout.GetBlocks()[blk1].CompSize();
            auto blksize0           = inout.GetBlocks()[blk0].CompSize();

            for (unsigned nc = 0; nc < inout.GetNumComponents(); ++nc)
            {
                auto nLocCoeffBlock = pair.compCounts[nc];
                if (nLocCoeffBlock == 0)
                {
                    continue;
                }

                const unsigned int streamID = blk0 + 1;

                auto inptr =
                    inout.GetBlocks()[blk1].template GetPtr<MemSpace, ReadOnly>(
                        streamID);
                auto outptr =
                    inout.GetBlocks()[blk0]
                        .template GetPtr<MemSpace, WriteOnly>(streamID);
                const size_t offset = pair.compOffsets[nc];
                auto locid0Ptr =
                    pair.locid0.template GetPtr<MemSpace, ReadOnly>(streamID) +
                    offset;
                auto locid1Ptr =
                    pair.locid1.template GetPtr<MemSpace, ReadOnly>(streamID) +
                    offset;
                auto locsignPtr =
                    pair.locsign.template GetPtr<MemSpace, ReadOnly>(streamID) +
                    offset;

                LocalDirBndCondKernel<ExecSpace>(
                    nLocCoeffBlock, locid0Ptr, locid1Ptr, locsignPtr,
                    inptr + nc * blksize1, outptr + nc * blksize0, streamID);
            }
        }

        // Set dependencies.
        std::vector<unsigned int> eventIDs;
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            const unsigned int eventID = blk + 1;
            eventIDs.push_back(eventID);
        }

        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            SetStreamDependencies<ExecSpace>(streamID, eventIDs);

            auto &inoutBlk = inout.GetBlocks()[blk];

            // Initialize pointers.
            auto inoutPtr =
                inoutBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto inoutWidth = inoutBlk.GetInterleaveWidth();

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(
                inoutWidth, 1u,
                inoutBlk.GetNumElementsWithPadding() * inout.GetNumComponents(),
                inoutBlk.GetNumData(), inoutPtr, streamID);
        }
    }
};

} // namespace Nektar::Operators::detail
