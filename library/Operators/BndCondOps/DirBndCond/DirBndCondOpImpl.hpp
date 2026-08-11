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

#include <MultiRegions/ContField.h>

#include "Operators/BndCondOps/DirBndCond/DirBndCondKernels.hpp"
#include "Operators/BndCondOps/DirBndCond/DirBndCondOp.hpp"
#include "Operators/BndCondOps/FwdTransBC/FwdTransBCOp.hpp"

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/Expression/ExpressionOp.hpp"
#include "Operators/Math/MathKernels.hpp"

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
        unsigned int nhomo = 1;
        session->LoadParameter("HomModesZ", nhomo, 1);
        unsigned int nComp = components.size();

        // Create boundary conditions
        const SpatialDomains::BoundaryConditions bcs(session, graph);
        const SpatialDomains::BoundaryRegionCollection &bregions =
            bcs.GetBoundaryRegions();
        const SpatialDomains::BoundaryConditionCollection &bconditions =
            bcs.GetBoundaryConditions();

        // Create counters and temporary variables for BC Fields and Operators
        std::vector<std::vector<bool>> isDirichletByRegion;
        std::vector<size_t> numBcExpCoeffs;
        SpatialDomains::BoundaryConditionShPtr bc;
        MultiRegions::ExpListSharedPtr bcExpList;

        // Loop all boundary regions to create Fields and Operators
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

            // Get BoundaryCondition(SharedPtr)
            bc = (*conditionMapIter).second;

            // Create boundary condition expansion
            bcExpList = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                session, *(it.second), graph, true, components[0], false,
                bc->GetComm(), Collections::eNoImpType);

            // Set data warehouse for device support operators
            bcExpList->SetDataWarehouse();

            // Save number of coefficients
            numBcExpCoeffs.push_back(bcExpList->GetNcoeffs());

            // Check if any component has a Dirichlet-type condition
            bool hasDirichletCondition = false;
            std::vector<bool> isDirichlet(nComp);
            for (unsigned int nc = 0; nc < nComp; nc++)
            {
                // Get BoundaryCondition and check if it is Dirichlet
                auto cndMapIter = bndCondMap->find(components[nc]);
                bc              = (*cndMapIter).second;
                bool tmp        = (bool)std::dynamic_pointer_cast<
                           SpatialDomains::DirichletBoundaryCondition>(bc);

                // Save if component has a Dirichlet condition
                isDirichlet[nc] = tmp;

                // Check if any component has a Dirichlet condition
                hasDirichletCondition = hasDirichletCondition || tmp;
                this->m_hasTimeDependentBndCoeffs =
                    this->m_hasTimeDependentBndCoeffs ||
                    (tmp && bc->IsTimeDependent());
            }

            isDirichletByRegion.push_back(isDirichlet);

            // Skip, if no Dirichletn condition in this boundary region
            if (!hasDirichletCondition)
            {
                continue;
            }

            // Create fields for evaluating BCs into
            auto blocks_phys =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    bcExpList);
            this->m_wsp_phys.push_back(
                MultiRegions::Field<TData, FieldState::Phys>(
                    "Dirichlet BC phys", blocks_phys, nComp, nhomo));
            auto blocks_coeffs =
                MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                    bcExpList);
            this->m_wsp_coeffs.push_back(
                MultiRegions::Field<TData, FieldState::Coeff>(
                    "Dirichlet BC coeff", blocks_coeffs, nComp, nhomo));

            // Compute number of boundary coefficients.
            for (unsigned int blk = 0; blk < blocks_coeffs.size(); ++blk)
            {
                const auto ncoeff = blocks_coeffs[blk].GetNumData();
                const auto nelmt  = blocks_coeffs[blk].GetNumElements();
                m_numBndCoeffCompSize += nelmt * ncoeff;
            }

            m_dirNumCoeffs.push_back(bcExpList->GetNcoeffs());

            // Initialize memory regions
            this->m_wsp_phys.back().template Initialize<MemSpace>(0.0);
            this->m_wsp_coeffs.back().template Initialize<MemSpace>(0.0);

            // Create operators for this boundary condition
            this->m_expressionOps.push_back(
                ExpressionOp<TData>::Create(bcExpList, components));
            this->m_expressionOps.back()->SetComponentMask(isDirichlet);

            this->m_fwdTransBCOps.push_back(FwdTransBCOp<TData>::Create(
                bcExpList, components, ExecSpace::name));

            // Gather equations for each field/component
            std::vector<LibUtilities::EquationSharedPtr> listOfEquations;
            for (unsigned int nc = 0; nc < nComp; nc++)
            {
                // Get map for each field
                auto cndMapIter = bndCondMap->find(components[nc]);

                // Get BoundaryCondition and extract equation
                bc = (*cndMapIter).second;
                listOfEquations.push_back(bc->GetEquation());
            }

            // Set boundary conditions for each component in this boundary
            // region
            this->m_expressionOps.back()->SetExpressions(listOfEquations);
        }

        // Return if no Dirichlet boundary coefficients. This
        // must be a global reduction rather than the purely local.
        size_t hasAnyBndCoeff = m_numBndCoeffCompSize;
        session->GetComm()->GetRowComm()->AllReduce(hasAnyBndCoeff,
                                                    LibUtilities::ReduceMax);
        m_hasAnyBndCoeff = hasAnyBndCoeff > 0;

        if (!m_hasAnyBndCoeff)
        {
            return;
        }

        // Compute block bound.
        auto domainBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                this->m_expansionList);
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

        // Collect the compact Dirichlet-to-full-boundary coefficient index map
        // for one component. Boundary coefficients themselves are dynamic and
        // are filled by UpdateBndCoeffs().
        std::vector<size_t> index(m_numBndCoeffCompSize);
        std::vector<std::vector<size_t>> dirIndexByComp(nComp);
        size_t bndcnt = 0, cnt = 0;
        m_dirCoeffOffsets.clear();
        for (unsigned int i = 0; i < bregions.size(); ++i)
        {
            // Get number of coefficients for this BC
            auto nBndExpCoeff = numBcExpCoeffs[i];

            // Process Dirichlet boundary conditions
            if (std::any_of(isDirichletByRegion[i].begin(),
                            isDirichletByRegion[i].end(),
                            [=](bool i) { return i == 1; }))
            {
                m_dirCoeffOffsets.push_back(bndcnt);

                // Gather index
                for (size_t j = 0; j < nBndExpCoeff; ++j)
                {
                    index[bndcnt + j] = cnt + j;
                }
                for (unsigned int nc = 0; nc < nComp; ++nc)
                {
                    if (isDirichletByRegion[i][nc])
                    {
                        for (size_t j = 0; j < nBndExpCoeff; ++j)
                        {
                            dirIndexByComp[nc].push_back(bndcnt + j);
                        }
                    }
                }
                bndcnt += nBndExpCoeff;
            }
            cnt += nBndExpCoeff;
        }

        ASSERTL1(bndcnt == m_numBndCoeffCompSize,
                 "The component size does not match the number of coefficients "
                 "for all Dirichlet boundaries.")

        m_map.clear();
        m_sign.clear();
        m_bndCoeff.clear();
        m_bndCoeffSrc.clear();
        m_bndCoeffHost.clear();
        m_compactBndCoeff.assign(nComp * m_numBndCoeffCompSize, 0.0);
        m_parDirBndSign.clear();
        m_parDirOffsets.assign(domainBlocks.size(),
                               std::vector<size_t>(nComp, 0));
        m_parDirCounts.assign(domainBlocks.size(),
                              std::vector<size_t>(nComp, 0));
        m_locid0.resize(domainBlocks.size());
        m_locid1.resize(domainBlocks.size());
        m_locsign.resize(domainBlocks.size());
        for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
        {
            m_locid0[blk].resize(domainBlocks.size());
            m_locid1[blk].resize(domainBlocks.size());
            m_locsign[blk].resize(domainBlocks.size());
        }
        m_locOffsets.assign(
            domainBlocks.size(),
            std::vector<std::vector<size_t>>(domainBlocks.size(),
                                             std::vector<size_t>(nComp, 0)));
        m_locCounts.assign(
            domainBlocks.size(),
            std::vector<std::vector<size_t>>(domainBlocks.size(),
                                             std::vector<size_t>(nComp, 0)));
        m_nParDirBndSignSize.resize(nComp, 0);
        m_localDirSize.resize(nComp, 0);
        m_signChange.resize(nComp, false);
        m_compOffsets.assign(domainBlocks.size(),
                             std::vector<size_t>(nComp, 0));
        m_compCounts.assign(domainBlocks.size(), std::vector<size_t>(nComp, 0));
        m_anySignChange = false;

        std::vector<std::vector<size_t>> mapBlockByBlk(domainBlocks.size());
        std::vector<std::vector<TData>> signBlockByBlk(domainBlocks.size());
        std::vector<std::vector<size_t>> bndCoeffSrcBlockByBlk(
            domainBlocks.size());
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

        auto expContField = std::dynamic_pointer_cast<MultiRegions::ContField>(
            this->m_expansionList);

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
                MultiRegions::ContField compfield(session, graph,
                                                  components[nc], true, false,
                                                  Collections::eNoCollection);
                assmbMap = compfield.GetLocalToGlobalMap();
            }
            auto &sign = assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
            auto &map  = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();
            auto &parallelDirBndSign = assmbMap->GetParallelDirBndSign();
            m_signChange[nc]         = assmbMap->GetSignChange();

            std::vector<std::tuple<size_t, size_t>> mapReordered;
            mapReordered.reserve(dirIndexByComp[nc].size());
            for (size_t idx : dirIndexByComp[nc])
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
                const size_t compCount  = blockIndices[blk].size();
                m_compOffsets[blk][nc]  = compOffset;
                m_compCounts[blk][nc]   = compCount;

                mapBlockByBlk[blk].reserve(compOffset + compCount);
                signBlockByBlk[blk].reserve(compOffset + compCount);
                bndCoeffSrcBlockByBlk[blk].reserve(compOffset + compCount);

                for (size_t idx : blockIndices[blk])
                {
                    mapBlockByBlk[blk].push_back(
                        std::get<1>(mapReordered[idx]) - offset);
                    signBlockByBlk[blk].push_back(
                        m_signChange[nc]
                            ? sign[index[std::get<0>(mapReordered[idx])]]
                            : static_cast<TData>(1));
                    bndCoeffSrcBlockByBlk[blk].push_back(
                        std::get<0>(mapReordered[idx]) +
                        nc * m_numBndCoeffCompSize);
                }
            }
            m_anySignChange = m_anySignChange || m_signChange[nc];

            // local dir dofs.
            m_localDirSize[nc] = assmbMap->GetCopyLocalDirDofs().size();
            std::vector<std::tuple<size_t, size_t, TData>> locReordered(
                m_localDirSize[nc]);
            if (m_localDirSize[nc] > 0)
            {
                cnt = 0;
                for (auto &it : assmbMap->GetCopyLocalDirDofs())
                {
                    locReordered[cnt] = it;
                    cnt++;
                }
                std::sort(std::begin(locReordered), std::end(locReordered),
                          [](std::tuple<size_t, size_t, TData> const &t1,
                             std::tuple<size_t, size_t, TData> const &t2) {
                              return std::tie(std::get<1>(t1),
                                              std::get<0>(t1)) <
                                     std::tie(std::get<1>(t2), std::get<0>(t2));
                          });
            }

            if (m_localDirSize[nc] > 0)
            {
                std::vector<size_t> nLocCoeffBlock(domainBlocks.size(), 0);
                std::vector<std::vector<size_t>> locid0Block(
                    domainBlocks.size());
                std::vector<std::vector<size_t>> locid1Block(
                    domainBlocks.size());
                std::vector<std::vector<TData>> locsignBlock(
                    domainBlocks.size());
                unsigned int blk0 = 0, blk1 = 0;
                size_t offset0 = 0, offset1 = 0;
                unsigned int i = 0;
                while (blk1 < domainBlocks.size())
                {
                    if (i == m_localDirSize[nc] ||
                        std::get<1>(locReordered[i]) >= blockBound[blk1])
                    {
                        for (size_t blk0 = 0; blk0 < domainBlocks.size();
                             ++blk0)
                        {
                            const size_t compOffset =
                                locid0BlockByPair[blk1][blk0].size();
                            const size_t compCount = locid0Block[blk0].size();
                            m_locOffsets[blk1][blk0][nc] = compOffset;
                            m_locCounts[blk1][blk0][nc]  = compCount;

                            auto &dst0 = locid0BlockByPair[blk1][blk0];
                            auto &dst1 = locid1BlockByPair[blk1][blk0];
                            auto &dsts = locsignBlockByPair[blk1][blk0];
                            dst0.insert(dst0.end(), locid0Block[blk0].begin(),
                                        locid0Block[blk0].end());
                            dst1.insert(dst1.end(), locid1Block[blk0].begin(),
                                        locid1Block[blk0].end());
                            dsts.insert(dsts.end(), locsignBlock[blk0].begin(),
                                        locsignBlock[blk0].end());
                            locid0Block[blk0].clear();
                            locid1Block[blk0].clear();
                            locsignBlock[blk0].clear();
                            nLocCoeffBlock[blk0] = 0;
                        }
                        offset1 = blockBound[blk1];
                        blk1++;
                    }
                    else if (std::get<0>(locReordered[i]) >= blockBound[blk0])
                    {
                        offset0 = blockBound[blk0];
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
            std::vector<int> parDirBndSignReordered(m_nParDirBndSignSize[nc]);
            if (m_nParDirBndSignSize[nc] > 0)
            {
                cnt = 0;
                for (auto &it : parallelDirBndSign)
                {
                    parDirBndSignReordered[cnt] = it;
                    cnt++;
                }
                std::sort(
                    std::begin(parDirBndSignReordered),
                    std::end(parDirBndSignReordered),
                    [](const size_t t1, const size_t t2) { return t1 < t2; });
            }

            if (m_nParDirBndSignSize[nc] > 0)
            {
                std::vector<int> parDirBndSignBlock;
                unsigned int i = 0, blk = 0, offset = 0;
                while (blk < domainBlocks.size())
                {
                    if (i == m_nParDirBndSignSize[nc] ||
                        parDirBndSignReordered[i] >= blockBound[blk])
                    {
                        const size_t compOffset  = parDirBlockByBlk[blk].size();
                        const size_t compCount   = parDirBndSignBlock.size();
                        m_parDirOffsets[blk][nc] = compOffset;
                        m_parDirCounts[blk][nc]  = compCount;

                        auto &dst = parDirBlockByBlk[blk];
                        dst.insert(dst.end(), parDirBndSignBlock.begin(),
                                   parDirBndSignBlock.end());
                        parDirBndSignBlock.clear();
                        offset = blockBound[blk];
                        blk++;
                    }
                    else
                    {
                        parDirBndSignBlock.push_back(parDirBndSignReordered[i] -
                                                     offset);
                        i++;
                    }
                }
            }
        }

        m_map.reserve(domainBlocks.size());
        m_bndCoeff.reserve(domainBlocks.size());
        m_bndCoeffSrc.reserve(domainBlocks.size());
        m_bndCoeffHost.reserve(domainBlocks.size());
        if (m_anySignChange)
        {
            m_sign.reserve(domainBlocks.size());
        }
        m_parDirBndSign.reserve(domainBlocks.size());
        for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
        {
            ASSERTL1(mapBlockByBlk[blk].size() ==
                         bndCoeffSrcBlockByBlk[blk].size(),
                     "Mismatch between map and boundary coefficient sizes.");
            m_map.push_back(
                LibUtilities::MemoryRegion<size_t>::template FromVector<
                    MemSpace, size_t>(mapBlockByBlk[blk]));
            m_bndCoeffSrc.push_back(std::move(bndCoeffSrcBlockByBlk[blk]));
            m_bndCoeffHost.emplace_back(m_bndCoeffSrc.back().size(), 0.0);
            m_bndCoeff.push_back(
                LibUtilities::MemoryRegion<TData>::template FromVector<MemSpace,
                                                                       TData>(
                    m_bndCoeffHost.back()));
            if (m_anySignChange)
            {
                m_sign.push_back(
                    LibUtilities::MemoryRegion<TData>::template FromVector<
                        MemSpace, TData>(signBlockByBlk[blk]));
            }
            m_parDirBndSign.push_back(
                LibUtilities::MemoryRegion<int>::template FromVector<MemSpace,
                                                                     int>(
                    parDirBlockByBlk[blk]));
        }

        this->UpdateBndCoeffs(static_cast<TData>(0.0));

        for (unsigned int blk1 = 0; blk1 < domainBlocks.size(); ++blk1)
        {
            for (unsigned int blk0 = 0; blk0 < domainBlocks.size(); ++blk0)
            {
                m_locid0[blk1][blk0] =
                    LibUtilities::MemoryRegion<size_t>::template FromVector<
                        MemSpace, size_t>(locid0BlockByPair[blk1][blk0]);
                m_locid1[blk1][blk0] =
                    LibUtilities::MemoryRegion<size_t>::template FromVector<
                        MemSpace, size_t>(locid1BlockByPair[blk1][blk0]);
                m_locsign[blk1][blk0] =
                    LibUtilities::MemoryRegion<TData>::template FromVector<
                        MemSpace, TData>(locsignBlockByPair[blk1][blk0]);
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

    std::vector<MultiRegions::Field<TData, FieldState::Phys>> m_wsp_phys;
    std::vector<MultiRegions::Field<TData, FieldState::Coeff>> m_wsp_coeffs;
    std::vector<std::shared_ptr<ExpressionOp<TData>>> m_expressionOps;
    std::vector<std::shared_ptr<FwdTransBCOp<TData>>> m_fwdTransBCOps;
    std::vector<size_t> m_dirCoeffOffsets;
    std::vector<size_t> m_dirNumCoeffs;

    std::vector<LibUtilities::MemoryRegion<size_t>> m_map;
    std::vector<LibUtilities::MemoryRegion<TData>> m_sign;
    std::vector<LibUtilities::MemoryRegion<TData>> m_bndCoeff;
    std::vector<std::vector<size_t>> m_bndCoeffSrc;
    std::vector<TData> m_compactBndCoeff;
    std::vector<std::vector<TData>> m_bndCoeffHost;
    std::vector<std::vector<size_t>> m_compOffsets;
    std::vector<std::vector<size_t>> m_compCounts;

    std::vector<size_t> m_nParDirBndSignSize;
    std::vector<size_t> m_localDirSize;

    std::vector<LibUtilities::MemoryRegion<int>> m_parDirBndSign;
    std::vector<std::vector<size_t>> m_parDirOffsets;
    std::vector<std::vector<size_t>> m_parDirCounts;
    std::vector<std::vector<LibUtilities::MemoryRegion<size_t>>> m_locid0;
    std::vector<std::vector<LibUtilities::MemoryRegion<size_t>>> m_locid1;
    std::vector<std::vector<LibUtilities::MemoryRegion<TData>>> m_locsign;
    std::vector<std::vector<std::vector<size_t>>> m_locOffsets;
    std::vector<std::vector<std::vector<size_t>>> m_locCounts;

    void v_UpdateBndCoeffs(const TData &time) override
    {
        if (!m_hasAnyBndCoeff)
        {
            return;
        }

        const unsigned int nComp = this->m_components.size();
        std::fill(m_compactBndCoeff.begin(), m_compactBndCoeff.end(), 0.0);

        for (unsigned int iDir = 0; iDir < m_expressionOps.size(); ++iDir)
        {
            const size_t nBndExpCoeff = m_dirNumCoeffs[iDir];
            const size_t bndOffset    = m_dirCoeffOffsets[iDir];

            // Zero wsp_phys and wsp_coeff
            zero<ExecSpace>(m_wsp_phys[iDir]);
            zero<ExecSpace>(m_wsp_coeffs[iDir]);

            // Update time
            m_expressionOps[iDir]->SetTime(time);

            // Evaluate Dirichlet expression and project onto boundary
            m_expressionOps[iDir]->Apply(m_wsp_phys[iDir], m_wsp_phys[iDir]);
            m_fwdTransBCOps[iDir]->Apply(m_wsp_phys[iDir], m_wsp_coeffs[iDir]);

            std::vector<TData> tmp =
                m_wsp_coeffs[iDir].template ToVector<TData>();

            ASSERTL1(tmp.size() == nComp * nBndExpCoeff,
                     "Unexpected boundary coefficient vector size.");
            for (unsigned int nc = 0; nc < nComp; ++nc)
            {
                std::copy(tmp.begin() + nc * nBndExpCoeff,
                          tmp.begin() + (nc + 1) * nBndExpCoeff,
                          m_compactBndCoeff.begin() +
                              nc * m_numBndCoeffCompSize + bndOffset);
            }
        }

        for (unsigned int blk = 0; blk < m_bndCoeff.size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &bndCoeffBlock = m_bndCoeffHost[blk];
            for (size_t i = 0; i < m_bndCoeffSrc[blk].size(); ++i)
            {
                bndCoeffBlock[i] = m_compactBndCoeff[m_bndCoeffSrc[blk][i]];
            }
            m_bndCoeff[blk].template CopyVector<MemSpace, TData>(bndCoeffBlock,
                                                                 streamID);
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

            auto &inoutBlk = inout.GetBlocks()[blk];

            // Initialize pointers.
            auto inoutPtr =
                inoutBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto inoutWidth  = inoutBlk.GetInterleaveWidth();
            unsigned blksize = inoutBlk.CompSize();

            // if block is interlaced deInterleave block since currently mapping
            // set up assuming serial alignment
            ReshapeStorage<ExecSpace>(
                1u, inoutWidth,
                inoutBlk.GetNumElementsWithPadding() * inout.GetNumComponents(),
                inoutBlk.GetNumData(), inoutPtr, streamID);

            auto mapPtrBlock =
                m_map[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
            auto bndcoeffPtrBlock =
                m_bndCoeff[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
            const TData *signPtrBlock =
                m_anySignChange
                    ? m_sign[blk].template GetPtr<MemSpace, ReadOnly>(streamID)
                    : nullptr;

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

        // TODO: Universal assembly on device.
        // This gather is also required in serial to resolve duplicated local
        // Dirichlet boundary coefficients before copy-local-dir processing,
        // mirroring ContField::v_ImposeDirichletConditions().
        auto contfield = std::dynamic_pointer_cast<MultiRegions::ContField>(
            this->m_expansionList);

        // Copy the data from the input field.
        auto inoutarr = inout.template ToArray<double>();

        contfield->GetLocalToGlobalMap()->UniversalAbsMaxBnd(inoutarr);

        // Copy the data to the output field.
        inout.template CopyArray<MemSpace, double>(inoutarr);

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

        for (unsigned nc = 0; nc < inout.GetNumComponents(); ++nc)
        {
            if (m_localDirSize[nc] == 0)
            {
                continue;
            }

            for (unsigned int blk1 = 0; blk1 < inout.GetBlocks().size(); ++blk1)
            {
                auto blksize1 = inout.GetBlocks()[blk1].CompSize();
                for (unsigned int blk0 = 0; blk0 < inout.GetBlocks().size();
                     ++blk0)
                {
                    const unsigned int streamID = blk0 + 1;

                    auto nLocCoeffBlock = m_locCounts[blk1][blk0][nc];
                    if (nLocCoeffBlock == 0)
                    {
                        continue;
                    }

                    auto inptr =
                        inout.GetBlocks()[blk1]
                            .template GetPtr<MemSpace, ReadOnly>(streamID);
                    auto outptr =
                        inout.GetBlocks()[blk0]
                            .template GetPtr<MemSpace, WriteOnly>(streamID);
                    const size_t offset = m_locOffsets[blk1][blk0][nc];
                    auto locid0Ptr =
                        m_locid0[blk1][blk0]
                            .template GetPtr<MemSpace, ReadOnly>(streamID) +
                        offset;
                    auto locid1Ptr =
                        m_locid1[blk1][blk0]
                            .template GetPtr<MemSpace, ReadOnly>(streamID) +
                        offset;
                    auto locsignPtr =
                        m_locsign[blk1][blk0]
                            .template GetPtr<MemSpace, ReadOnly>(streamID) +
                        offset;

                    auto blksize0 = inout.GetBlocks()[blk0].CompSize();
                    LocalDirBndCondKernel<ExecSpace>(
                        nLocCoeffBlock, locid0Ptr, locid1Ptr, locsignPtr,
                        inptr + nc * blksize1, outptr + nc * blksize0,
                        streamID);
                }
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
