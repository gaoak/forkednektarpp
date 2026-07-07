///////////////////////////////////////////////////////////////////////////////
//
// File: DGDirBndCondOpImpl.hpp
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

#include <MultiRegions/DisContField.h>

#include "Operators/BndCondOps/DGDirBndCond/DGDirBndCondKernels.hpp"
#include "Operators/BndCondOps/DGDirBndCond/DGDirBndCondOp.hpp"

#include "Operators/ElmtOps/Expression/ExpressionOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class DGDirBndCondOpImpl : public DGDirBndCondOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DGDirBndCondOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                       const std::vector<std::string> &components)
        : DGDirBndCondOp<TData>(expansionList, components)
    {
        auto session       = this->m_expansionList->GetSession();
        auto graph         = this->m_expansionList->GetGraph();
        unsigned int nhomo = 1;
        session->LoadParameter("HomModesZ", nhomo, 1);
        unsigned int nComp = components.size();

        // Get trace and trace map from expansion list
        auto &trace    = expansionList->GetTrace();
        auto &traceMap = expansionList->GetTraceMap();

        // Create boundary conditions
        const SpatialDomains::BoundaryConditions bcs(session, graph);
        const SpatialDomains::BoundaryRegionCollection &bregions =
            bcs.GetBoundaryRegions();
        const SpatialDomains::BoundaryConditionCollection &bconditions =
            bcs.GetBoundaryConditions();

        // Create counters and temporary variables for BC Fields and Operators
        size_t nTotalBcExpSize = 0;
        std::vector<size_t> dirBCID;
        std::vector<std::vector<bool>> isDirichletByRegion;
        std::vector<size_t> numBcExpPhys;
        std::vector<size_t> bcExpSizes;
        std::vector<size_t> bcExpPhysPerElmt;
        std::vector<size_t> bcExpPhysOffset;
        std::vector<size_t> bcTraceOffset;
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
            numBcExpPhys.push_back(bcExpList->GetTotPoints());

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
            }

            isDirichletByRegion.push_back(isDirichlet);

            // Skip, if no Dirichletn condition in this boundary region
            if (!hasDirichletCondition)
            {
                continue;
            }

            // Save number of elements in this BC expansion
            bcExpSizes.push_back(bcExpList->GetExpSize());
            for (size_t e = 0; e < bcExpList->GetExpSize(); ++e)
            {
                // Save number of phys points per element
                bcExpPhysPerElmt.push_back(
                    bcExpList->GetExp(e)->GetTotPoints());
                // Save offsets of volume
                bcExpPhysOffset.push_back(bcExpList->GetPhys_Offset(e));
                // Save offsets of trace
                bcTraceOffset.push_back(
                    trace->GetPhys_Offset(traceMap->GetBndCondIDToGlobalTraceID(
                        nTotalBcExpSize + e)));
            }

            // Update total number of elements processed so far
            nTotalBcExpSize += bcExpList->GetExpSize();

            // Create fields for evaluating BCs into
            auto blocks_phys =
                GetBlockAttributes<TData, FieldState::Phys>(bcExpList);
            this->m_wsp_phys.push_back(Field<TData, FieldState::Phys>(
                "Dirichlet BC phys", blocks_phys, nComp, nhomo));

            // Compute number of boundary coefficients.
            for (unsigned int blk = 0; blk < blocks_phys.size(); ++blk)
            {
                const auto nphys = blocks_phys[blk].GetNumData();
                const auto nelmt = blocks_phys[blk].GetNumElements();
                m_numBndPhysCompSize += nelmt * nphys;
            }

            // Initialize memory regions
            this->m_wsp_phys.back().template Initialize<MemSpace>(0.0);

            // Create operators for this boundary condition
            this->m_expressionOps.push_back(
                ExpressionOp<TData>::Create(bcExpList, components, "Serial"));
            this->m_expressionOps.back()->SetComponentMask(isDirichlet);

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

        // Return if no Dirichlet boundary coefficients.
        if (m_numBndPhysCompSize == 0)
        {
            return;
        }

        // Compute block bound.
        auto domainBlocks = GetBlockAttributes<TData, FieldState::Phys>(trace);
        std::vector<size_t> blockBound(domainBlocks.size());
        size_t bound = 0;
        for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
        {
            const auto &block = domainBlocks[blk];
            const auto nphy   = block.GetNumData();
            const auto nelmt  = block.GetNumElements();
            bound += nelmt * nphy;
            blockBound[blk] = bound;
        }

        // Collecting boundary phys.
        // and index map (map for one component only)
        std::vector<TData> bndphys(nComp * m_numBndPhysCompSize);
        std::vector<size_t> index(m_numBndPhysCompSize);
        std::vector<std::vector<size_t>> dirIndexByComp(nComp);
        size_t bndcnt = 0, el = 0;
        for (unsigned int i = 0, iDir = 0; i < bregions.size(); ++i)
        {
            // Get number of coefficients for this BC
            auto nBndExpPhys = numBcExpPhys[i];

            // Process Dirichlet boundary conditions
            if (std::any_of(isDirichletByRegion[i].begin(),
                            isDirichletByRegion[i].end(),
                            [=](bool i) { return i == 1; }))
            {
                // Evaluate Dirichlet BCs and put into vector
                m_expressionOps[iDir]->Apply(m_wsp_phys[iDir],
                                             m_wsp_phys[iDir]);

                // Concatenate vector of all boundary phys (ToVector
                // removes padding)
                std::vector<TData> tmp =
                    m_wsp_phys[iDir].template ToVector<TData>();

                // Sort vector in boundary-region major to be consistent with
                // map layout
                ASSERTL1(tmp.size() == nComp * nBndExpPhys,
                         "Unexpected boundary phys vector size.");

                auto ne = bcExpSizes[iDir];
                for (size_t e = 0; e < ne; ++e)
                {
                    auto npts = bcExpPhysPerElmt[el + e];
                    auto id1  = bcExpPhysOffset[el + e];
                    auto id2  = bcTraceOffset[el + e];

                    for (unsigned int nc = 0; nc < nComp; nc++)
                    {
                        std::copy(tmp.data() + nc * nBndExpPhys + id1,
                                  tmp.data() + nc * nBndExpPhys + id1 + npts,
                                  bndphys.data() + nc * nBndExpPhys + bndcnt);
                    }

                    // Gather index
                    for (unsigned int j = 0; j < npts; ++j)
                    {
                        index[bndcnt + j] = id2 + j;
                    }

                    for (unsigned int nc = 0; nc < nComp; ++nc)
                    {
                        if (isDirichletByRegion[i][nc])
                        {
                            for (unsigned int j = 0; j < npts; ++j)
                            {
                                dirIndexByComp[nc].push_back(bndcnt + j);
                            }
                        }
                    }

                    bndcnt += npts;
                }

                el += ne;

                // Increment Dirichlet boundaries
                iDir++;
            }
        }

        ASSERTL1(bndcnt == m_numBndPhysCompSize,
                 "The component size does not match the number of phys"
                 "for all Dirichlet boundaries.")

        m_bndPhys.clear();
        m_compOffsets.assign(domainBlocks.size(),
                             std::vector<size_t>(nComp, 0));
        m_compCounts.assign(domainBlocks.size(), std::vector<size_t>(nComp, 0));

        std::vector<std::vector<size_t>> mapBlockByBlk(domainBlocks.size());
        std::vector<std::vector<TData>> bndPhysBlockByBlk(domainBlocks.size());

        for (unsigned int nc = 0; nc < nComp; ++nc)
        {
            std::vector<std::tuple<size_t, size_t, TData>> mapReordered;
            mapReordered.reserve(dirIndexByComp[nc].size());
            for (size_t idx : dirIndexByComp[nc])
            {
                mapReordered.push_back(std::make_tuple(
                    idx, index[idx], bndphys[idx + nc * m_numBndPhysCompSize]));
            }
            std::sort(mapReordered.begin(), mapReordered.end(),
                      [](std::tuple<size_t, size_t, TData> const &t1,
                         std::tuple<size_t, size_t, TData> const &t2) {
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
                bndPhysBlockByBlk[blk].reserve(compOffset + compCount);

                for (size_t idx : blockIndices[blk])
                {
                    mapBlockByBlk[blk].push_back(
                        std::get<1>(mapReordered[idx]) - offset);
                    bndPhysBlockByBlk[blk].push_back(
                        std::get<2>(mapReordered[idx]));
                }
            }
        }

        m_map.reserve(domainBlocks.size());
        m_bndPhys.reserve(domainBlocks.size());

        for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
        {
            ASSERTL1(mapBlockByBlk[blk].size() == bndPhysBlockByBlk[blk].size(),
                     "Mismatch between map and boundary coefficient sizes.");
            m_map.push_back(
                MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                    mapBlockByBlk[blk]));
            m_bndPhys.push_back(
                MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                    bndPhysBlockByBlk[blk]));
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<DGDirBndCondOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    size_t m_numBndPhysCompSize = 0;

    std::vector<Field<TData, FieldState::Phys>> m_wsp_phys;
    std::vector<std::shared_ptr<ExpressionOp<TData>>> m_expressionOps;

    std::vector<MemoryRegion<size_t>> m_map;
    std::vector<MemoryRegion<TData>> m_bndPhys;
    std::vector<std::vector<size_t>> m_compOffsets;
    std::vector<std::vector<size_t>> m_compCounts;

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Return if no Dirichlet boundary condition.
        if (m_numBndPhysCompSize == 0)
        {
            return;
        }

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            auto &inoutBlk = inout.GetBlocks()[blk];

            // Initialize pointers.
            auto inoutPtr = inoutBlk.template GetPtr<MemSpace, ReadWrite>();

            // if block is interlaced deInterleave block since currently mapping
            // set up assuming serial alignment
            auto inoutWidth  = inoutBlk.GetInterleaveWidth();
            unsigned blksize = inoutBlk.CompSize();
            if (inoutWidth != 1)
            {
                for (unsigned nc = 0; nc < inout.GetNumComponents(); ++nc)
                {
                    ReshapeStorage<ExecSpace>(
                        1u, inoutWidth, inoutBlk.GetNumElementsWithPadding(),
                        inoutBlk.GetNumData(), inoutPtr + nc * blksize);
                }
                inoutBlk.template SetInterleaveWidth<TData>(1);
            }

            auto mapPtrBlock = m_map[blk].template GetPtr<MemSpace, ReadOnly>();
            auto bndPhysPtrBlock =
                m_bndPhys[blk].template GetPtr<MemSpace, ReadOnly>();

            for (unsigned nc = 0; nc < inout.GetNumComponents(); ++nc)
            {
                auto nbndPhysBlk = m_compCounts[blk][nc];
                if (nbndPhysBlk == 0)
                {
                    continue;
                }

                const size_t offset = m_compOffsets[blk][nc];
                auto mapPtr         = mapPtrBlock + offset;
                auto bndPhysPtr     = bndPhysPtrBlock + offset;

                DGDirBndCondKernel<ExecSpace>(nbndPhysBlk, mapPtr, bndPhysPtr,
                                              inoutPtr + nc * blksize);
            }
        }
    }
};

} // namespace Nektar::Operators::detail
