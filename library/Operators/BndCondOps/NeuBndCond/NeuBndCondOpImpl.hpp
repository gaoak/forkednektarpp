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

#include <MultiRegions/ContField.h>

#include "Operators/BndCondOps/NeuBndCond/NeuBndCondKernels.hpp"
#include "Operators/BndCondOps/NeuBndCond/NeuBndCondOp.hpp"

#include "Operators/ElmtOps/Expression/ExpressionOp.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp"
#include "Operators/Math/MathKernels.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
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
        std::vector<std::vector<bool>> isNeumannByRegion;
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

            auto conditionMapIter = bndCondMap->find(components[0]);

            ASSERTL1(conditionMapIter != bndCondMap->end(),
                     "Unable to locate condition map.");

            // Get BoundaryCondition(SharedPtr)
            bc = (*conditionMapIter).second;

            // Create boundary condition expansion
            bcExpList = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                session, *(it.second), graph, true, components[0], false,
                bc->GetComm(), Collections::eNoImpType);

            // Set data warehouse for redesign operators
            bcExpList->SetDataWarehouse();

            // Save number of coefficients
            numBcExpCoeffs.push_back(bcExpList->GetNcoeffs());

            // Check if any component has a Neumann-type condition
            bool hasNeumanCondition = false;
            std::vector<bool> isNeumann(nComp);
            for (unsigned int nc = 0; nc < nComp; nc++)
            {
                // Get BoundaryCondition and check if it is Neumann
                auto cndMapIter = bndCondMap->find(components[nc]);
                bc              = (*cndMapIter).second;
                bool tmp        = (bool)std::dynamic_pointer_cast<
                    SpatialDomains::NeumannBoundaryCondition>(bc);

                // Save if component has a Neumann condition
                isNeumann[nc] = tmp;

                // Check if any component has a Neumann condition
                hasNeumanCondition = hasNeumanCondition || tmp;
            }

            isNeumannByRegion.push_back(isNeumann);

            // Skip, if no Neumann condition in this boundary region
            if (!hasNeumanCondition)
            {
                continue;
            }

            // Create fields for evaluating BCs into
            auto blocks_phys =
                GetBlockAttributes<TData, FieldState::Phys>(bcExpList);
            this->m_wsp_phys.push_back(Field<TData, FieldState::Phys>(
                "Neumann BC phys", blocks_phys, nComp, nhomo));
            auto blocks_coeffs =
                GetBlockAttributes<TData, FieldState::Coeff>(bcExpList);
            this->m_wsp_coeffs.push_back(Field<TData, FieldState::Coeff>(
                "Neumann BC coeff", blocks_coeffs, nComp, nhomo));

            // Compute number of boundary coefficients.
            for (unsigned int blk = 0; blk < blocks_coeffs.size(); ++blk)
            {
                const auto ncoeff = blocks_coeffs[blk].GetNumData();
                const auto nelmt  = blocks_coeffs[blk].GetNumElements();
                m_numBndCoeffCompSize += nelmt * ncoeff;
            }

            // Initialize memory regions
            this->m_wsp_phys.back().template Initialize<MemSpace>(0.0);
            this->m_wsp_coeffs.back().template Initialize<MemSpace>(0.0);

            // Create operators for this boundary condition
            this->m_expressionOps.push_back(
                ExpressionOp<TData>::Create(bcExpList, components, "Serial"));
            this->m_expressionOps.back()->SetComponentMask(isNeumann);

            // Note we do a copy for 0D (points)
            if (bcExpList->GetShapeDimension() != 0)
            {
                this->m_iprodOps.push_back(IProductWRTBaseOp<TData>::Create(
                    bcExpList, components, ExecSpace::name));
            }

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

        // Return if no Neumann boundary coefficients.
        if (m_numBndCoeffCompSize == 0)
        {
            return;
        }

        // Compute block bound.
        auto domainBlocks =
            GetBlockAttributes<TData, FieldState::Coeff>(this->m_expansionList);
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

        // Collecting boundary coefficients.
        // and index map (map for one component only)
        std::vector<TData> bndcoeff(nComp * m_numBndCoeffCompSize);
        std::vector<size_t> index(m_numBndCoeffCompSize);
        std::vector<std::vector<size_t>> neuIndexByComp(nComp);
        size_t bndcnt = 0, cnt = 0;
        for (unsigned int i = 0, iNeu = 0; i < bregions.size(); ++i)
        {
            // Get number of coefficients for this BC
            auto nBndExpCoeff = numBcExpCoeffs[i];

            // Process Neumann boundary conditions
            if (std::any_of(isNeumannByRegion[i].begin(),
                            isNeumannByRegion[i].end(),
                            [=](bool i) { return i == 1; }))
            {
                // Evaluate Neumann BC string
                m_expressionOps[iNeu]->Apply(m_wsp_phys[iNeu],
                                             m_wsp_phys[iNeu]);

                // Evaluate IProduct for Neumann BC
                if (bcExpList->GetShapeDimension() != 0)
                {
                    m_iprodOps[iNeu]->Apply(m_wsp_phys[iNeu],
                                            m_wsp_coeffs[iNeu]);
                }
                // Note that for 0D we simply need a copy
                else
                {
                    auto physptr = m_wsp_phys[iNeu]
                                       .GetBlocks()[0]
                                       .template GetPtr<MemSpace, ReadOnly>();
                    auto coeffptr = m_wsp_coeffs[iNeu]
                                        .GetBlocks()[0]
                                        .template GetPtr<MemSpace, WriteOnly>();
                    size_t nsize = nBndExpCoeff;
                    for (size_t nc = 0; nc < nComp; nc++)
                    {
                        copyKernel<ExecSpace>(nsize, physptr, coeffptr);
                        physptr += m_wsp_phys[iNeu].GetBlocks()[0].CompSize();
                        coeffptr +=
                            m_wsp_coeffs[iNeu].GetBlocks()[0].CompSize();
                    }
                }

                // Concatenate vector of all boundary coefficients
                std::vector<TData> tmp =
                    m_wsp_coeffs[iNeu].template ToVector<TData>();
                ASSERTL1(tmp.size() == nComp * nBndExpCoeff,
                         "Unexpected boundary coefficient vector size.");
                for (unsigned int nc = 0; nc < nComp; ++nc)
                {
                    std::copy(tmp.begin() + nc * nBndExpCoeff,
                              tmp.begin() + (nc + 1) * nBndExpCoeff,
                              bndcoeff.begin() + nc * m_numBndCoeffCompSize +
                                  bndcnt);
                }

                // Gather index
                for (size_t j = 0; j < nBndExpCoeff; ++j)
                {
                    index[bndcnt + j] = cnt + j;
                }
                for (unsigned int nc = 0; nc < nComp; ++nc)
                {
                    if (isNeumannByRegion[i][nc])
                    {
                        for (size_t j = 0; j < nBndExpCoeff; ++j)
                        {
                            neuIndexByComp[nc].push_back(bndcnt + j);
                        }
                    }
                }
                bndcnt += nBndExpCoeff;

                // Increment Neumann boundaries
                iNeu++;
            }
            cnt += nBndExpCoeff;
        }

        ASSERTL1(bndcnt == m_numBndCoeffCompSize,
                 "The component size does not match the number of coefficients "
                 "for all Neumann boundaries.")

        m_map.clear();
        m_sign.clear();
        m_bndCoeff.clear();
        m_signChange.resize(nComp, false);
        m_compOffsets.assign(domainBlocks.size(),
                             std::vector<size_t>(nComp, 0));
        m_compCounts.assign(domainBlocks.size(), std::vector<size_t>(nComp, 0));
        m_anySignChange = false;

        std::vector<std::vector<size_t>> mapBlockByBlk(domainBlocks.size());
        std::vector<std::vector<TData>> signBlockByBlk(domainBlocks.size());
        std::vector<std::vector<TData>> bndCoeffBlockByBlk(domainBlocks.size());

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
            auto &sign       = assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
            auto &map        = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();
            m_signChange[nc] = assmbMap->GetSignChange();

            std::vector<std::tuple<size_t, size_t, double>> mapReordered;
            mapReordered.reserve(neuIndexByComp[nc].size());
            for (size_t idx : neuIndexByComp[nc])
            {
                mapReordered.push_back(std::make_tuple(
                    idx, map[index[idx]],
                    bndcoeff[idx + nc * m_numBndCoeffCompSize]));
            }
            std::sort(mapReordered.begin(), mapReordered.end(),
                      [](std::tuple<size_t, size_t, double> const &t1,
                         std::tuple<size_t, size_t, double> const &t2) {
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
                bndCoeffBlockByBlk[blk].reserve(compOffset + compCount);

                for (size_t idx : blockIndices[blk])
                {
                    mapBlockByBlk[blk].push_back(
                        std::get<1>(mapReordered[idx]) - offset);
                    signBlockByBlk[blk].push_back(
                        m_signChange[nc]
                            ? sign[index[std::get<0>(mapReordered[idx])]]
                            : static_cast<TData>(1));
                    bndCoeffBlockByBlk[blk].push_back(
                        std::get<2>(mapReordered[idx]));
                }
            }
            m_anySignChange = m_anySignChange || m_signChange[nc];
        }

        m_map.reserve(domainBlocks.size());
        m_bndCoeff.reserve(domainBlocks.size());
        if (m_anySignChange)
        {
            m_sign.reserve(domainBlocks.size());
        }
        for (unsigned int blk = 0; blk < domainBlocks.size(); ++blk)
        {
            ASSERTL1(mapBlockByBlk[blk].size() ==
                         bndCoeffBlockByBlk[blk].size(),
                     "Mismatch between map and boundary coefficient sizes.");
            m_map.push_back(
                MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                    mapBlockByBlk[blk]));
            m_bndCoeff.push_back(
                MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                    bndCoeffBlockByBlk[blk]));
            if (m_anySignChange)
            {
                m_sign.push_back(
                    MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                        signBlockByBlk[blk]));
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
        return std::make_unique<NeuBndCondOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    bool m_anySignChange         = false;
    size_t m_numBndCoeffCompSize = 0;
    std::vector<bool> m_signChange;

    std::vector<Field<TData, FieldState::Phys>> m_wsp_phys;
    std::vector<Field<TData, FieldState::Coeff>> m_wsp_coeffs;
    std::vector<std::shared_ptr<ExpressionOp<TData>>> m_expressionOps;
    std::vector<std::shared_ptr<IProductWRTBaseOp<TData>>> m_iprodOps;

    std::vector<MemoryRegion<size_t>> m_map;
    std::vector<MemoryRegion<TData>> m_sign;
    std::vector<MemoryRegion<TData>> m_bndCoeff;
    std::vector<std::vector<size_t>> m_compOffsets;
    std::vector<std::vector<size_t>> m_compCounts;

    void v_Apply(Field<TData, FieldState::Coeff> &inout) override
    {
        // Return if no Dirichlet boundary condition.
        if (m_numBndCoeffCompSize == 0)
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
                    deInterleave<ExecSpace>(
                        inoutWidth,
                        inoutBlk.GetNumElementsWithPadding() / inoutWidth,
                        inoutBlk.GetNumData(), inoutPtr + nc * blksize);
                }
                inoutBlk.template SetInterleaveWidth<TData>(1);
            }

            auto mapPtrBlock = m_map[blk].template GetPtr<MemSpace, ReadOnly>();
            auto bndcoeffPtrBlock =
                m_bndCoeff[blk].template GetPtr<MemSpace, ReadOnly>();
            const TData *signPtrBlock =
                m_anySignChange
                    ? m_sign[blk].template GetPtr<MemSpace, ReadOnly>()
                    : nullptr;

            // Add weak boundary condition forcing.
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
                    NeuBndCondKernel<ExecSpace>(nbndCoeffBlk, signPtr, mapPtr,
                                                bndcoeffPtr,
                                                inoutPtr + nc * blksize);
                }
                else
                {
                    NeuBndCondKernel<ExecSpace>(nbndCoeffBlk, mapPtr,
                                                bndcoeffPtr,
                                                inoutPtr + nc * blksize);
                }
            }
        }
    }
};

} // namespace Nektar::Operators::detail
