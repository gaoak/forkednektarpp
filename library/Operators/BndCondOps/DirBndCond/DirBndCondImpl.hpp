///////////////////////////////////////////////////////////////////////////////
//
// File: DirBndCondImpl.hpp
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

#include "Operators/BndCondOps/DirBndCond/OperatorDirBndCond.hpp"

#include "Operators/BndCondOps/DirBndCond/DirBndCondDeviceKernels.hpp"
#include "Operators/BndCondOps/DirBndCond/DirBndCondSerialAVXKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class OperatorDirBndCondImpl : public OperatorDirBndCond<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorDirBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDirBndCond<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto &bndCondExpansions = contfield->GetBndCondExpansions();
        auto &bndConditions     = contfield->GetBndConditions();
        auto assmbMap           = contfield->GetLocalToGlobalMap();
        auto &sign              = assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map               = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();
        auto &parallelDirBndSign = assmbMap->GetParallelDirBndSign();
        m_signChange             = assmbMap->GetSignChange();

        // Compute number boundary coefficients
        for (unsigned int i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::eDirichlet)
            {
                m_nBndCoeff += bndCondExpansions[i]->GetNcoeffs();
            }
        }

        // Return if no Dirichlet boundary condition.
        if (m_nBndCoeff == 0)
        {
            return;
        }

        // Compute block bound.
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);
        std::vector<int> blockBound(blocks.size());
        int bound = 0;
        for (int blk = 0; blk < blocks.size(); ++blk)
        {
            const auto &block = blocks[blk];
            const auto ncoeff = block.GetNumData();
            const auto nElmts = block.GetNumElements();
            bound += nElmts * ncoeff;
            blockBound[blk] = bound;
        }

        // Collecting boundary coefficients.
        std::vector<TData> bndcoeff(m_nBndCoeff);
        std::vector<int> index(m_nBndCoeff);
        unsigned int bndcnt = 0, cnt = 0;
        for (unsigned int i = 0; i < bndCondExpansions.size(); ++i)
        {
            auto nBndExpCoeff = bndCondExpansions[i]->GetNcoeffs();

            if (bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::eDirichlet)
            {
                auto &bndExpCoeff = bndCondExpansions[i]->GetCoeffs();
                std::copy(bndExpCoeff.data(), bndExpCoeff.data() + nBndExpCoeff,
                          bndcoeff.data() + bndcnt);
                for (unsigned int j = 0; j < nBndExpCoeff; ++j)
                {
                    index[bndcnt + j] = cnt + j;
                }
                bndcnt += nBndExpCoeff;
            }
            cnt += nBndExpCoeff;
        }

        // Sort boundary coefficients by increasing order of "map[index[i]]".
        std::vector<std::tuple<int, int, double>> mapReordered;
        for (int i = 0; i < m_nBndCoeff; i++)
        {
            mapReordered.push_back(
                std::make_tuple(i, map[index[i]], bndcoeff[i]));
        }
        std::sort(std::begin(mapReordered), std::end(mapReordered),
                  [](std::tuple<int, int, double> const &t1,
                     std::tuple<int, int, double> const &t2) {
                      return std::tie(std::get<1>(t1), std::get<0>(t1)) <
                             std::tie(std::get<1>(t2), std::get<0>(t2));
                  });

        // local dir dofs.
        m_localDirSize = assmbMap->GetCopyLocalDirDofs().size();
        std::vector<std::tuple<int, int, double>> locReordered(m_localDirSize);
        if (m_localDirSize > 0)
        {
            unsigned int cnt = 0;
            for (auto &it : assmbMap->GetCopyLocalDirDofs())
            {
                locReordered[cnt] = it;
                cnt++;
            }
            std::sort(std::begin(locReordered), std::end(locReordered),
                      [](std::tuple<int, int, double> const &t1,
                         std::tuple<int, int, double> const &t2) {
                          return std::tie(std::get<1>(t1), std::get<0>(t1)) <
                                 std::tie(std::get<1>(t2), std::get<0>(t2));
                      });
        }

        // Parallel dir sign.
        m_nParDirBndSignSize = parallelDirBndSign.size();
        std::vector<int> parDirBndSignReordered(m_nParDirBndSignSize);
        if (m_nParDirBndSignSize > 0)
        {
            unsigned int cnt = 0;
            for (auto &it : parallelDirBndSign)
            {
                parDirBndSignReordered[cnt] = it;
                cnt++;
            }
            std::sort(std::begin(parDirBndSignReordered),
                      std::end(parDirBndSignReordered),
                      [](const int t1, const int t2) { return t1 < t2; });
        }

        const bool device_only = true;

        // Split bndcoeff per block.
        std::vector<TData> bndCoeffBlock;
        std::vector<int> mapBlock;
        std::vector<TData> signBlock;
        int i = 0, blk = 0, offset = 0, nbndCoeffBlock = 0;
        while (blk < blocks.size())
        {
            if (i == m_nBndCoeff ||
                std::get<1>(mapReordered[i]) >= blockBound[blk])
            {
                m_nBndCoeffBlock.push_back(nbndCoeffBlock);
                m_bndCoeff.push_back(
                    MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                        bndCoeffBlock, ExecSpace::alignment, device_only));
                m_map.push_back(
                    MemoryRegion<int>::template FromVector<MemSpace, int>(
                        mapBlock, ExecSpace::alignment, device_only));
                if (m_signChange)
                {
                    m_sign.push_back(
                        MemoryRegion<TData>::template FromVector<MemSpace,
                                                                 TData>(
                            signBlock, ExecSpace::alignment, device_only));
                }
                nbndCoeffBlock = 0;
                bndCoeffBlock.clear();
                mapBlock.clear();
                signBlock.clear();
                offset = blockBound[blk];
                blk++;
            }
            else
            {
                bndCoeffBlock.push_back(std::get<2>(mapReordered[i]));
                mapBlock.push_back(std::get<1>(mapReordered[i]) - offset);
                if (m_signChange)
                {
                    signBlock.push_back(
                        sign[index[std::get<0>(mapReordered[i])]]);
                }
                nbndCoeffBlock++;
                i++;
            }
        }

        // Split locReordered per block.
        if (m_localDirSize > 0)
        {
            std::vector<unsigned int> nLocCoeffBlock(blocks.size(), 0);
            std::vector<std::vector<int>> locid0Block(blocks.size());
            std::vector<std::vector<int>> locid1Block(blocks.size());
            std::vector<std::vector<TData>> locsignBlock(blocks.size());
            m_nLocCoeffBlock =
                std::vector<std::vector<unsigned int>>(blocks.size());
            m_locid0 =
                std::vector<std::vector<MemoryRegion<int>>>(blocks.size());
            m_locid1 =
                std::vector<std::vector<MemoryRegion<int>>>(blocks.size());
            m_locsign =
                std::vector<std::vector<MemoryRegion<TData>>>(blocks.size());
            int i = 0, blk0 = 0, blk1 = 0, offset0 = 0, offset1 = 0;
            while (blk1 < blocks.size())
            {
                if (i == m_localDirSize ||
                    std::get<1>(locReordered[i]) >= blockBound[blk1])
                {
                    for (auto &nloc : nLocCoeffBlock)
                    {
                        m_nLocCoeffBlock[blk1].push_back(nloc);
                        nloc = 0;
                    }
                    for (auto &locid0 : locid0Block)
                    {
                        m_locid0[blk1].push_back(
                            MemoryRegion<int>::template FromVector<MemSpace,
                                                                   int>(
                                locid0, ExecSpace::alignment, device_only));
                        locid0.clear();
                    }
                    for (auto &locid1 : locid1Block)
                    {
                        m_locid1[blk1].push_back(
                            MemoryRegion<int>::template FromVector<MemSpace,
                                                                   int>(
                                locid1, ExecSpace::alignment, device_only));
                        locid1.clear();
                    }
                    for (auto &locsign : locsignBlock)
                    {
                        m_locsign[blk1].push_back(
                            MemoryRegion<TData>::template FromVector<MemSpace,
                                                                     TData>(
                                locsign, ExecSpace::alignment, device_only));
                        locsign.clear();
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
                    locid0Block[blk0].push_back(std::get<0>(locReordered[i]) -
                                                offset0);
                    locid1Block[blk0].push_back(std::get<1>(locReordered[i]) -
                                                offset1);
                    locsignBlock[blk0].push_back(std::get<2>(locReordered[i]));
                    nLocCoeffBlock[blk0]++;
                    offset0 = 0;
                    blk0    = 0;
                    i++;
                }
            }
        }

        // Split parDirBndSignReordered per block.
        if (m_nParDirBndSignSize > 0)
        {
            std::vector<int> parDirBndSignBlock;
            int i = 0, blk = 0, offset = 0, nParDirBndSignBlock = 0;
            while (blk < blocks.size())
            {
                if (i == m_nParDirBndSignSize ||
                    parDirBndSignReordered[i] >= blockBound[blk])
                {
                    m_nParDirBndSignBlock.push_back(nParDirBndSignBlock);
                    m_parDirBndSign.push_back(
                        MemoryRegion<int>::template FromVector<MemSpace, int>(
                            parDirBndSignBlock, ExecSpace::alignment,
                            device_only));
                    nParDirBndSignBlock = 0;
                    parDirBndSignBlock.clear();
                    offset = blockBound[blk];
                    blk++;
                }
                else
                {
                    parDirBndSignBlock.push_back(parDirBndSignReordered[i] -
                                                 offset);
                    nParDirBndSignBlock++;
                    i++;
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        // Return if no Dirichlet boundary condition.
        if (m_nBndCoeff == 0)
        {
            return;
        }

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            auto nbndCoeffBlock = m_nBndCoeffBlock[blk];

            if (nbndCoeffBlock > 0)
            {
                // Initialize pointers.
                auto inoutptr = inout.GetBlocks()[blk]
                                    .template GetPtr<MemSpace, ReadWrite>();
                auto mapPtr = m_map[blk].template GetPtr<MemSpace, ReadOnly>();
                auto bndcoeffPtr =
                    m_bndCoeff[blk].template GetPtr<MemSpace, ReadOnly>();
                const TData *signPtr =
                    m_signChange
                        ? m_sign[blk].template GetPtr<MemSpace, ReadOnly>()
                        : nullptr;

                // Add Dirichlet boundary conditions.
                if (m_signChange)
                {
                    DirBndCondKernel<ExecSpace>(nbndCoeffBlock, signPtr, mapPtr,
                                                bndcoeffPtr, inoutptr);
                }
                else
                {
                    DirBndCondKernel<ExecSpace>(nbndCoeffBlock, mapPtr,
                                                bndcoeffPtr, inoutptr);
                }
            }
        }

        if (m_nParDirBndSignSize > 0)
        {
            for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
            {
                auto nParDirBndSignBlock = m_nParDirBndSignBlock[blk];

                if (nParDirBndSignBlock > 0)
                {
                    // Initialize pointers.
                    auto inoutptr = inout.GetBlocks()[blk]
                                        .template GetPtr<MemSpace, ReadWrite>();
                    auto parDirBndSignPtr =
                        m_parDirBndSign[blk]
                            .template GetPtr<MemSpace, ReadOnly>();

                    ParallelDirBndSignKernel<ExecSpace>(
                        nParDirBndSignBlock, parDirBndSignPtr, inoutptr);
                }
            }
        }

        // TODO: Universal assembly on device.
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        if (contfield->GetSession()->GetComm()->GetRowComm()->GetSize() > 1)
        {
            // Copy the data from the input field.
            auto inoutarr = inout.template ToArray<NekDouble>();

            contfield->GetLocalToGlobalMap()->UniversalAbsMaxBnd(inoutarr);

            // Copy the data to the output field.
            inout.template CopyArray<MemSpace, NekDouble>(inoutarr);
        }

        if (m_nParDirBndSignSize > 0)
        {
            for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
            {
                auto nParDirBndSignBlock = m_nParDirBndSignBlock[blk];

                if (nParDirBndSignBlock > 0)
                {
                    // Initialize pointers.
                    auto inoutptr = inout.GetBlocks()[blk]
                                        .template GetPtr<MemSpace, ReadWrite>();
                    auto parDirBndSignPtr =
                        m_parDirBndSign[blk]
                            .template GetPtr<MemSpace, ReadOnly>();

                    ParallelDirBndSignKernel<ExecSpace>(
                        nParDirBndSignBlock, parDirBndSignPtr, inoutptr);
                }
            }
        }

        if (m_localDirSize > 0)
        {
            for (unsigned int blk1 = 0; blk1 < inout.GetBlocks().size(); ++blk1)
            {
                // Initialize pointers.
                auto inptr = inout.GetBlocks()[blk1]
                                 .template GetPtr<MemSpace, ReadOnly>();
                for (unsigned int blk0 = 0; blk0 < inout.GetBlocks().size();
                     ++blk0)
                {
                    auto nLocCoeffBlock = m_nLocCoeffBlock[blk1][blk0];

                    if (nLocCoeffBlock > 0)
                    {
                        // Initialize pointers.
                        auto outptr =
                            inout.GetBlocks()[blk0]
                                .template GetPtr<MemSpace, WriteOnly>();
                        auto locid0Ptr =
                            m_locid0[blk1][blk0]
                                .template GetPtr<MemSpace, ReadOnly>();
                        auto locid1Ptr =
                            m_locid1[blk1][blk0]
                                .template GetPtr<MemSpace, ReadOnly>();
                        auto locsignPtr =
                            m_locsign[blk1][blk0]
                                .template GetPtr<MemSpace, ReadOnly>();

                        LocalDirBndCondKernel<ExecSpace>(
                            nLocCoeffBlock, locid0Ptr, locid1Ptr, locsignPtr,
                            inptr, outptr);
                    }
                }
            }
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDirBndCondImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    std::vector<MemoryRegion<int>> m_map;
    std::vector<MemoryRegion<TData>> m_sign;
    std::vector<MemoryRegion<TData>> m_bndCoeff;
    std::vector<unsigned int> m_nBndCoeffBlock;
    std::vector<MemoryRegion<int>> m_parDirBndSign;
    std::vector<unsigned int> m_nParDirBndSignBlock;
    std::vector<std::vector<MemoryRegion<int>>> m_locid0;
    std::vector<std::vector<MemoryRegion<int>>> m_locid1;
    std::vector<std::vector<MemoryRegion<TData>>> m_locsign;
    std::vector<std::vector<unsigned int>> m_nLocCoeffBlock;

    unsigned int m_nBndCoeff          = 0;
    unsigned int m_nParDirBndSignSize = 0;
    unsigned int m_localDirSize       = 0;
    bool m_signChange;
};

} // namespace Nektar::Operators::detail
