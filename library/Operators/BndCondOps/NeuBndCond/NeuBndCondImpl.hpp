///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondImpl.hpp
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

#include "Operators/BndCondOps/NeuBndCond/OperatorNeuBndCond.hpp"

#include "Operators/BndCondOps/NeuBndCond/NeuBndCondKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class OperatorNeuBndCondImpl : public OperatorNeuBndCond<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorNeuBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorNeuBndCond<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto &bndCondExpansions = contfield->GetBndCondExpansions();
        auto &bndConditions     = contfield->GetBndConditions();
        auto &assmbMap          = contfield->GetLocalToGlobalMap();
        auto &sign              = assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map               = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();
        m_signChange            = assmbMap->GetSignChange();

        // Compute number boundary coefficients.
        for (unsigned int i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eNeumann ||
                bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eRobin)
            {
                m_nBndCoeff += bndCondExpansions[i]->GetNcoeffs();
            }
        }

        // Return if no Neumann boundary condition.
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
                    SpatialDomains::eNeumann ||
                bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eRobin)
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

        // Compute number of bndcoeff per block.
        std::vector<TData> bndCoeffBlock;
        std::vector<int> mapBlock;
        std::vector<TData> signBlock;
        int i = 0, blk = 0, offset = 0, nbndCoeffBlock = 0;
        while (blk < blocks.size())
        {
            if (i == m_nBndCoeff || map[index[i]] >= blockBound[blk])
            {
                m_nBndCoeffBlock.push_back(nbndCoeffBlock);
                m_bndCoeff.push_back(
                    MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                        bndCoeffBlock, ExecSpace::alignment));
                m_map.push_back(
                    MemoryRegion<int>::template FromVector<MemSpace, int>(
                        mapBlock, ExecSpace::alignment));
                if (m_signChange)
                {
                    m_sign.push_back(
                        MemoryRegion<TData>::template FromVector<MemSpace,
                                                                 TData>(
                            signBlock, ExecSpace::alignment));
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
                bndCoeffBlock.push_back(bndcoeff[i]);
                mapBlock.push_back(map[index[i]] - offset);
                if (m_signChange)
                {
                    signBlock.push_back(sign[index[i]]);
                }
                nbndCoeffBlock++;
                i++;
            }
        }
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        // Return if no Neumann boundary condition.
        if (m_nBndCoeff == 0)
        {
            return;
        }

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            // Initialize pointers.
            auto inoutptr =
                inout.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>();
            auto mapPtr = m_map[blk].template GetPtr<MemSpace, ReadOnly>();
            auto bndcoeffPtr =
                m_bndCoeff[blk].template GetPtr<MemSpace, ReadOnly>();
            auto signPtr =
                m_signChange ? m_sign[blk].template GetPtr<MemSpace, ReadOnly>()
                             : nullptr;

            // Block dependent.
            auto nbndCoeffBlock = m_nBndCoeffBlock[blk];

            // Add weak boundary conditions to the forcing.
            if (m_signChange)
            {
                NeuBndCondKernel<ExecSpace>(nbndCoeffBlock, signPtr, mapPtr,
                                            bndcoeffPtr, inoutptr);
            }
            else
            {
                NeuBndCondKernel<ExecSpace>(nbndCoeffBlock, mapPtr, bndcoeffPtr,
                                            inoutptr);
            }
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorNeuBndCondImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    std::vector<MemoryRegion<int>> m_map;
    std::vector<MemoryRegion<TData>> m_sign;
    std::vector<MemoryRegion<TData>> m_bndCoeff;
    std::vector<unsigned int> m_nBndCoeffBlock;
    unsigned int m_nBndCoeff = 0;
    bool m_signChange;
};

} // namespace Nektar::Operators::detail
