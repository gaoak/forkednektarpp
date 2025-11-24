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

#include "Operators/BndCondOps/NeuBndCond/NeuBndCondOp.hpp"

#include "Operators/BndCondOps/NeuBndCond/NeuBndCondKernels.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class NeuBndCondOpImpl : public NeuBndCondOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NeuBndCondOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : NeuBndCondOp<TData>(expansionList)
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
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
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
        std::vector<size_t> blockBound(blocks.size());
        size_t bound = 0;
        for (unsigned int blk = 0; blk < blocks.size(); ++blk)
        {
            const auto &block = blocks[blk];
            const auto ncoeff = block.GetNumData();
            const auto nelmt  = block.GetNumElements();
            bound += nelmt * ncoeff;
            blockBound[blk] = bound;
        }

        // Collecting boundary coefficients.
        std::vector<TData> bndcoeff(m_nBndCoeff);
        std::vector<size_t> index(m_nBndCoeff);
        size_t bndcnt = 0, cnt = 0;
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
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
        std::vector<size_t> mapBlock;
        std::vector<TData> signBlock;
        unsigned int blk = 0;
        size_t i = 0, offset = 0, nbndCoeffBlock = 0;
        while (blk < blocks.size())
        {
            if (i == m_nBndCoeff || map[index[i]] >= blockBound[blk])
            {
                m_nBndCoeffBlock.push_back(nbndCoeffBlock);
                m_bndCoeff.push_back(
                    MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                        bndCoeffBlock));
                m_map.push_back(
                    MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                        mapBlock));
                if (m_signChange)
                {
                    m_sign.push_back(MemoryRegion<TData>::template FromVector<
                                     MemSpace, TData>(signBlock));
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

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<NeuBndCondOpImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    std::vector<MemoryRegion<size_t>> m_map;
    std::vector<MemoryRegion<TData>> m_sign;
    std::vector<MemoryRegion<TData>> m_bndCoeff;
    std::vector<size_t> m_nBndCoeffBlock;
    size_t m_nBndCoeff = 0;
    bool m_signChange;

    void v_Apply(Field<TData, FieldState::Coeff> &inout) override
    {
        // Return if no Neumann boundary condition.
        if (m_nBndCoeff == 0)
        {
            return;
        }

        ASSERTL0(inout.GetNumComponents() == 1,
                 "Not yet set up for multiple components");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            auto nbndCoeffBlk = m_nBndCoeffBlock[blk];

            if (nbndCoeffBlk == 0)
            {
                continue;
            }

            auto &inoutBlk = inout.GetBlocks()[blk];

            // Initialize pointers.
            auto inoutPtr = inoutBlk.template GetPtr<MemSpace, ReadWrite>();
            auto mapPtr   = m_map[blk].template GetPtr<MemSpace, ReadOnly>();
            auto bndcoeffPtr =
                m_bndCoeff[blk].template GetPtr<MemSpace, ReadOnly>();
            auto signPtr =
                m_signChange ? m_sign[blk].template GetPtr<MemSpace, ReadOnly>()
                             : nullptr;

            // if block is interlaced deInterleave block since currently mapping
            // set up assuming serial alignment
            auto inoutWidth = inoutBlk.GetInterleaveWidth();
            if (inoutWidth != 1)
            {
                deInterleave<ExecSpace>(inoutWidth,
                                        inoutBlk.GetNumElementsWithPadding() /
                                            inoutWidth,
                                        inoutBlk.GetNumData(), inoutPtr);
                inoutBlk.template SetInterleaveWidth<TData>(1);
            }

            // Add weak boundary conditions to the forcing.
            if (m_signChange)
            {
                NeuBndCondKernel<ExecSpace>(nbndCoeffBlk, signPtr, mapPtr,
                                            bndcoeffPtr, inoutPtr);
            }
            else
            {
                NeuBndCondKernel<ExecSpace>(nbndCoeffBlk, mapPtr, bndcoeffPtr,
                                            inoutPtr);
            }
        }
    }
};

} // namespace Nektar::Operators::detail
