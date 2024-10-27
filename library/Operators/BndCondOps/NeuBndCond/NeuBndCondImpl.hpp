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

#include "Operators/BndCondOps/OperatorNeuBndCond.hpp"

#include "Operators/BndCondOps/NeuBndCond/NeuBndCondCUDAKernels.cuh"
#include "Operators/BndCondOps/NeuBndCond/NeuBndCondKokkosKernels.hpp"
#include "Operators/BndCondOps/NeuBndCond/NeuBndCondSYCLKernels.hpp"
#include "Operators/BndCondOps/NeuBndCond/NeuBndCondSerialAVXKernels.hpp"

#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
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

        // Compute number boundary coefficients
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

        // Collecting boundary coefficients
        Array<OneD, TData> bndcoeff(m_nBndCoeff);
        Array<OneD, int> index(m_nBndCoeff);
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
                for (size_t j = 0; j < nBndExpCoeff; ++j)
                {
                    index[bndcnt + j] = cnt + j;
                }
                bndcnt += nBndExpCoeff;
            }
            cnt += nBndExpCoeff;
        }

        const bool device_only = true;

        m_bndCoeff = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            bndcoeff, ExecSpace::alignment, device_only);

        // Compute block bound.
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);
        Array<OneD, int> blockBound(blocks.size());
        int bound = 0;
        for (int block_idx = 0; block_idx < blocks.size(); ++block_idx)
        {
            const auto &block = blocks[block_idx];
            const auto ncoeff = block.num_pts;
            const auto nElmts = block.num_elements;
            bound += nElmts * ncoeff;
            blockBound[block_idx] = bound;
        }

        // Compute number of bndcoeff per block.
        Array<OneD, int> alignedMap(m_nBndCoeff);
        int block_idx = 0, offset = 0, nbndCoeffBlock = 0;
        for (int i = 0; i < m_nBndCoeff; i++)
        {
            while (map[index[i]] - offset > blockBound[block_idx])
            {
                offset = blockBound[block_idx];
                block_idx++;
                m_nBndCoeffBlock.push_back(nbndCoeffBlock);
                nbndCoeffBlock = 0;
            }
            alignedMap[i] = map[index[i]] - offset;
            nbndCoeffBlock++;
        }
        m_nBndCoeffBlock.push_back(nbndCoeffBlock);

        m_map = MemoryRegion<int>::template fromArray<MemSpace, int>(
            alignedMap, ExecSpace::alignment, device_only);

        if (m_signChange)
        {
            Array<OneD, TData> alignedSign(m_nBndCoeff);
            for (int i = 0; i < m_nBndCoeff; i++)
            {
                alignedSign[i] = sign[index[i]];
            }

            m_sign = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                alignedSign, ExecSpace::alignment, device_only);
        }
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        // Return if no Neumann boundary condition.
        if (m_nBndCoeff == 0)
        {
            return;
        }

        auto *mapPtr      = m_map.template GetPtr<MemSpace, ReadOnly>();
        auto *bndcoeffPtr = m_bndCoeff.template GetPtr<MemSpace, ReadOnly>();
        auto *inoutPtr    = inout.template GetPtr<MemSpace, ReadWrite>();
        auto *signPtr     = m_signChange
                                ? m_sign.template GetPtr<MemSpace, ReadOnly>()
                                : nullptr;

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < inout.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            auto &block         = inout.GetBlocks()[block_idx];
            auto nbndCoeffBlock = m_nBndCoeffBlock[block_idx];

            // Add weak boundary conditions to the forcing.
            if (m_signChange)
            {
                NeuBndCondKernel<ExecSpace, TData>(
                    nbndCoeffBlock, signPtr, mapPtr, bndcoeffPtr, inoutPtr);
            }
            else
            {
                NeuBndCondKernel<ExecSpace, TData>(nbndCoeffBlock, mapPtr,
                                                   bndcoeffPtr, inoutPtr);
            }

            // Increment pointer for the next block.
            signPtr += nbndCoeffBlock;
            bndcoeffPtr += nbndCoeffBlock;
            mapPtr += nbndCoeffBlock;
            inoutPtr += block.block_size;
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorNeuBndCondImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

protected:
    MemoryRegion<int> m_map;
    MemoryRegion<TData> m_sign;
    MemoryRegion<TData> m_bndCoeff;
    std::vector<size_t> m_nBndCoeffBlock;
    size_t m_nBndCoeff = 0;
    bool m_signChange;
};

} // namespace Nektar::Operators::detail
