///////////////////////////////////////////////////////////////////////////////
//
// File: DGPerBndCondOpImpl.hpp
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

#include "Operators/BndCondOps/DGPerBndCond/DGPerBndCondOp.hpp"

#include "Operators/BndCondOps/DGPerBndCond/DGPerBndCondKernels.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class DGPerBndCondOpImpl : public DGPerBndCondOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DGPerBndCondOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                       const std::vector<std::string> &components)
        : DGPerBndCondOp<TData>(expansionList, components)
    {
        auto discontfield =
            std::dynamic_pointer_cast<DisContField>(this->m_expansionList);
        auto &bndCondExpansions = discontfield->GetBndCondExpansions();
        auto &bndConditions     = discontfield->GetBndConditions();
        auto &trace             = expansionList->GetTrace();
        auto periodicFwdCopy    = discontfield->GetPeriodicFwdCopy();
        auto periodicBwdCopy    = discontfield->GetPeriodicBwdCopy();

        // Compute number of boundary phys.
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::ePeriodic)
            {
                m_nBndPhys += bndCondExpansions[i]->GetTotPoints();
            }
        }

        // Return if no periodic boundary condition.
        if (m_nBndPhys == 0)
        {
            return;
        }

        // Compute trace block offset.
        auto blocks = GetBlockAttributes<TData, FieldState::Phys>(trace);
        std::vector<size_t> traceBlockOffset(blocks.size());
        std::vector<size_t> traceTotOffset(blocks.size());
        std::vector<size_t> traceBlockSize(blocks.size());
        size_t blockOffset = 0;
        size_t totOffset   = 0;
        traceBlockOffset.push_back(blockOffset);
        traceTotOffset.push_back(totOffset);
        for (unsigned int traceBlk = 0; traceBlk < blocks.size(); ++traceBlk)
        {
            auto &traceBlock = blocks[traceBlk];
            auto nelmt       = traceBlock.GetNumElements();
            auto ndata       = traceBlock.GetNumData();
            blockOffset += nelmt * ndata;
            totOffset += traceBlock.CompSize();
            traceBlockOffset.push_back(blockOffset);
            traceTotOffset.push_back(totOffset);
            traceBlockSize.push_back(traceBlock.CompSize());
        }

        auto nComps    = components.size();
        auto nTraceBlk = blocks.size();
        std::vector<size_t> periodicFwdCopyOffset(nComps * m_nBndPhys);
        std::vector<size_t> periodicBwdCopyOffset(nComps * m_nBndPhys);
        for (unsigned int i = 0; i < periodicFwdCopy.size(); ++i)
        {
            auto fwdCopy = periodicFwdCopy[i];
            auto bwdCopy = periodicBwdCopy[i];

            // Find fwdCopy in which block
            unsigned int fwdTraceBlk = 0;
            while (fwdTraceBlk < nTraceBlk &&
                   !(fwdCopy >= traceBlockOffset[fwdTraceBlk] &&
                     fwdCopy < traceBlockOffset[fwdTraceBlk + 1]))
            {
                fwdTraceBlk++;
            }

            // Find bwdCopy in which block
            unsigned int bwdTraceBlk = 0;
            while (bwdTraceBlk < nTraceBlk &&
                   !(bwdCopy >= traceBlockOffset[bwdTraceBlk] &&
                     bwdCopy < traceBlockOffset[bwdTraceBlk + 1]))
            {
                bwdTraceBlk++;
            }

            for (unsigned int nc = 0; nc < nComps; ++nc)
            {
                periodicFwdCopyOffset[nc * periodicFwdCopy.size() + i] =
                    fwdCopy - traceBlockOffset[fwdTraceBlk] +
                    nc * traceBlockSize[fwdTraceBlk] +
                    nComps * traceTotOffset[fwdTraceBlk];

                periodicBwdCopyOffset[nc * periodicBwdCopy.size() + i] =
                    bwdCopy - traceBlockOffset[bwdTraceBlk] +
                    nc * traceBlockSize[bwdTraceBlk] +
                    nComps * traceTotOffset[bwdTraceBlk];
            }
        }

        m_periodicFwdCopyOffset =
            MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                periodicFwdCopyOffset);
        m_periodicBwdCopyOffset =
            MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                periodicBwdCopyOffset);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<DGPerBndCondOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    MemoryRegion<size_t> m_periodicFwdCopyOffset;
    MemoryRegion<size_t> m_periodicBwdCopyOffset;
    size_t m_nBndPhys = 0;

    void v_Apply(Field<TData, FieldState::Phys> &in,
                 Field<TData, FieldState::Phys> &out) override
    {
        // Return if no Periodic boundary condition.
        if (m_nBndPhys == 0)
        {
            return;
        }

        const auto nTraceBlk = in.GetBlocks().size();
        auto &inBlock        = in.GetBlocks()[0];
        auto &outBlock       = out.GetBlocks()[0];
        auto inptr           = inBlock.template GetPtr<MemSpace, WriteOnly>();
        auto outptr          = outBlock.template GetPtr<MemSpace, WriteOnly>();
        size_t traceSize     = in.GetBlocks()[0].CompSize();

        // Synchronize memory for all blocks.
        for (unsigned int traceBlk = 1; traceBlk < nTraceBlk; ++traceBlk)
        {
            traceSize += in.GetBlocks()[traceBlk].CompSize();
            auto &inBlock  = in.GetBlocks()[traceBlk];
            auto &outBlock = out.GetBlocks()[traceBlk];
            inBlock.template GetPtr<MemSpace, WriteOnly>();
            outBlock.template GetPtr<MemSpace, WriteOnly>();
        }

        auto periodicFwdCopyOffsetPtr =
            m_periodicFwdCopyOffset.template GetPtr<MemSpace, ReadOnly>();

        auto periodicBwdCopyOffsetPtr =
            m_periodicBwdCopyOffset.template GetPtr<MemSpace, ReadOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < outBlock.GetNumComponents(); ++nc)
        {
            DGPerBndCondKernel<ExecSpace>(m_nBndPhys, periodicFwdCopyOffsetPtr,
                                          periodicBwdCopyOffsetPtr, nc, inptr,
                                          outptr);
        }
    }
};

} // namespace Nektar::Operators::detail
