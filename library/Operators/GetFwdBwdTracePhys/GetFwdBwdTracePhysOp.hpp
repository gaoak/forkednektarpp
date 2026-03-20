///////////////////////////////////////////////////////////////////////////////
//
// File: GetFwdBwdTracePhysOp.hpp
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

#include "Operators/Common/Operator.hpp"

#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysBlockOp.hpp"

#include "Operators/BndCondOps/DGDirBndCond/DGDirBndCondOp.hpp"
#include "Operators/BndCondOps/DGPerBndCond/DGPerBndCondOp.hpp"

namespace Nektar::Operators
{

// GetFwdBwdTracePhys base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class GetFwdBwdTracePhysOp : public Operator<TData>
{
public:
    static std::shared_ptr<GetFwdBwdTracePhysOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? session->GetCmdLineArgument<std::string>("opExecSpace")
                : execStr;

        auto op = Operator<TData>::template Create<GetFwdBwdTracePhysOp>(
            expansionList, components, execStr0);

        auto traceBlockAttr = GetBlockAttributes<TData, FieldState::Phys>(
            expansionList->GetTrace());
        std::vector<size_t> traceBlockOffset;
        std::vector<size_t> traceTotOffset;
        std::vector<size_t> traceBlockSize;
        size_t blockOffset = 0;
        size_t totOffset   = 0;
        traceBlockOffset.push_back(blockOffset);
        traceTotOffset.push_back(totOffset);
        for (unsigned int traceBlk = 0; traceBlk < traceBlockAttr.size();
             ++traceBlk)
        {
            auto &traceBlock = traceBlockAttr[traceBlk];
            auto nelmt       = traceBlock.GetNumElements();
            auto nq          = traceBlock.GetNumData();
            blockOffset += nelmt * nq;
            totOffset += traceBlock.CompSize();
            traceBlockOffset.push_back(blockOffset);
            traceTotOffset.push_back(totOffset);
            traceBlockSize.push_back(traceBlock.CompSize());
        }

        auto blocks =
            GetBlockAttributes<TData, FieldState::Phys>(expansionList);
        auto traceBlocks = GetBlockAttributes<TData, FieldState::Phys>(
            expansionList->GetTrace());

        // Loop over the blocks.
        for (unsigned int block_idx = 0; block_idx < blocks.size(); block_idx++)
        {
            const auto exp_idx = GetCollection(expansionList, block_idx)
                                     .GetExpVector()[0]
                                     ->GetElmtId();
            const auto exp = expansionList->GetExp(exp_idx);

            op->m_blockOp.push_back(GetFwdBwdTracePhysBlockOp<TData>::Create(
                block_idx, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0));

            auto interleaveWidth =
                (execStr0 == "Device")
                    ? NektarSpaces::GetVectorWidth<TData>(execStr0)
                    : 1;
            auto locToTracePhysOffset =
                expansionList->GetDataWarehouseSharedPtr()
                    ->template GetData<NektarSpaces::HostSpace>(
                        LocToTracePhysOffsetKey<TData>(block_idx,
                                                       interleaveWidth));
            auto nComps    = components.size();
            auto nTraceBlk = traceBlocks.size();
            auto nTraces   = exp->GetNtraces();
            auto nelmt     = blocks[block_idx].GetNumElementsWithPadding();
            auto totTrace  = nTraces * nelmt;
            std::vector<size_t> tracePhysOffset(nComps * totTrace);
            for (unsigned int i = 0; i < totTrace; ++i)
            {
                auto offset = locToTracePhysOffset[i];

                // Find offset in which block
                unsigned int traceBlk = 0;
                while (traceBlk < nTraceBlk &&
                       !(offset >= traceBlockOffset[traceBlk] &&
                         offset < traceBlockOffset[traceBlk + 1]))
                {
                    traceBlk++;
                }

                for (unsigned int nc = 0; nc < nComps; ++nc)
                {
                    tracePhysOffset[nc * totTrace + i] =
                        offset - traceBlockOffset[traceBlk] +
                        nc * traceBlockSize[traceBlk] +
                        nComps * traceTotOffset[traceBlk];
                }
            }
            op->m_blockOp[block_idx]->SetTracePhysOffset(tracePhysOffset);
        }

        return op;
    }

    static inline const std::string name = "GetFwdBwdTracePhys";

    void Apply(Field<TData, FieldState::Phys> &phys,
               Field<TData, FieldState::Phys> &fwd,
               Field<TData, FieldState::Phys> &bwd)
    {
        v_Apply(phys, fwd, bwd); // v_Apply(in, out); out = [fwd bwd];
    }

    void operator()(Field<TData, FieldState::Phys> &phys,
                    Field<TData, FieldState::Phys> &fwd,
                    Field<TData, FieldState::Phys> &bwd)
    {
        v_Apply(phys, fwd, bwd);
    }

    void SetFwdOnly(bool fwdOnly)
    {
        m_fwdOnly = fwdOnly;
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetFwdOnly(fwdOnly);
        }
    }

protected:
    std::vector<std::shared_ptr<GetFwdBwdTracePhysBlockOp<TData>>> m_blockOp;
    std::shared_ptr<DGDirBndCondOp<TData>> m_DirBCOp;
    std::shared_ptr<DGPerBndCondOp<TData>> m_PerBCOp;

    bool m_fwdOnly = false;

    GetFwdBwdTracePhysOp(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
        m_DirBCOp = DGDirBndCondOp<TData>::Create(expansionList, components);
        m_PerBCOp = DGPerBndCondOp<TData>::Create(expansionList, components);
    }

    ~GetFwdBwdTracePhysOp() override = default;

    virtual void v_Apply(Field<TData, FieldState::Phys> &phys,
                         Field<TData, FieldState::Phys> &fwd,
                         Field<TData, FieldState::Phys> &bwd) = 0;
};

} // namespace Nektar::Operators
