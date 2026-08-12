///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionScalarIPTraceFluxOpImpl.hpp
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
// Description: Scalar IP diffusion trace flux implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ADRSolverRedesign/DiffusionScalarIPTraceFlux/DiffusionScalarIPTraceFluxKernels.hpp"
#include "ADRSolverRedesign/DiffusionScalarIPTraceFlux/DiffusionScalarIPTraceFluxOp.hpp"
#include "MultiRegions/Field/Math.hpp"
#include "Operators/Common/DataWarehouse/TraceDataWarehouse.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class DiffusionScalarIPTraceFluxOpImpl
    : public DiffusionScalarIPTraceFluxOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DiffusionScalarIPTraceFluxOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : DiffusionScalarIPTraceFluxOp<TData>(std::move(expansionList),
                                              components),
          m_traceAver(MultiRegions::Field<TData, FieldState::Phys>(
              "Scalar diffusion trace average",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_traceJump(MultiRegions::Field<TData, FieldState::Phys>(
              "Scalar diffusion trace jump",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_symmCoeff(MultiRegions::Field<TData, FieldState::Coeff>(
              "Scalar diffusion symmetric trace coeff",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components.size(), 1))
    {
        m_nDim  = expansionList->GetCoordim(0);
        m_nComp = components.size();

        const std::string execName = ExecSpace::name;
        m_traceInterleaveWidth =
            (execName == "AVX" || execName == "Device")
                ? NektarSpaces::GetVectorWidth<TData>(execName)
                : 1;

        std::vector<TData> diffCoeff(m_nDim * (m_nDim + 1) / 2, TData(0.0));
        for (unsigned int d = 0; d < m_nDim; ++d)
        {
            diffCoeff[d * (d + 3) / 2] = TData(1.0);
        }
        this->SetDiffCoeff(diffCoeff);

        BuildSymmetricTraceOffsets(expansionList);
    }

    static std::string className;

    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            DiffusionScalarIPTraceFluxOpImpl<ExecSpace, TData>>(expansionList,
                                                                components);
    }

protected:
    unsigned int m_nDim;
    unsigned int m_nComp;
    unsigned int m_traceInterleaveWidth;
    MultiRegions::Field<TData, FieldState::Phys> m_traceAver, m_traceJump;
    MultiRegions::Field<TData, FieldState::Coeff> m_symmCoeff;
    LibUtilities::MemoryRegion<TData> m_diffCoeff;
    std::vector<LibUtilities::MemoryRegion<unsigned int>> m_symmTraceBlockId;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_symmTraceOffset;
    std::vector<LibUtilities::MemoryRegion<unsigned int>> m_symmNqOffset;
    std::vector<unsigned int> m_symmNTraces;
    std::vector<unsigned int> m_symmNLocTracePts;

    void v_Apply(MultiRegions::Field<TData, FieldState::Phys> &fwd,
                 MultiRegions::Field<TData, FieldState::Phys> &bwd,
                 MultiRegions::Field<TData, FieldState::Phys> &derivFwd,
                 MultiRegions::Field<TData, FieldState::Phys> &derivBwd,
                 MultiRegions::Field<TData, FieldState::Phys> &out) override
    {
        for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &fwdblock       = fwd.GetBlocks()[blk];
            auto &bwdblock       = bwd.GetBlocks()[blk];
            auto &traceAverblock = m_traceAver.GetBlocks()[blk];
            auto &traceJumpblock = m_traceJump.GetBlocks()[blk];
            auto &outblock       = out.GetBlocks()[blk];

            auto normalbase = this->m_dataWarehouse->template GetData<MemSpace>(
                IPTraceNormalKey<TData>(blk));

            auto bwdWeightAverBase =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    IPTraceScalarKey<TData>(blk,
                                            IPTraceScalarData::BwdWeightAver));
            auto bwdWeightJumpBase =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    IPTraceScalarKey<TData>(blk,
                                            IPTraceScalarData::BwdWeightJump));
            auto lengthRecipBase =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    IPTraceScalarKey<TData>(blk,
                                            IPTraceScalarData::LengthRecip));
            auto penaltyFactorBase =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    IPTraceScalarKey<TData>(blk,
                                            IPTraceScalarData::PenaltyFactor));
            auto diffCoeffBase =
                m_diffCoeff.template GetPtr<MemSpace, ReadOnly>(streamID);

            auto fwdbase =
                fwdblock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto bwdbase =
                bwdblock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto derivTraceFwdbase =
                derivFwd.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID);
            auto derivTraceBwdbase =
                derivBwd.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID);
            auto traceAverbase =
                traceAverblock.template GetPtr<MemSpace, WriteOnly>(streamID);
            auto traceJumpbase =
                traceJumpblock.template GetPtr<MemSpace, WriteOnly>(streamID);
            auto outbase =
                outblock.template GetPtr<MemSpace, WriteOnly>(streamID);

            DiffuseScalarTraceFluxKernel<ExecSpace>(
                outblock.CompSize(), m_nDim, m_nComp, outblock.CompSize(),
                derivFwd.GetBlocks()[blk].CompSize(), this->m_IPPenaltyCoeff,
                diffCoeffBase, normalbase, bwdWeightAverBase, bwdWeightJumpBase,
                lengthRecipBase, penaltyFactorBase, fwdbase, bwdbase,
                derivTraceFwdbase, derivTraceBwdbase, traceAverbase,
                traceJumpbase, outbase, streamID);
        }
    }

    void v_SetDiffCoeff(std::vector<TData> &diffCoeff) override
    {
        m_diffCoeff = LibUtilities::MemoryRegion<TData>::template FromVector<
            MemSpace, TData>(diffCoeff);
    }

    void v_Apply(MultiRegions::Field<TData, FieldState::Coeff> &out) override
    {
        m_symmCoeff.template Initialize<MemSpace>(TData(0.0));

        auto diffCoeffBase = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            auto &outBlock = m_symmCoeff.GetBlocks()[blk];

            outBlock.template SetInterleaveWidth<TData>(
                out.GetBlocks()[blk].GetInterleaveWidth());

            const unsigned int nTraces      = m_symmNTraces[blk];
            const unsigned int nLocTracePts = m_symmNLocTracePts[blk];

            auto derivBaseTrace =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    IPTraceDerivBaseKey<TData>(blk));
            auto orientationMaps =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    OrientationMapsKey<TData>(blk, m_traceInterleaveWidth));
            auto orientationMapsOffset =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    OrientationMapsOffsetKey<TData>(blk,
                                                    m_traceInterleaveWidth));
            auto traceBlockId =
                m_symmTraceBlockId[blk].template GetPtr<MemSpace, ReadOnly>();
            auto traceOffset =
                m_symmTraceOffset[blk].template GetPtr<MemSpace, ReadOnly>();
            auto nqOffset =
                m_symmNqOffset[blk].template GetPtr<MemSpace, ReadOnly>();
            auto outBase = outBlock.template GetPtr<MemSpace, ReadWrite>();

            for (unsigned int traceBlk = 0;
                 traceBlk < m_traceJump.GetBlocks().size(); ++traceBlk)
            {
                const unsigned int streamID = traceBlk + 1;

                auto &jumpBlock = m_traceJump.GetBlocks()[traceBlk];
                auto traceJumpBlockBase =
                    jumpBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto traceNormalBlockBase =
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        IPTraceNormalKey<TData>(traceBlk));

                AddScalarSymmetricTraceFluxCoeffKernel<ExecSpace>(
                    outBlock.GetNumElements(),
                    outBlock.GetNumElementsWithPadding(), nTraces, nLocTracePts,
                    outBlock.GetNumData(), m_nDim, m_nComp, outBlock.CompSize(),
                    outBlock.GetInterleaveWidth(), traceBlk,
                    jumpBlock.CompSize(), diffCoeffBase, traceJumpBlockBase,
                    traceNormalBlockBase, traceBlockId, traceOffset, nqOffset,
                    orientationMaps, orientationMapsOffset, derivBaseTrace,
                    outBase, streamID);
            }
        }

        Math::add<ExecSpace>(out, m_symmCoeff, out);
    }

    void BuildSymmetricTraceOffsets(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        const auto blocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                expansionList);
        const auto traceBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList->GetTrace());

        std::vector<size_t> traceBlockOffset(1, 0);
        for (const auto &traceBlock : traceBlocks)
        {
            traceBlockOffset.push_back(traceBlockOffset.back() +
                                       traceBlock.GetNumElements() *
                                           traceBlock.GetNumData());
        }

        m_symmTraceBlockId.resize(blocks.size());
        m_symmTraceOffset.resize(blocks.size());
        m_symmNqOffset.resize(blocks.size());
        m_symmNTraces.resize(blocks.size());
        m_symmNLocTracePts.resize(blocks.size());

        for (unsigned int blk = 0; blk < blocks.size(); ++blk)
        {
            auto exp = GetCollection(expansionList, blk).GetExpVector()[0];
            const unsigned int nTraces = exp->GetNtraces();
            const size_t nelmt    = blocks[blk].GetNumElementsWithPadding();
            const size_t totTrace = nTraces * nelmt;

            auto locToTracePhysOffset =
                expansionList->GetDataWarehouseSharedPtr()
                    ->template GetData<NektarSpaces::HostSpace>(
                        LocToTracePhysOffsetKey<TData>(blk,
                                                       m_traceInterleaveWidth));

            std::vector<unsigned int> traceBlockId(totTrace, 0);
            std::vector<size_t> traceOffset(totTrace, 0);

            for (size_t i = 0; i < totTrace; ++i)
            {
                const auto offset = locToTracePhysOffset[i];

                unsigned int traceBlk = 0;
                while (traceBlk < traceBlocks.size() &&
                       !(offset >= traceBlockOffset[traceBlk] &&
                         offset < traceBlockOffset[traceBlk + 1]))
                {
                    traceBlk++;
                }
                ASSERTL1(traceBlk < traceBlocks.size(),
                         "Trace offset is outside trace storage.");

                const auto traceBlockLocalOffset =
                    offset - traceBlockOffset[traceBlk];

                traceBlockId[i] = traceBlk;
                traceOffset[i]  = traceBlockLocalOffset;
            }

            std::vector<unsigned int> nqOffset(nTraces, 0u);
            unsigned int offset = 0;
            for (unsigned int t = 0; t < nTraces; ++t)
            {
                nqOffset[t] = offset;
                offset += exp->GetTraceNumPoints(t);
            }
            m_symmNTraces[blk]      = nTraces;
            m_symmNLocTracePts[blk] = offset;

            m_symmTraceBlockId[blk] =
                LibUtilities::MemoryRegion<unsigned int>::template FromVector<
                    MemSpace, unsigned int>(traceBlockId);
            m_symmTraceOffset[blk] = LibUtilities::MemoryRegion<
                size_t>::template FromVector<MemSpace, size_t>(traceOffset);
            m_symmNqOffset[blk] =
                LibUtilities::MemoryRegion<unsigned int>::template FromVector<
                    MemSpace, unsigned int>(nqOffset);
        }
    }
};

} // namespace Nektar::Operators::detail
