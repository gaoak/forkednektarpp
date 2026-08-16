///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionIPOpImpl.hpp
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
// Description: DiffusionIP Operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp"
#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivOp.hpp"
#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysOp.hpp"
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <MultiRegions/DataWarehouse/TraceDataWarehouse.hpp>
#include <Operators/AddTraceIntegral/AddTraceIntegralOp.hpp>
#include <Operators/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <SolverCore/Diffusion/DiffusionIP/DiffusionIPKernels.hpp>
#include <SolverCore/Diffusion/DiffusionIP/DiffusionIPOp.hpp>

#include <MultiRegions/AssemblyMap/AssemblyMapDG.h>
#include <MultiRegions/DisContField.h>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class DiffusionIPOpImpl : public DiffusionIPOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DiffusionIPOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components)
        : DiffusionIPOp<TData>(std::move(expansionList), components),
          m_coeff(LibUtilities::Field<TData, FieldState::Coeff>(
              "Diffusion coeff",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components.size(), 1)),
          m_tmp(LibUtilities::Field<TData, FieldState::Coeff>(
              "Diffusion tmp",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components.size(), 1)),
          m_deriv(LibUtilities::Field<TData, FieldState::Phys>(
              "Diffusion deriv",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList),
              expansionList->GetCoordim(0) * components.size(), 1)),
          m_fluxvector(LibUtilities::Field<TData, FieldState::Phys>(
              "Flux vector",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList),
              expansionList->GetCoordim(0) * components.size(), 1)),
          m_numflux(LibUtilities::Field<TData, FieldState::Phys>(
              "Num flux",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_fwd(LibUtilities::Field<TData, FieldState::Phys>(
              "Fwd Trace",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_bwd(LibUtilities::Field<TData, FieldState::Phys>(
              "Bwd Trace",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_derivTraceFwd(LibUtilities::Field<TData, FieldState::Phys>(
              "Deriv Fwd Trace",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              expansionList->GetCoordim(0) * components.size(), 1)),
          m_derivTraceBwd(LibUtilities::Field<TData, FieldState::Phys>(
              "Deriv Bwd Trace",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              expansionList->GetCoordim(0) * components.size(), 1))
    {
        m_nDim  = expansionList->GetCoordim(0);
        m_nComp = components.size();

        // Core operator building blocks used by the DG/IP diffusion path.
        m_physDerivOp = Operators::PhysDerivOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_bwdTransOp = Operators::BwdTransOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_iProductWRTDerivBaseOp =
            Operators::IProductWRTDerivBaseOp<FieldState ::Coeff,
                                              TData>::Create(expansionList,
                                                             components,
                                                             ExecSpace::name);
        m_getFwdBwdTracePhysOp = Operators::GetFwdBwdTracePhysOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        std::vector<std::string> derivComponents(m_nDim * m_nComp,
                                                 "DiffusionDerivTrace");
        m_getFwdBwdTraceDerivOp =
            Operators::GetFwdBwdTracePhysOp<TData>::Create(
                expansionList, derivComponents, ExecSpace::name);
        m_getFwdBwdTraceDerivOp->SetApplyDirBC(false);
        m_addTraceIntegralOp = Operators::AddTraceIntegralOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_multiplyByElmtInvMassOp =
            Operators::MultiplyByElmtInvMassOp<TData>::Create(
                expansionList, components, ExecSpace::name);

        m_iProductWRTDerivBaseOp->SetScale(-1.0);

        BuildDerivBndTraceOffsets(expansionList);
    }

    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<DiffusionIPOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    unsigned int m_nDim;
    unsigned int m_nComp;

    LibUtilities::Field<TData, FieldState::Coeff> m_coeff, m_tmp;
    LibUtilities::Field<TData, FieldState::Phys> m_deriv, m_fluxvector,
        m_numflux, m_fwd, m_bwd, m_derivTraceFwd, m_derivTraceBwd;
    std::shared_ptr<Operators::PhysDerivOp<TData>> m_physDerivOp;
    std::shared_ptr<Operators::BwdTransOp<TData>> m_bwdTransOp;
    std::shared_ptr<Operators::IProductWRTDerivBaseOp<FieldState::Coeff, TData>>
        m_iProductWRTDerivBaseOp;
    std::shared_ptr<Operators::GetFwdBwdTracePhysOp<TData>>
        m_getFwdBwdTracePhysOp;
    std::shared_ptr<Operators::GetFwdBwdTracePhysOp<TData>>
        m_getFwdBwdTraceDerivOp;
    std::shared_ptr<Operators::AddTraceIntegralOp<TData>> m_addTraceIntegralOp;
    std::shared_ptr<Operators::MultiplyByElmtInvMassOp<TData>>
        m_multiplyByElmtInvMassOp;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_derivBndTraceOffset;
    std::vector<size_t> m_numDerivBndTracePts;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        Diffuse(in, out);
    }

    void v_SetAppend(const bool &append) override
    {
        this->m_append = append;
        m_bwdTransOp->SetAppend(this->m_append);
    }

    void Diffuse(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        // High-level operator flow:
        // build the weak DG diffusion action in coefficient space, then map
        // that coefficient result back to physical space for the caller.
        DiffuseCoeffs(in, m_coeff);

        // The internal operator assembly lives in coefficient space, while the
        // public DiffusionOp interface returns a physical-space field. Append
        // is applied only at this final output stage so all internal workspaces
        // remain overwrite-style temporaries.
        m_bwdTransOp->Apply(m_coeff, out);

        // Loop over blocks to reshape output storage
        for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &outblock = out.GetBlocks()[blk];
            auto outptr =
                outblock.template GetPtr<MemSpace, WriteOnly>(streamID);

            // Reshape, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outblock.GetInterleaveWidth(),
                outblock.GetNumElementsWithPadding() *
                    outblock.GetNumComponents() * outblock.GetNumHomoModes(),
                outblock.GetNumData(), (TData *)outptr, streamID);

            // Set output block to new interleave.
            outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        }
    }

    void DiffuseCoeffs(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        ASSERTL1(this->m_volumeFluxOp,
                 "DiffusionIPOp requires a volume flux op.");
        ASSERTL1(this->m_traceFluxOp,
                 "DiffusionIPOp requires a trace flux op.");

        // Step 1: compute physical derivatives of the conservative state.
        // These derivatives drive both the volume viscous tensor and the trace
        // numerical-flux construction.
        m_physDerivOp->Apply(in, m_deriv);

        // Step 2: assemble the viscous volume flux tensor F_v(q, grad q)
        this->m_volumeFluxOp->Apply(in, m_deriv, m_fluxvector);

        // Step 3: integrate the volume flux contribution against derivative
        // bases to enter coefficient space.
        m_iProductWRTDerivBaseOp->Apply(m_fluxvector, m_tmp);

        // Step 4: add the interface numerical-flux contribution.
        CalcTraceNumFlux(in);
        m_addTraceIntegralOp->Apply(m_numflux, m_tmp);

        // Step 5: add the symmetric IP correction
        // ScalarIP implemented, coupled IP not implemented yet.
        this->m_traceFluxOp->Apply(m_tmp);

        // Step 6: apply the inverse element mass matrix to complete the weak DG
        // diffusion action in coefficient space.
        m_multiplyByElmtInvMassOp->Apply(m_tmp, out);

        if (this->m_scale != 1.0)
        {
            Math::mul<ExecSpace>(this->m_scale, out, out);
        }
    }

    void CalcTraceNumFlux(LibUtilities::Field<TData, FieldState::Phys> &in)
    {
        // This serial trace-flux path matches the legacy DiffusionIP trace
        // construction for periodic/no-special boundary treatment with
        // IP2ndDervCoeff == 0 and IPSymmFluxCoeff == 0. It does not include
        // special boundary average treatment, second-derivative correction, or
        // boundary trace-flux corrections such as WallAdiabatic.
        ASSERTL1(std::abs(this->m_IP2ndDervCoeff) < 1.0e-12,
                 "Serial DiffusionIPOp trace kernel currently supports "
                 "IP2ndDervCoeff == 0 only.");

        // Extract solution traces q^+ and q^- on element interfaces.
        m_getFwdBwdTracePhysOp->Apply(in, m_fwd, m_bwd);

        // Extract derivative traces dq^+ and dq^- in the same trace layout.
        // In the current serial path they are used directly with no exchange
        // or periodic correction stage.
        m_getFwdBwdTraceDerivOp->Apply(m_deriv, m_derivTraceFwd,
                                       m_derivTraceBwd);
        CopyBwdDerivTraceFromFwdOnBnd();

        // calculate the adverage and jump conditions
        this->m_traceFluxOp->Apply(m_fwd, m_bwd, m_derivTraceFwd,
                                   m_derivTraceBwd, m_numflux);
    }

    void AddSecondDerivToTrace(const Array<OneD, TData> &)
    {
        ASSERTL0(false, "AddSecondDerivToTrace is not available in the serial "
                        "field-only DiffusionIP path.");
    }

    void CopyBwdDerivTraceFromFwdOnBnd()
    {
        // Legacy DiffusionIP calls GetFwdBwdTracePhys(qfield, ..., true,
        // true, false) for derivative traces. The important BC behaviour is
        // PutFwdInBwdOnBCs=true: derivative components do not evaluate
        // physical BC expressions, so non-periodic boundary dq^- is set to
        // dq^+. Periodic derivative rotations/exchange are still outside this
        // simplified serial path.
        for (unsigned int blk = 0; blk < m_derivTraceBwd.GetBlocks().size();
             ++blk)
        {
            const unsigned int streamID = blk + 1;

            if (m_numDerivBndTracePts[blk] == 0)
            {
                continue;
            }

            auto &fwdBlk = m_derivTraceFwd.GetBlocks()[blk];
            auto &bwdBlk = m_derivTraceBwd.GetBlocks()[blk];

            auto fwdBase = fwdBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto bwdBase =
                bwdBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto offsetBase =
                m_derivBndTraceOffset[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID);

            CopyBwdDerivTraceFromFwdOnBndKernel<ExecSpace>(
                m_numDerivBndTracePts[blk], m_nDim * m_nComp, bwdBlk.CompSize(),
                offsetBase, fwdBase, bwdBase, streamID);
        }
    }

    void BuildDerivBndTraceOffsets(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        const auto discontField =
            std::dynamic_pointer_cast<MultiRegions::DisContField>(
                expansionList);
        const auto trace    = expansionList->GetTrace();
        const auto traceMap = expansionList->GetTraceMap();
        const auto traceBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(trace);

        m_derivBndTraceOffset.resize(traceBlocks.size());
        m_numDerivBndTracePts.assign(traceBlocks.size(), 0);
        std::vector<std::vector<size_t>> offsetsByBlock(traceBlocks.size());

        if (!discontField)
        {
            return;
        }

        std::vector<size_t> traceBlockOffset;
        traceBlockOffset.reserve(traceBlocks.size() + 1);
        traceBlockOffset.push_back(0);
        for (const auto &block : traceBlocks)
        {
            traceBlockOffset.push_back(traceBlockOffset.back() +
                                       block.GetNumElements() *
                                           block.GetNumData());
        }

        const auto &bndCondExpansions = discontField->GetBndCondExpansions();
        const auto &bndConditions     = discontField->GetBndConditions();

        for (unsigned int n = 0, cnt = 0; n < bndCondExpansions.size(); ++n)
        {
            const auto bndType = bndConditions[n]->GetBoundaryConditionType();
            const auto ne      = bndCondExpansions[n]->GetExpSize();

            if (bndType == SpatialDomains::ePeriodic)
            {
                continue;
            }
            ASSERTL1(bndType == SpatialDomains::eDirichlet ||
                         bndType == SpatialDomains::eNeumann ||
                         bndType == SpatialDomains::eRobin,
                     "Derivative trace boundary treatment is not available "
                     "for this boundary condition type.");

            for (unsigned int e = 0; e < ne; ++e)
            {
                const auto npts =
                    bndCondExpansions[n]->GetExp(e)->GetTotPoints();
                const auto traceId =
                    traceMap->GetBndCondIDToGlobalTraceID(cnt + e);
                const auto traceOffset = trace->GetPhys_Offset(traceId);

                unsigned int blk = 0;
                while (blk < traceBlocks.size() &&
                       !(traceOffset >= traceBlockOffset[blk] &&
                         traceOffset < traceBlockOffset[blk + 1]))
                {
                    ++blk;
                }
                ASSERTL1(blk < traceBlocks.size(),
                         "Boundary trace offset is outside trace storage.");

                const auto blockOffset = traceBlockOffset[blk];
                for (unsigned int p = 0; p < npts; ++p)
                {
                    offsetsByBlock[blk].push_back(traceOffset - blockOffset +
                                                  p);
                }
            }

            cnt += ne;
        }

        for (unsigned int blk = 0; blk < traceBlocks.size(); ++blk)
        {
            m_numDerivBndTracePts[blk] = offsetsByBlock[blk].size();
            m_derivBndTraceOffset[blk] =
                LibUtilities::MemoryRegion<size_t>::template FromVector<
                    MemSpace, size_t>(offsetsByBlock[blk]);
        }
    }
};

} // namespace Nektar::SolverCore::detail
