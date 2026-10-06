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

#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include "SolverCore/Diffusion/DiffusionIP/DiffusionIPOp.hpp"
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivOp.hpp>

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceOp.hpp>
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp>

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
          m_coefftmp(LibUtilities::Field<TData, FieldState::Coeff>(
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
              MultiRegions::GetLocTraceBlockAttributes<TData, FieldState::Phys>(
                  expansionList),
              components.size(), 1)),
          m_trace(LibUtilities::Field<TData, FieldState::Phys>(
              "Trace",
              MultiRegions::GetLocTraceBlockAttributes<TData, FieldState::Phys>(
                  expansionList),
              components.size(), 1)),
          m_traceDeriv(LibUtilities::Field<TData, FieldState::Phys>(
              "TraceDeriv",
              MultiRegions::GetLocTraceBlockAttributes<TData, FieldState::Phys>(
                  expansionList),
              expansionList->GetExp(0)->GetShapeDimension() * components.size(),
              1)),
          m_phystmp(LibUtilities::Field<TData, FieldState::Phys>(
              "Advect phys tmp",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList),
              components.size(), 1))
    {

        m_physDerivOp = MultiRegions::PhysDerivOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_bwdTransOp = MultiRegions::BwdTransOp<TData>::Create(
            expansionList, components, ExecSpace::name);

        m_multiplyByElmtInvMassOp =
            MultiRegions::MultiplyByElmtInvMassOp<TData>::Create(
                expansionList, components, ExecSpace::name);

        // Core operator building blocks used by the DG/IP diffusion path.
        m_physTraceExtractOp = MultiRegions::PhysTraceExtractOp<TData>::Create(
            expansionList, components, ExecSpace::name);

        m_iProductWRTDerivPhysOpNegOut = MultiRegions::IProductWRTDerivBaseOp<
            FieldState::Phys, TData>::Create(expansionList, components,
                                             ExecSpace::name);
        m_iProductWRTDerivPhysOpNegOut->SetScale(-1.0);

        m_iProductWRTPhysTraceOpAppend =
            MultiRegions::IProductWRTPhysTraceOp<TData>::Create(
                expansionList, components, ExecSpace::name);
        m_iProductWRTPhysTraceOpAppend->SetAppend(true);

        m_BTransposeOp = MultiRegions::IProductWRTBaseOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_BTransposeOp->SetIntegration(false);

        m_numflux.template Initialize<MemSpace>(TData(0.0));
    }

    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<DiffusionIPOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_coeff, m_coefftmp;
    LibUtilities::Field<TData, FieldState::Phys> m_deriv, m_fluxvector,
        m_numflux, m_trace, m_traceDeriv, m_phystmp;

    std::shared_ptr<MultiRegions::PhysTraceExtractOp<TData>>
        m_physTraceExtractOp;
    std::shared_ptr<MultiRegions::IProductWRTPhysTraceOp<TData>>
        m_iProductWRTPhysTraceOpAppend;
    std::shared_ptr<
        MultiRegions::IProductWRTDerivBaseOp<FieldState::Phys, TData>>
        m_iProductWRTDerivPhysOpNegOut;
    std::shared_ptr<MultiRegions::IProductWRTBaseOp<TData>> m_BTransposeOp;
    std::shared_ptr<MultiRegions::PhysDerivOp<TData>> m_physDerivOp;
    std::shared_ptr<MultiRegions::BwdTransOp<TData>> m_bwdTransOp;
    std::shared_ptr<MultiRegions::MultiplyByElmtInvMassOp<TData>>
        m_multiplyByElmtInvMassOp;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
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
    }

    void v_SetAppend(const bool &append) override
    {
        m_bwdTransOp->SetAppend(append);
    }

    void DiffuseCoeffs(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        ASSERTL1(this->m_volumeFluxOp,
                 "DiffusionIPOp requires a volume flux op.");
        ASSERTL1(this->m_traceFluxOp,
                 "DiffusionIPOp requires a trace flux op.");

        // Step 0: Extract trace from "in"
        m_physTraceExtractOp->Apply(in, m_trace);

        // The state trace is complete, so the neighbours can be sent theirs.
        // Channel 0; the gradient follows on channel 1 once step 2 has made
        // it, and both overlap the work between here and the wait below.
        this->m_traceFluxOp->BeginParallelExchange(m_trace, 0);

        // Step 1: compute physical derivatives of the conservative state.
        // These derivatives drive both the volume viscous tensor and the trace
        // numerical-flux construction.
        m_physDerivOp->Apply(in, m_deriv);

        // Step 2: Extract trace from "m_deriv"
        m_physTraceExtractOp->Apply(m_deriv, m_traceDeriv);

        // The gradient trace carries nDim times as many components as the
        // state, so this is much the larger of the two messages and has much
        // less local work left to hide behind - which is why it goes as soon
        // as it exists rather than alongside the first.
        this->m_traceFluxOp->BeginParallelExchange(m_traceDeriv, 1);

        // Step 3: assemble the viscous volume flux tensor F_v(q, grad q)
        this->m_volumeFluxOp->Apply(in, m_deriv, m_fluxvector);

        // Step 4: integrate the volume flux contribution against derivative
        // bases in physical space and negate output
        m_iProductWRTDerivPhysOpNegOut->Apply(m_fluxvector, m_phystmp);

        // Step 5: evaluate diffusion fluxes on the interior and boundary
        // traces, which need nothing from the exchanges, so this too overlaps
        // the messages.
        this->m_traceFluxOp->Apply(m_trace, m_traceDeriv, m_numflux);

        // Both exchanges have to have landed before the parallel traces' flux
        // is evaluated, and not before that.
        this->m_traceFluxOp->EndParallelExchange();
        this->m_traceFluxOp->ApplyParallel(m_trace, m_traceDeriv, m_numflux);

        // Step 6: Integral of numflux, B^T and Multiply by Inv Mass - could be
        // fused: zero<ExecSpace>(m_phystmp);
        m_iProductWRTPhysTraceOpAppend->Apply(m_numflux, m_phystmp);
        m_BTransposeOp->Apply(m_phystmp, m_coefftmp);
        m_multiplyByElmtInvMassOp->Apply(m_coefftmp, out);
        if (this->m_scale != 1.0)
        {
            Math::mul<ExecSpace>(this->m_scale, out, out);
        }
    }
};

} // namespace Nektar::SolverCore::detail
