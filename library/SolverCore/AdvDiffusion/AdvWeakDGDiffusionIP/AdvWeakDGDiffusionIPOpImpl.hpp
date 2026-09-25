///////////////////////////////////////////////////////////////////////////////
//
// File: AdvWeakDGDiffusionIPOpImpl.hpp
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
// Description: AdvWeakDGDiffusionIP Operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransOp.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp"
#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivOp.hpp"
#include "SolverCore/AdvDiffusion/AdvWeakDGDiffusionIP/AdvWeakDGDiffusionIPOp.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp"
#include "Operators/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceOp.hpp"
#include "Operators/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp"

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class AdvWeakDGDiffusionIPOpImpl : public AdvWeakDGDiffusionIPOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AdvWeakDGDiffusionIPOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : AdvWeakDGDiffusionIPOp<TData>(expansionList, components),
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
              expansionList->GetExp(0)->GetShapeDimension() * components.size(),
              1)),
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

        m_physDerivOp = Operators::PhysDerivOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_bwdTransOp = Operators::BwdTransOp<TData>::Create(
            expansionList, components, ExecSpace::name);

        m_multiplyByElmtInvMassOp =
            Operators::MultiplyByElmtInvMassOp<TData>::Create(
                expansionList, components, ExecSpace::name);

        // Core operator building blocks used by the DG/IP diffusion path.
        m_physTraceExtractOp = Operators::PhysTraceExtractOp<TData>::Create(
            expansionList, components, ExecSpace::name);

        m_iProductWRTDerivPhysOpNegOut =
            Operators::IProductWRTDerivBaseOp<FieldState::Phys, TData>::Create(
                expansionList, components, ExecSpace::name);
        m_iProductWRTDerivPhysOpNegOut->SetScale(-1.0);

        m_iProductWRTPhysTraceOpAppend =
            Operators::IProductWRTPhysTraceOp<TData>::Create(
                expansionList, components, ExecSpace::name);
        m_iProductWRTPhysTraceOpAppend->SetAppend(true);

        m_BTransposeOp = Operators::IProductWRTBaseOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_BTransposeOp->SetIntegration(false);

        // initialise internal fields for the first call
        m_numflux.template Initialize<MemSpace>(0.0);
    }

    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AdvWeakDGDiffusionIPOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_coeff, m_coefftmp;
    LibUtilities::Field<TData, FieldState::Phys> m_deriv, m_fluxvector,
        m_numflux, m_trace, m_traceDeriv, m_phystmp;

    std::shared_ptr<Operators::PhysTraceExtractOp<TData>> m_physTraceExtractOp;
    std::shared_ptr<Operators::IProductWRTPhysTraceOp<TData>>
        m_iProductWRTPhysTraceOpAppend;
    std::shared_ptr<Operators::IProductWRTDerivBaseOp<FieldState::Phys, TData>>
        m_iProductWRTDerivPhysOpNegOut;
    std::shared_ptr<Operators::IProductWRTBaseOp<TData>> m_BTransposeOp;
    std::shared_ptr<Operators::PhysDerivOp<TData>> m_physDerivOp;
    std::shared_ptr<Operators::BwdTransOp<TData>> m_bwdTransOp;
    std::shared_ptr<Operators::MultiplyByElmtInvMassOp<TData>>
        m_multiplyByElmtInvMassOp;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        // High-level operator flow:
        // build the weak DG diffusion action in coefficient space, then map
        // that coefficient result back to physical space for the caller.
        AdvDiffCoeffs(in, m_coeff);

        // The internal operator assembly lives in coefficient space, while the
        // public AdvWeakDGDiffusionIPOp interface returns a physical-space
        // field. Append is applied only at this final output stage so all
        // internal workspaces remain overwrite-style temporaries.
        m_bwdTransOp->Apply(m_coeff, out);
    }

    void v_SetAppend(const bool &append) override
    {
        m_bwdTransOp->SetAppend(append);
    }

    void AdvDiffCoeffs(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        ASSERTL1(
            this->m_advVolFluxOpNegOut,
            "AdvWeakDGDiffusionIPOp requires an advection volume flux op.");
        ASSERTL1(this->m_diffVolFluxOpAppend,
                 "AdvWeakDGDiffusionIPOp requires a diffusion volume flux op.");
        ASSERTL1(this->m_advDiffTraceFluxOp,
                 "AdvWeakDGDiffusionIPOp requires a trace flux op.");

        // Step 0: Extract trace from "in"
        m_physTraceExtractOp->Apply(in, m_trace);

        // The state trace is complete, so the neighbours can be sent theirs.
        // Channel 0; the gradient follows on channel 1 once step 2 has made
        // it, and both overlap the local work between here and the wait
        // before step 5. In serial every one of these calls does nothing.
        this->m_advDiffTraceFluxOp->BeginParallelExchange(m_trace, 0);

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
        this->m_advDiffTraceFluxOp->BeginParallelExchange(m_traceDeriv, 1);

        // Step 2b: boundary values that are functions of the interior state -
        // a no-slip wall, say - can only be formed now that the trace exists,
        // and must be in place before the trace flux gathers them in step 5.
        // The trace flux operator seeds its own boundary storage with the
        // interior state, since it owns the layout, and the boundary operator
        // then transforms that in place. It touches only the boundary blocks,
        // which neither exchange reads, so it sits after both begins and
        // overlaps both messages; the derivative work above cannot, since the
        // gradient message waits on it.
        this->m_advDiffTraceFluxOp->UpdateBndCond(m_trace);

        // Step 3: assemble the negative of advection - the volume flux carries
        // its sign from SetAdvVolFlux() - and add the viscous volume flux
        // Both operators are held through a shared pointer, so another
        // holder could have changed either setting since it was attached:
        // the advective term enters with the opposite sign to the diffusive
        // one, and the diffusive contribution accumulates onto it.
        this->m_advVolFluxOpNegOut->SetScale(-1.0);
        this->m_diffVolFluxOpAppend->SetAppend(true);

        this->m_advVolFluxOpNegOut->Apply(in, m_fluxvector);
        this->m_diffVolFluxOpAppend->Apply(in, m_deriv, m_fluxvector);

        // Step 4: integrate the volume flux contribution against derivative
        // bases in physical space and negate output
        m_iProductWRTDerivPhysOpNegOut->Apply(m_fluxvector, m_phystmp);

        // Step 5: evaluate negative of advection and diffusion trace fluxes
        // on the interior and boundary traces, which need nothing from the
        // exchanges, so this too overlaps the messages.
        this->m_advDiffTraceFluxOp->Apply(m_trace, m_traceDeriv, m_numflux);

        // Both exchanges have to have landed before the parallel traces' flux
        // is evaluated, and not before that.
        this->m_advDiffTraceFluxOp->EndParallelExchange();
        this->m_advDiffTraceFluxOp->ApplyParallel(m_trace, m_traceDeriv,
                                                  m_numflux);

        // Step 6: Integral of numflux, B^T and Multiply by Inv Mass - (could be
        // fused):
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
