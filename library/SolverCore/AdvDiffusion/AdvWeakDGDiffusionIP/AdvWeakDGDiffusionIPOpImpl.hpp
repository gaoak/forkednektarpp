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
#include "SolverCore/AdvDiffusion/AdvDiffVolumeFluxOp.hpp"
#include "SolverCore/AdvDiffusion/AdvWeakDGDiffusionIP/AdvWeakDGDiffusionIPOp.hpp"
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivOp.hpp>

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceOp.hpp>
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief The weak DG / interior-penalty right-hand side from the separate
 * element operators: the state's trace and gradient, the volume fluxes and
 * their derivative inner product, the trace flux and its lifting, then the
 * projection through coefficient space.
 *
 * Under the average-of-normal-fluxes viscous trace form the trace flux takes
 * each element's outward normal viscous flux on its traces in place of the
 * gradient trace, which only the boundary traces then read.
 */
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

        // initialise internal fields for the first call
        m_numflux.template Initialize<MemSpace>(0.0);
    }

    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
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

    /// The viscous flux tensor alone, under the average-of-normal-fluxes
    /// form, for its normal component on the traces.
    LibUtilities::Field<TData, FieldState::Phys> m_viscFlux;
    /// Each element's outward normal viscous flux on its traces, under the
    /// average-of-normal-fluxes form: the second input of the trace flux.
    LibUtilities::Field<TData, FieldState::Phys> m_traceNormVisc;
    /// The viscous and total volume flux together, average-of-normal-fluxes
    /// form only: the attached one, or the composition of the advection and
    /// diffusion volume fluxes.
    std::shared_ptr<AdvDiffVolumeFluxOp<TData>> m_advDiffVolFlux;
    /// The trace flux takes the average-of-normal-fluxes viscous form.
    bool m_avgNormalFlux = false;
    /// This rank holds a boundary trace, which reads the gradient trace.
    bool m_hasBndTraces = false;

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
        ReshapeToWidth(out, 1u);
    }

    void v_SetAppend(const bool &append) override
    {
        m_bwdTransOp->SetAppend(append);
    }

    // Interleave width the AVX element operators work at. The volume fields
    // and the element-local traces are carried at it for the whole RHS, so
    // each operator's per-chunk reshape is a no-op, and the trace
    // gather/scatter addresses the traces at it directly. The caller's fields
    // enter and leave at width 1.
    static constexpr unsigned int m_rhsInterleaveWidth =
        std::is_same_v<ExecSpace, NektarSpaces::AVX>
            ? tinysimd::simd<TData>::width
            : 1u;

    template <typename TField>
    static void ReshapeToWidth(TField &field,
                               [[maybe_unused]] const unsigned int width)
    {
        if constexpr (m_rhsInterleaveWidth != 1u)
        {
            field.ReshapeStorage(width, ExecSpace::name);
        }
    }

    /// What follows from the attached operators, built on the first
    /// Apply() after they change.
    void Configure()
    {
        ASSERTL0(this->m_advVolFluxOpNegOut && this->m_diffVolFluxOpAppend &&
                     this->m_advDiffTraceFluxOp,
                 "AdvWeakDGDiffusionIPOp needs an advection volume flux, a "
                 "diffusion volume flux and a trace flux.");
        m_avgNormalFlux =
            this->m_advDiffTraceFluxOp->GetViscousTraceForm() ==
            TraceFluxOp<TData>::ViscousTraceForm::AverageOfNormalFluxes;
        m_hasBndTraces = this->m_advDiffTraceFluxOp->HasBoundaryTraces();
        if (m_avgNormalFlux)
        {
            const auto nvar = m_trace.GetNumComponents();
            const auto ndim =
                this->m_expansionList->GetExp(0)->GetShapeDimension();
            if (!m_viscFlux)
            {
                m_viscFlux = LibUtilities::Field<TData, FieldState::Phys>(
                    "Viscous flux",
                    MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                        this->m_expansionList),
                    ndim * nvar, 1);
                m_traceNormVisc = LibUtilities::Field<TData, FieldState::Phys>(
                    "Trace normal viscous flux",
                    MultiRegions::GetLocTraceBlockAttributes<TData,
                                                             FieldState::Phys>(
                        this->m_expansionList),
                    nvar, 1);
                m_traceNormVisc.template Initialize<MemSpace>(0.0);
            }
            if (this->m_advDiffVolFluxOp)
            {
                m_advDiffVolFlux = this->m_advDiffVolFluxOp;
            }
            else
            {
                auto composed = std::make_shared<
                    AdvDiffVolumeFluxComposed<ExecSpace, TData>>(
                    this->m_expansionList, this->m_components);
                composed->SetAdvVolFlux(this->m_advVolFluxOpNegOut);
                composed->SetDiffVolFlux(this->m_diffVolFluxOpAppend);
                m_advDiffVolFlux = composed;
            }
        }
        this->m_needsConfigure = false;
    }

    void AdvDiffCoeffs(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        if (this->m_needsConfigure)
        {
            Configure();
        }

        ReshapeToWidth(in, m_rhsInterleaveWidth);

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

        // Step 2: Extract trace from "m_deriv". Under the
        // average-of-normal-fluxes form only the boundary traces read it, and
        // the normal viscous flux is exchanged in its place (step 3).
        if (!m_avgNormalFlux || m_hasBndTraces)
        {
            m_physTraceExtractOp->Apply(m_deriv, m_traceDeriv);
        }

        // The gradient trace carries nDim times as many components as the
        // state, so this is much the larger of the two messages and has much
        // less local work left to hide behind - which is why it goes as soon
        // as it exists rather than alongside the first.
        if (!m_avgNormalFlux)
        {
            this->m_advDiffTraceFluxOp->BeginParallelExchange(m_traceDeriv, 1);
        }

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

        // Step 3: the volume flux, the viscous flux minus the advective one.
        if (m_avgNormalFlux)
        {
            // The viscous flux on its own as well, for each element's
            // outward normal viscous flux on its traces: the second input
            // of the trace flux and the one exchanged with the neighbours.
            // The boundary traces keep the flux-of-averages form and read
            // the gradient trace.
            m_advDiffVolFlux->Apply(in, m_deriv, m_viscFlux, m_fluxvector);
            m_physTraceExtractOp->ApplyNormalComponent(m_viscFlux,
                                                       m_traceNormVisc);
            this->m_advDiffTraceFluxOp->BeginParallelExchange(m_traceNormVisc,
                                                              1);
            this->m_advDiffTraceFluxOp->SetBoundaryGradientTrace(&m_traceDeriv);
        }
        else
        {
            // Assemble the negative of advection - the volume flux carries
            // its sign from SetAdvVolFlux() - and add the viscous volume
            // flux. Both operators are held through a shared pointer, so
            // another holder could have changed either setting since it was
            // attached: the advective term enters with the opposite sign to
            // the diffusive one, and the diffusive contribution accumulates
            // onto it.
            this->m_advVolFluxOpNegOut->SetScale(-1.0);
            this->m_diffVolFluxOpAppend->SetAppend(true);

            this->m_advVolFluxOpNegOut->Apply(in, m_fluxvector);
            this->m_diffVolFluxOpAppend->Apply(in, m_deriv, m_fluxvector);
        }
        ReshapeToWidth(in, 1u);

        // Step 4: integrate the volume flux contribution against derivative
        // bases in physical space and negate output
        m_iProductWRTDerivPhysOpNegOut->Apply(m_fluxvector, m_phystmp);

        // Step 5: evaluate negative of advection and diffusion trace fluxes
        // on the interior and boundary traces, which need nothing from the
        // exchanges, so this too overlaps the messages.
        // The trace flux writes m_numflux at the width of the traces. Its
        // second input is the gradient trace, or under the
        // average-of-normal-fluxes form the normal viscous flux.
        auto &traceSecond = m_avgNormalFlux ? m_traceNormVisc : m_traceDeriv;
        this->m_advDiffTraceFluxOp->Apply(m_trace, traceSecond, m_numflux);

        // Both exchanges have to have landed before the parallel traces' flux
        // is evaluated, and not before that.
        this->m_advDiffTraceFluxOp->EndParallelExchange();
        this->m_advDiffTraceFluxOp->ApplyParallel(m_trace, traceSecond,
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
