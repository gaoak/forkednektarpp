///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionWeakDGOpImpl.hpp
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
// Description: AdvectionWeakDG Operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>

#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "SolverCore/Advection/AdvectionWeakDG/AdvectionWeakDGKernels.hpp"
#include "SolverCore/Advection/AdvectionWeakDG/AdvectionWeakDGOp.hpp"

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceOp.hpp>
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class AdvectionWeakDGOpImpl : public AdvectionWeakDGOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AdvectionWeakDGOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                          const std::vector<std::string> &components)
        : AdvectionWeakDGOp<TData>(std::move(expansionList), components),
          m_coeff(LibUtilities::Field<TData, FieldState::Coeff>(
              "Advect coeff",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components.size(), 1)),
          m_coefftmp(LibUtilities::Field<TData, FieldState::Coeff>(
              "Advect coeff tmp",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components.size(), 1)),
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
          m_phystmp(LibUtilities::Field<TData, FieldState::Phys>(
              "Advect phys tmp",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList),
              components.size(), 1))
    {
        m_bwdTransOp = MultiRegions::BwdTransOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_physTraceExtractOp = MultiRegions::PhysTraceExtractOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_iProductWRTDerivBaseOpNegOut = MultiRegions::IProductWRTDerivBaseOp<
            FieldState::Phys, TData>::Create(expansionList, components,
                                             ExecSpace::name);
        m_iProductWRTDerivBaseOpNegOut->SetScale(-1.0);
        m_iProductWRTPhysTraceOpAppend =
            MultiRegions::IProductWRTPhysTraceOp<TData>::Create(
                expansionList, components, ExecSpace::name);
        m_iProductWRTPhysTraceOpAppend->SetAppend(true);

        m_BTransposeOp = MultiRegions::IProductWRTBaseOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_BTransposeOp->SetIntegration(false);
        m_multiplyByElmtInvMassOp =
            MultiRegions::MultiplyByElmtInvMassOp<TData>::Create(
                expansionList, components, ExecSpace::name);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AdvectionWeakDGOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    LibUtilities::Field<TData, FieldState::Coeff> m_coeff, m_coefftmp;
    LibUtilities::Field<TData, FieldState::Phys> m_fluxvector, m_numflux,
        m_trace, m_phystmp;
    std::shared_ptr<MultiRegions::PhysTraceExtractOp<TData>>
        m_physTraceExtractOp;
    std::shared_ptr<MultiRegions::IProductWRTPhysTraceOp<TData>>
        m_iProductWRTPhysTraceOpAppend;
    std::shared_ptr<
        MultiRegions::IProductWRTDerivBaseOp<FieldState::Phys, TData>>
        m_iProductWRTDerivBaseOpNegOut;
    std::shared_ptr<MultiRegions::IProductWRTBaseOp<TData>> m_BTransposeOp;
    std::shared_ptr<MultiRegions::BwdTransOp<TData>> m_bwdTransOp;
    std::shared_ptr<MultiRegions::MultiplyByElmtInvMassOp<TData>>
        m_multiplyByElmtInvMassOp;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        Advect(in, out);
    }

    void v_SetAppend(const bool &append) override
    {
        this->m_append = append;
        m_bwdTransOp->SetAppend(this->m_append);
    }

    void Advect(LibUtilities::Field<TData, FieldState::Phys> &in,
                LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        AdvectCoeffs(in, m_coeff);

        m_bwdTransOp->Apply(m_coeff, out);
    }

    void AdvectCoeffs(LibUtilities::Field<TData, FieldState::Phys> &in,
                      LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        //  Extract trace from "in"
        m_physTraceExtractOp->Apply(in, m_trace);

        // The trace now holds this rank's side of every trace the partitioner
        // cut, so the neighbours can be sent theirs. Everything between here
        // and the matching wait below is local work the messages overlap with;
        // in serial both calls do nothing.
        this->m_traceFluxOp->BeginParallelExchange(m_trace);

        // Boundary values that are functions of the interior state - an
        // outflow extrapolating everything but the pressure, say - can only be
        // formed now that the trace exists, and must be in place before the
        // trace flux gathers them. The trace flux operator seeds its own
        // boundary storage with the interior state, since it owns the layout,
        // and the boundary operator then transforms that in place.
        this->m_traceFluxOp->UpdateBndCond(m_trace);

        // Compute volume flux and do IPWRTDB - could be fused
        this->m_volumeFluxOp->Apply(in, m_fluxvector);
        m_iProductWRTDerivBaseOpNegOut->Apply(m_fluxvector, m_phystmp);

        // GetTraces, interpolate, reorient, compute numerical flux,
        // reorient, interpolate back and put into m_numflux - the interior
        // and boundary traces only, which need nothing from the exchange, so
        // this too overlaps the messages.
        this->m_traceFluxOp->Apply(m_trace, m_numflux);

        // The numerical flux on a cut trace needs the neighbour's state, so
        // the exchange has to have landed before ApplyParallel() and not
        // before that.
        this->m_traceFluxOp->EndParallelExchange();
        this->m_traceFluxOp->ApplyParallel(m_trace, m_numflux);

        // Integral of numflux, B^T and Multiply by Inv Mass - could be fused:
        // zero<ExecSpace>(m_phystmp);
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
