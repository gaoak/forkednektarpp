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

#include "Operators/AddTraceIntegral/AddTraceIntegralOp.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransOp.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp"
#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp"
#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysOp.hpp"

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "SolverCore/Advection/AdvectionWeakDG/AdvectionWeakDGKernels.hpp"
#include "SolverCore/Advection/AdvectionWeakDG/AdvectionWeakDGOp.hpp"

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
          m_coeff(MultiRegions::Field<TData, FieldState::Coeff>(
              "Advet coeff",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components.size(), 1)),
          m_tmp(MultiRegions::Field<TData, FieldState::Coeff>(
              "Advet tmp",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components.size(), 1)),
          m_fluxvector(MultiRegions::Field<TData, FieldState::Phys>(
              "Flux vector",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList),
              expansionList->GetExp(0)->GetShapeDimension() * components.size(),
              1)),
          m_numflux(MultiRegions::Field<TData, FieldState::Phys>(
              "Num flux",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_fwd(MultiRegions::Field<TData, FieldState::Phys>(
              "Fwd Trace",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_bwd(MultiRegions::Field<TData, FieldState::Phys>(
              "Bwd Trace",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1))
    {
        m_bwdTransOp = Operators::BwdTransOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_iProductWRTDerivBaseOp =
            Operators::IProductWRTDerivBaseOp<FieldState::Coeff, TData>::Create(
                expansionList, components, ExecSpace::name);
        m_getFwdBwdTracePhysOp = Operators::GetFwdBwdTracePhysOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_addTraceIntegralOp = Operators::AddTraceIntegralOp<TData>::Create(
            expansionList, components, ExecSpace::name);
        m_multiplyByElmtInvMassOp =
            Operators::MultiplyByElmtInvMassOp<TData>::Create(
                expansionList, components, ExecSpace::name);

        m_iProductWRTDerivBaseOp->SetScale(-1.0);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AdvectionWeakDGOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;
    MultiRegions::Field<TData, FieldState::Coeff> m_coeff, m_tmp;
    MultiRegions::Field<TData, FieldState::Phys> m_fluxvector, m_numflux, m_fwd,
        m_bwd;
    std::shared_ptr<Operators::BwdTransOp<TData>> m_bwdTransOp;
    std::shared_ptr<Operators::IProductWRTDerivBaseOp<FieldState::Coeff, TData>>
        m_iProductWRTDerivBaseOp;
    std::shared_ptr<Operators::GetFwdBwdTracePhysOp<TData>>
        m_getFwdBwdTracePhysOp;
    std::shared_ptr<Operators::AddTraceIntegralOp<TData>> m_addTraceIntegralOp;
    std::shared_ptr<Operators::MultiplyByElmtInvMassOp<TData>>
        m_multiplyByElmtInvMassOp;

    void v_Apply(MultiRegions::Field<TData, FieldState::Phys> &in,
                 MultiRegions::Field<TData, FieldState::Phys> &out) override
    {
        Advect(in, out);
    }

    void v_SetAppend(const bool &append) override
    {
        this->m_append = append;
        m_bwdTransOp->SetAppend(this->m_append);
    }

    void Advect(MultiRegions::Field<TData, FieldState::Phys> &in,
                MultiRegions::Field<TData, FieldState::Phys> &out)
    {
        AdvectCoeffs(in, m_coeff);

        m_bwdTransOp->Apply(m_coeff, out);

        // Loop over blocks to reshape output storage
        for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &outblock = out.GetBlocks()[blk];
            auto outptr =
                outblock.template GetPtr<MemSpace, WriteOnly>(streamID);

            // Reshape, if necessary.
            Operators::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outblock.GetInterleaveWidth(),
                outblock.GetNumElementsWithPadding() *
                    outblock.GetNumComponents() * outblock.GetNumHomoModes(),
                outblock.GetNumData(), (TData *)outptr, streamID);

            // Set output block to new interleave.
            outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        }
    }

    void AdvectCoeffs(MultiRegions::Field<TData, FieldState::Phys> &in,
                      MultiRegions::Field<TData, FieldState::Coeff> &out)
    {
        // Compute interior flux vector
        this->m_volumeFluxOp->Apply(in, m_fluxvector);

        // Compute volume term contribution
        m_iProductWRTDerivBaseOp->Apply(m_fluxvector, m_tmp);

        // Compute numerical flux on element traces
        AdvectTraceFlux(in, m_numflux);

        // Add trace integral contribution
        m_addTraceIntegralOp->Apply(m_numflux, m_tmp);

        // Apply inverse mass matrix
        m_multiplyByElmtInvMassOp->Apply(m_tmp, out);

        if (this->m_scale != 1.0)
        {
            Math::mul<ExecSpace>(this->m_scale, out, out);
        }
    }

    void AdvectTraceFlux(MultiRegions::Field<TData, FieldState::Phys> &in,
                         MultiRegions::Field<TData, FieldState::Phys> &out)
    {

        // Get forward and backward trace values
        m_getFwdBwdTracePhysOp->Apply(in, m_fwd, m_bwd);

        // Compute numerical flux
        this->m_riemannSolverOp->Apply(m_fwd, m_bwd, out);
    }
};

} // namespace Nektar::SolverCore::detail
