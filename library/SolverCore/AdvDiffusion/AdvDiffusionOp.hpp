///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffusionOp.hpp
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
// Description: AdvDiffusion operator base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Common/Operator.hpp"
#include "SolverCore/Advection/AdvectionVolumeFluxOp.hpp"
#include "SolverCore/BndCond/BndCondUpdateOp.hpp"
#include "SolverCore/Diffusion/DiffusionVolumeFluxOp.hpp"
#include "SolverCore/TraceFlux/TraceFluxOp.hpp"

namespace Nektar::SolverCore
{

// AdvDiffusion operator base class
template <typename TData>
class AdvDiffusionOp : public Operators::Operator<TData>
{
public:
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        this->v_Apply(in, out);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        this->v_Apply(in, out);
    }

    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

    void SetAppend(const bool &append)
    {
        v_SetAppend(append);
    }

    void SetAdvVolFlux(const std::shared_ptr<AdvectionVolumeFluxOp<TData>> &ptr)
    {
        m_advVolFluxOpNegOut = ptr;
    }

    void SetDiffVolFlux(
        const std::shared_ptr<DiffusionVolumeFluxOp<TData>> &ptr)
    {
        m_diffVolFluxOpAppend = ptr;
    }

    void SetAdvDiffTraceFlux(const std::shared_ptr<TraceFluxOp<TData>> &ptr)
    {
        m_advDiffTraceFluxOp = ptr;
    }

    /**
     * @brief Attach an operator that computes boundary values from the
     * interior trace, such as a no-slip wall state.
     *
     * Applied once the interior trace is available and before the trace flux
     * is evaluated, so the flux operator gathers the values it produced. The
     * storage it writes into is taken from the trace flux operator, so the two
     * necessarily agree.
     */
    void AddBndCondUpdateOp(const std::shared_ptr<BndCondUpdateOp<TData>> &ptr)
    {
        ASSERTL0(m_advDiffTraceFluxOp,
                 "A boundary condition was attached before the trace flux "
                 "operator that supplies its storage.");
        m_advDiffTraceFluxOp->AddBndCondUpdateOp(ptr);
    }

    /// Refresh time-dependent boundary values before the next Apply().
    void UpdateBndPhys(const TData &time = 0.0)
    {
        if (m_advDiffTraceFluxOp)
        {
            m_advDiffTraceFluxOp->UpdateBndPhys(time);
        }
    }

protected:
    std::shared_ptr<AdvectionVolumeFluxOp<TData>> m_advVolFluxOpNegOut;
    std::shared_ptr<DiffusionVolumeFluxOp<TData>> m_diffVolFluxOpAppend;
    std::shared_ptr<TraceFluxOp<TData>> m_advDiffTraceFluxOp;
    TData m_scale = 1.0;

    AdvDiffusionOp(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : Operators::Operator<TData>(expansionList, components)
    {
    }

    ~AdvDiffusionOp() override = default;

    virtual void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                         LibUtilities::Field<TData, FieldState::Phys> &out) = 0;

    /// Forward the append flag to whatever writes this operator's output.
    virtual void v_SetAppend(const bool &append) = 0;
};

} // namespace Nektar::SolverCore
