///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDGOp.hpp
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
// Description: Advection operator base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Common/Operator.hpp"
#include "SolverCore/Advection/AdvectionVolumeFluxOp.hpp"
#include "SolverCore/BndCond/BndCondUpdateOp.hpp"
#include "SolverCore/TraceFlux/TraceFluxOp.hpp"

namespace Nektar::SolverCore
{

// Advection operator base class
template <typename TData>
class AdvectionDGOp : public Operators::Operator<TData>
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
        this->m_scale = scale;
    }

    void SetAppend(const bool &append)
    {
        v_SetAppend(append);
    }

    /// The velocity is referenced, not adopted, so that it stays usable by
    /// whoever owns it; it must outlive this operator.
    void SetAdvVel(LibUtilities::Field<TData, FieldState::Phys> &advVel)
    {
        this->m_advVel = &advVel;
    }

    void SetTraceFlux(const std::shared_ptr<TraceFluxOp<TData>> &ptr)
    {
        this->m_traceFluxOp = ptr;
    }

    /**
     * @brief Attach an operator that computes boundary values from the
     * interior trace.
     *
     * The same conditions the advection-diffusion path takes, and the same
     * objects: an inviscid run needs the inflow and outflow states just as a
     * viscous one does. Only the thermal wall conditions are specific to the
     * viscous path, and those are claimed by their own operator, which an
     * Euler solver simply does not attach.
     */
    void AddBndCondUpdateOp(const std::shared_ptr<BndCondUpdateOp<TData>> &ptr)
    {
        ASSERTL0(this->m_traceFluxOp,
                 "A boundary condition was attached before the trace flux "
                 "operator that supplies its storage.");
        this->m_traceFluxOp->AddBndCondUpdateOp(ptr);
    }

    /// Refresh time-dependent boundary values before the next Apply(). A no-op
    /// unless a boundary condition actually depends on time.
    void UpdateBndPhys(const TData &time = 0.0)
    {
        if (this->m_traceFluxOp)
        {
            this->m_traceFluxOp->UpdateBndPhys(time);
        }
    }

    void SetVolumeFluxOp(
        const std::shared_ptr<AdvectionVolumeFluxOp<TData>> &ptr)
    {
        this->m_volumeFluxOp = ptr;
    }

protected:
    LibUtilities::Field<TData, FieldState::Phys> *m_advVel = nullptr;
    std::shared_ptr<TraceFluxOp<TData>> m_traceFluxOp;
    std::shared_ptr<AdvectionVolumeFluxOp<TData>> m_volumeFluxOp;
    TData m_scale = 1.0;
    bool m_append = false;

    AdvectionDGOp(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : Operators::Operator<TData>(expansionList, components)
    {
    }

    ~AdvectionDGOp() override = default;

    virtual void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                         LibUtilities::Field<TData, FieldState::Phys> &out) = 0;

    virtual void v_SetAppend(const bool &append) = 0;
};

} // namespace Nektar::SolverCore
