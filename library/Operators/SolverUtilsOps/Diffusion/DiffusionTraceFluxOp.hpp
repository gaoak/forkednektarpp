///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionTraceFluxOp.hpp
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
// Description: DiffusionTrace flux operator base classes.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/SolverUtilsOps/Flux/FluxOp.hpp"

namespace Nektar::Operators
{

template <typename TData> class DiffusionTraceFluxOp : public FluxOp<TData>
{
public:
    // Build the physical numerical trace flux. For IP, this is where the
    // penalty term is added to the averaged normal diffusive flux.
    void Apply(MultiRegions::Field<TData, FieldState::Phys> &fwd,
               MultiRegions::Field<TData, FieldState::Phys> &bwd,
               MultiRegions::Field<TData, FieldState::Phys> &derivFwd,
               MultiRegions::Field<TData, FieldState::Phys> &derivBwd,
               MultiRegions::Field<TData, FieldState::Phys> &out)
    {
        this->v_Apply(fwd, bwd, derivFwd, derivBwd, out);
    }

    void operator()(MultiRegions::Field<TData, FieldState::Phys> &fwd,
                    MultiRegions::Field<TData, FieldState::Phys> &bwd,
                    MultiRegions::Field<TData, FieldState::Phys> &derivFwd,
                    MultiRegions::Field<TData, FieldState::Phys> &derivBwd,
                    MultiRegions::Field<TData, FieldState::Phys> &out)
    {
        this->v_Apply(fwd, bwd, derivFwd, derivBwd, out);
    }

    // Add the coefficient space symmetric IP trace term.
    // ScalarIP implemented, coupledIP not yet implemented.
    void Apply(MultiRegions::Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(out);
    }

    void operator()(MultiRegions::Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(out);
    }

protected:
    DiffusionTraceFluxOp(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : FluxOp<TData>(expansionList, components)
    {
    }

    ~DiffusionTraceFluxOp() override = default;

    virtual void v_Apply(MultiRegions::Field<TData, FieldState::Phys> &fwd,
                         MultiRegions::Field<TData, FieldState::Phys> &bwd,
                         MultiRegions::Field<TData, FieldState::Phys> &derivFwd,
                         MultiRegions::Field<TData, FieldState::Phys> &derivBwd,
                         MultiRegions::Field<TData, FieldState::Phys> &out) = 0;

    virtual void v_Apply(
        MultiRegions::Field<TData, FieldState::Coeff> &out) = 0;
};

template <typename TData>
class ScalarIPDiffusionTraceFluxOp : public DiffusionTraceFluxOp<TData>
{
public:
    void SetDiffCoeff(std::vector<TData> &diffCoeff)
    {
        this->v_SetDiffCoeff(diffCoeff);
    }

protected:
    ScalarIPDiffusionTraceFluxOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : DiffusionTraceFluxOp<TData>(expansionList, components)
    {
    }

    ~ScalarIPDiffusionTraceFluxOp() override = default;

    virtual void v_SetDiffCoeff(std::vector<TData> &diffCoeff) = 0;
};

} // namespace Nektar::Operators
