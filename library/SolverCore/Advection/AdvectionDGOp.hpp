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

#include "MultiRegions/Field/Math.hpp"
#include "Operators/Common/Operator.hpp"
#include "SolverCore/Advection/AdvectionVolumeFluxOp.hpp"
#include "SolverCore/RiemannSolvers/RiemannSolverOp.hpp"

namespace Nektar::SolverCore
{

// Advection operator base class
template <typename TData>
class AdvectionDGOp : public Operators::Operator<TData>
{
public:
    void Apply(MultiRegions::Field<TData, FieldState::Phys> &in,
               MultiRegions::Field<TData, FieldState::Phys> &out)
    {
        this->v_Apply(in, out);
    }

    void operator()(MultiRegions::Field<TData, FieldState::Phys> &in,
                    MultiRegions::Field<TData, FieldState::Phys> &out)
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

    void SetAdvectVel(MultiRegions::Field<TData, FieldState::Phys> &advectVel)
    {
        this->m_advectVel = std::move(advectVel);
    }

    void SetRiemannSolver(const std::shared_ptr<RiemannSolverOp<TData>> &ptr)
    {
        this->m_riemannSolverOp = ptr;
    }

    void SetVolumeFluxOp(
        const std::shared_ptr<AdvectionVolumeFluxOp<TData>> &ptr)
    {
        this->m_volumeFluxOp = ptr;
    }

protected:
    MultiRegions::Field<TData, FieldState::Phys> m_advectVel;
    std::shared_ptr<RiemannSolverOp<TData>> m_riemannSolverOp;
    std::shared_ptr<AdvectionVolumeFluxOp<TData>> m_volumeFluxOp;
    TData m_scale = 1.0;
    bool m_append = false;

    AdvectionDGOp(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : Operators::Operator<TData>(expansionList, components)
    {
    }

    ~AdvectionDGOp() override = default;

    virtual void v_Apply(MultiRegions::Field<TData, FieldState::Phys> &in,
                         MultiRegions::Field<TData, FieldState::Phys> &out) = 0;

    virtual void v_SetAppend(const bool &append) = 0;
};

} // namespace Nektar::SolverCore
