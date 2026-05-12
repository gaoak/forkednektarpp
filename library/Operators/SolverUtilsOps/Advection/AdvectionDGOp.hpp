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
#include "Operators/Math/MathKernels.hpp"
#include "Operators/SolverUtilsOps/Advection/VolumeFluxOp.hpp"
#include "Operators/SolverUtilsOps/RiemannSolvers/RiemannSolverOp.hpp"

namespace Nektar::Operators
{

// Advection operator base class
template <typename TData> class AdvectionDGOp : public Operator<TData>
{
public:
    void Apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out)
    {
        this->v_Apply(in, out);
    }

    void operator()(Field<TData, FieldState::Phys> &in,
                    Field<TData, FieldState::Phys> &out)
    {
        this->v_Apply(in, out);
    }

    void SetScale(const TData &scale)
    {
        this->m_scale = scale;
    }

    void SetAdvectVel(Field<TData, FieldState::Phys> &advectVel)
    {
        this->m_advectVel = std::move(advectVel);
    }

    void SetRiemannSolver(const std::shared_ptr<RiemannSolverOp<TData>> &ptr)
    {
        this->m_riemannSolverOp = ptr;
    }

    void SetVolumeFluxOp(const std::shared_ptr<VolumeFluxOp<TData>> &ptr)
    {
        this->m_volumeFluxOp = ptr;
    }

protected:
    Field<TData, FieldState::Phys> m_advectVel;
    std::shared_ptr<RiemannSolverOp<TData>> m_riemannSolverOp;
    std::shared_ptr<VolumeFluxOp<TData>> m_volumeFluxOp;
    TData m_scale = 1.0;

    AdvectionDGOp(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    ~AdvectionDGOp() override = default;

    virtual void v_Apply(Field<TData, FieldState::Phys> &in,
                         Field<TData, FieldState::Phys> &out) = 0;
};

} // namespace Nektar::Operators
