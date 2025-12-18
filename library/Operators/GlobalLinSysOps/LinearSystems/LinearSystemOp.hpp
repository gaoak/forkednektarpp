///////////////////////////////////////////////////////////////////////////////
//
// File: LinearSystemOp.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Common/Operator.hpp"

namespace Nektar::Operators
{

// LinearSystem base class
template <typename TData> class LinearSystemOp : public Operator<TData>
{
public:
    void Apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(in, out);
    }

    void operator()(Field<TData, FieldState::Phys> &in,
                    Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(in, out);
    }

    void SetLinearSolver(const std::shared_ptr<LinearSolverOp<TData>> &linsolve)
    {
        v_SetLinearSolver(linsolve);
    }

    void SetPrecon(const std::shared_ptr<PreconOp<TData>> &precon)
    {
        v_SetPrecon(precon);
    }

protected:
    LinearSystemOp(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    ~LinearSystemOp() override = default;

    virtual void v_Apply(Field<TData, FieldState::Phys> &in,
                         Field<TData, FieldState::Coeff> &out) = 0;

    virtual void v_SetLinearSolver(
        const std::shared_ptr<LinearSolverOp<TData>> &linsolve) = 0;

    virtual void v_SetPrecon(
        const std::shared_ptr<PreconOp<TData>> &precon) = 0;
};

} // namespace Nektar::Operators
