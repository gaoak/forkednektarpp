//////////////////////////////////////////////////////////////////////////////
//
// File: HelmSolveOpImpl.hpp
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

#include "Operators/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp"

#include "Operators/Math/MathKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class HelmSolveOpImpl : public HelmSolveOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    HelmSolveOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : HelmSolveOp<TData>(expansionList, components)
    {
        this->m_ElmtOp = HelmholtzOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
        this->m_IProdOp = IProductWRTBaseOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
        this->m_IProdOp->SetScale(-1.0);
        this->m_DirBCOp = DirBndCondOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
        this->m_NeuBCOp = NeuBndCondOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
        this->m_RobBCOp = RobBndCondOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<HelmSolveOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    void v_Apply(Field<TData, FieldState::Phys> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // IProductWRT of RHS.
        this->m_IProdOp->Apply(in, this->m_rhs);

        // Handle Neumann BCs on RHS.
        this->m_NeuBCOp->Apply(this->m_rhs);

        // Handle Dirichlet BCs.
        this->m_DirBCOp->Apply(out);

        // Apply Helmholtz operator.
        this->m_ElmtOp->Apply(out, this->m_tmp);

        // Handle Robin BCs.
        this->m_RobBCOp->Apply(out, this->m_tmp);

        // Solve linear system.
        sub<ExecSpace>(this->m_rhs, this->m_tmp, this->m_rhs);
        this->m_LinSolverOp->Apply(this->m_rhs, this->m_tmp);

        // Add Dirichlet BCs.
        add<ExecSpace>(out, this->m_tmp, out);
    }
};

} // namespace Nektar::Operators::detail
