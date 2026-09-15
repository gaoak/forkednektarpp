//////////////////////////////////////////////////////////////////////////////
//
// File: LinearADRSolveOpImpl.hpp
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

#include "SolverCore/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp"

#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class LinearADRSolveOpImpl : public LinearADRSolveOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    LinearADRSolveOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : LinearADRSolveOp<TData>(expansionList, components)
    {
        this->m_ElmtOp = Operators::LinAdvDiffReactionOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
        this->m_IProdOp = Operators::IProductWRTBaseOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
        this->m_IProdOp->SetScale(-1.0);
        // Add default Dirichlet BC
        this->m_DirBCOps.push_back(Operators::DirBndCondOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name));
        // Add default Neumann BC
        this->m_NeuBCOps.push_back(Operators::NeuBndCondOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name));
        this->m_RobBCOp = Operators::RobBndCondOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<LinearADRSolveOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // IProductWRT of RHS.
        this->m_IProdOp->Apply(in, this->m_rhs);

        // Handle Neumann BCs on RHS.
        for (auto &neuBCOp : this->m_NeuBCOps)
        {
            neuBCOp->Apply(this->m_rhs);
        }

        // Handle Dirichlet BCs.
        for (auto &dirBCOp : this->m_DirBCOps)
        {
            dirBCOp->Apply(out);
        }

        // Apply ADR operator.
        this->m_ElmtOp->Apply(out, this->m_tmp);

        // Handle Robin BCs.
        this->m_RobBCOp->Apply(out, this->m_tmp);

        // Solve linear system.
        Math::sub<ExecSpace>(this->m_rhs, this->m_tmp, this->m_rhs);
        this->m_LinSolverOp->Apply(this->m_rhs, this->m_tmp);

        // Add Dirichlet BCs.
        Math::add<ExecSpace>(out, this->m_tmp, out);
    }
};

} // namespace Nektar::SolverCore::detail
