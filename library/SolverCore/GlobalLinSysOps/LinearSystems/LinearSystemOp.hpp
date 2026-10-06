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

#include <MultiRegions/Common/Operator.hpp>

#include "SolverCore/GlobalLinSysOps/LinearSolvers/LinearSolverOp.hpp"
#include "SolverCore/PreconOps/PreconOp.hpp"
#include <MultiRegions/BndCondOps/DirBndCond/DirBndCondOp.hpp>
#include <MultiRegions/BndCondOps/NeuBndCond/NeuBndCondOp.hpp>
#include <MultiRegions/BndCondOps/RobBndCond/RobBndCondOp.hpp>
#include <MultiRegions/ElmtOps/ElmtOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>

namespace Nektar::SolverCore
{

// LinearSystem base class
template <typename TData>
class LinearSystemOp : public MultiRegions::Operator<TData>
{
public:
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(in, out);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(in, out);
    }

    void SetLinearSolver(const std::shared_ptr<LinearSolverOp<TData>> &linsolve)
    {
        m_LinSolverOp = linsolve;
        m_LinSolverOp->SetLHS(m_ElmtOp);
    }

    void SetPrecon(const std::shared_ptr<PreconOp<TData>> &precon)
    {
        m_LinSolverOp->SetPrecon(precon);
    }

    void UpdatePrecon(void)
    {
        m_LinSolverOp->UpdatePrecon();
    }

    void UpdateBndCoeffs(const TData &time = 0.0)
    {
        // Update default Dirichlet bc operator
        if (m_DirBCOps.size() > 0)
        {
            m_DirBCOps[0]->UpdateBndCoeffs(time);
        }

        // Update default Neumann bc operator
        if (m_NeuBCOps.size() > 0)
        {
            m_NeuBCOps[0]->UpdateBndCoeffs(time);
        }
    }

    void AddDirichletBCOp(
        const std::shared_ptr<MultiRegions::DirBndCondOp<TData>> &op)
    {
        m_DirBCOps.push_back(op);
    }

    void AddNeumannBCOp(
        const std::shared_ptr<MultiRegions::NeuBndCondOp<TData>> &op)
    {
        m_NeuBCOps.push_back(op);
    }

protected:
    std::shared_ptr<LinearSolverOp<TData>> m_LinSolverOp;
    std::shared_ptr<
        MultiRegions::ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>>
        m_ElmtOp;
    std::vector<std::shared_ptr<MultiRegions::DirBndCondOp<TData>>> m_DirBCOps;
    std::shared_ptr<MultiRegions::IProductWRTBaseOp<TData>> m_IProdOp;
    std::vector<std::shared_ptr<MultiRegions::NeuBndCondOp<TData>>> m_NeuBCOps;
    std::shared_ptr<MultiRegions::RobBndCondOp<TData>> m_RobBCOp;

    LibUtilities::Field<TData, FieldState::Coeff> m_rhs;
    LibUtilities::Field<TData, FieldState::Coeff> m_tmp;

    LinearSystemOp(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components),
          m_rhs(LibUtilities::Field<TData, FieldState::Coeff>(
              "LinearSystem RHS",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1)),
          m_tmp(LibUtilities::Field<TData, FieldState::Coeff>(
              "LinearSystem TMP",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components, 1))
    {
    }

    ~LinearSystemOp() override = default;

    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Coeff> &out) = 0;
};

} // namespace Nektar::SolverCore
