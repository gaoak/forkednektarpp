///////////////////////////////////////////////////////////////////////////////
//
// File: FwdTransOpImpl.hpp
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

#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp"
#include "Operators/GlobalLinSysOps/LinearSystems/FwdTrans/FwdTransOp.hpp"

#include "Operators/Math/Math.hpp"

#include "MultiRegions/DisContField.h"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class FwdTransOpImpl : public FwdTransOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    FwdTransOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : FwdTransOp<TData>(expansionList, components)
    {
        auto session = this->m_expansionList->GetSession();

        if (session->DefinesSolverInfo("Projection"))
        {
            m_isDG = session->GetSolverInfo("Projection") == "DisContinuous";
        }
        else
        {
            m_isDG = std::dynamic_pointer_cast<MultiRegions::DisContField>(
                         this->m_expansionList) != nullptr;
        }

        this->m_IProdOp = IProductWRTBaseOp<TData>::Create(
            this->m_expansionList, components, ExecSpace::name);
        if (!m_isDG)
        {
            this->m_ElmtOp = MassOp<TData>::Create(this->m_expansionList,
                                                   components, ExecSpace::name);
            // Add default Dirichlet BC
            this->m_DirBCOps.push_back(DirBndCondOp<TData>::Create(
                this->m_expansionList, components, ExecSpace::name));
            this->m_RobBCOp = RobBndCondOp<TData>::Create(
                this->m_expansionList, components, ExecSpace::name);
        }
        else
        {
            m_MultiplyByElmtInvMassOp = MultiplyByElmtInvMassOp<TData>::Create(
                this->m_expansionList, components, ExecSpace::name);
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<FwdTransOpImpl<ExecSpace, TData>>(expansionList,
                                                                  components);
    }

protected:
    bool m_isDG = false;
    std::shared_ptr<MultiplyByElmtInvMassOp<TData>> m_MultiplyByElmtInvMassOp;

    void v_Apply(MultiRegions::Field<TData, FieldState::Phys> &in,
                 MultiRegions::Field<TData, FieldState::Coeff> &out) override
    {
        // IProductWRT of RHS.
        this->m_IProdOp->Apply(in, this->m_rhs);

        if (!m_isDG)
        {
            // Handle Dirichlet BCs.
            for (auto &dirBCOp : this->m_DirBCOps)
            {
                dirBCOp->Apply(out);
            }

            // Apply Mass operator.
            this->m_ElmtOp->Apply(out, this->m_tmp);

            // Handle Robin BCs.
            this->m_RobBCOp->Apply(out, this->m_tmp);

            // Solve linear system.
            sub<ExecSpace>(this->m_rhs, this->m_tmp, this->m_rhs);
            this->m_LinSolverOp->Apply(this->m_rhs, this->m_tmp);

            // Add Dirichlet BCs.
            add<ExecSpace>(out, this->m_tmp, out);
        }
        else
        {
            // DG forward transform is local: coeffs = M_e^{-1} (u, phi).
            m_MultiplyByElmtInvMassOp->Apply(this->m_rhs, out);
        }
    }
};

} // namespace Nektar::Operators::detail
