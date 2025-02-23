///////////////////////////////////////////////////////////////////////////////
//
// File: FwdTransImpl.hpp
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

#include "Operators/GlobalLinSysOps/ConjGrad/OperatorConjGrad.hpp"
#include "Operators/GlobalLinSysOps/FwdTrans/OperatorFwdTrans.hpp"

#include "Operators/BndCondOps/DirBndCond/OperatorDirBndCond.hpp"
#include "Operators/BndCondOps/RobBndCond/OperatorRobBndCond.hpp"
#include "Operators/ElmtOps/IProductWRTBase/OperatorIProductWRTBase.hpp"
#include "Operators/ElmtOps/Mass/OperatorMass.hpp"
#include "Operators/MathKernels/MathKernels.hpp"
#include "Operators/PreconOps/OperatorPrecon.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class OperatorFwdTransImpl : public OperatorFwdTrans<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorFwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorFwdTrans<TData>(expansionList),
          m_rhs(Field<TData, FieldState::Coeff>::template Create<MemSpace>(
              "FwdTrans RHS",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              ExecSpace::alignment)),
          m_tmp(Field<TData, FieldState::Coeff>::template Create<MemSpace>(
              "FwdTrans TMP",
              GetBlockAttributes<TData>(FieldState::Coeff, expansionList), 1,
              ExecSpace::alignment))
    {
        m_MassOp =
            OperatorMass<TData>::Create(this->m_expansionList, ExecSpace::name);
        m_DirBCOp = OperatorDirBndCond<TData>::Create(this->m_expansionList,
                                                      ExecSpace::name);
        m_RobBCOp = OperatorRobBndCond<TData>::Create(this->m_expansionList,
                                                      ExecSpace::name);
        m_IProdOp = OperatorIProductWRTBase<TData>::Create(
            this->m_expansionList, ExecSpace::name);
        m_CGOp = OperatorConjGrad<TData>::Create(this->m_expansionList,
                                                 ExecSpace::name);
        m_CGOp->setLHS(m_MassOp);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // IProductWRT of RHS
        m_IProdOp->apply(in, m_rhs);

        // Handle Dirichlet BCs
        m_DirBCOp->apply(out);
        m_MassOp->apply(out, m_tmp);
        sub<ExecSpace, TData>(m_rhs, m_tmp, m_rhs);

        // Handle Robin BCs
        m_RobBCOp->apply(out, m_rhs, true);

        // Solve for u_hat using Conjugate Gradient
        m_CGOp->apply(m_rhs, m_tmp);

        // Add Dirichlet BCs
        add<ExecSpace, TData>(out, m_tmp, out);
    }

    void setPrecon(
        const std::shared_ptr<OperatorPrecon<TData>> &precon) override
    {
        precon->configure(m_MassOp);

        m_CGOp->setPrecon(precon);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorFwdTransImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    std::shared_ptr<OperatorConjGrad<TData>> m_CGOp;
    std::shared_ptr<OperatorDirBndCond<TData>> m_DirBCOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProdOp;
    std::shared_ptr<OperatorMass<TData>> m_MassOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_RobBCOp;

    Field<TData, FieldState::Coeff> m_rhs;
    Field<TData, FieldState::Coeff> m_tmp;
};

} // namespace Nektar::Operators::detail
