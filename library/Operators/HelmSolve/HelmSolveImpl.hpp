///////////////////////////////////////////////////////////////////////////////
//
// File: HelmSolveImpl.hpp
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

#include "Operators/OperatorHelmSolve.hpp"

#include "Operators/MathKernels/MathKernels.hpp"
#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorDirBndCond.hpp"
#include "Operators/OperatorHelmholtz.hpp"
#include "Operators/OperatorHelper.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorNeuBndCond.hpp"
#include "Operators/OperatorPrecon.hpp"
#include "Operators/OperatorRobBndCond.hpp"

using vec_t = tinysimd::simd<double>;

namespace Nektar::Operators::detail
{

// CUDA implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorHelmSolveImpl : public OperatorHelmSolve<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmSolveImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmSolve<TData>(expansionList),
          m_rhs(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "HelmSolve RHS",
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment)),
          m_tmp(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "HelmSolve TMP",
              GetBlockAttributes(FieldState::Coeff, expansionList,
                                 vec_t::width),
              1, vec_t::alignment))
    {
        m_IProdOp =
            IProductWRTBase<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_DirBCOp =
            DirBndCond<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_NeuBCOp =
            NeuBndCond<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);

        if constexpr (std::is_same<ExecSpace, NektarSpaces::Serial>::value)
        {
            m_RobBCOp =
                RobBndCond<TData>::template create<ExecSpace, Implementation>(
                    this->m_expansionList);
        }

        m_HelmOp = Helmholtz<TData>::template create<ExecSpace, Implementation>(
            this->m_expansionList);
        m_CGOp = ConjGrad<TData>::template create<ExecSpace, Implementation>(
            this->m_expansionList);
        m_CGOp->setLHS(m_HelmOp);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // IProductWRT of RHS
        m_IProdOp->apply(in, m_rhs);

        neg<ExecSpace, TData>(m_rhs, m_rhs);

        // Handle Neumann BCs on RHS
        m_NeuBCOp->apply(m_rhs);

        // Handle Dirichlet BCs
        m_DirBCOp->apply(out);
        m_HelmOp->apply(out, m_tmp);

        sub<ExecSpace, TData>(m_rhs, m_tmp, m_rhs);

        // Handle Robin BCs
        if constexpr (std::is_same<ExecSpace, NektarSpaces::Serial>::value)
        {
            m_RobBCOp->apply(out, m_rhs, true);
        }

        // Solve using Conjugate Gradient
        m_CGOp->apply(m_rhs, m_tmp);

        // Add Dirichlet BCs
        add<ExecSpace, TData>(out, m_tmp, out);
    }

    void setLambda(const TData &lambda) override
    {
        m_HelmOp->setLambda(lambda);
    }

    void setPrecon(
        const std::shared_ptr<OperatorPrecon<TData>> &precon) override
    {
        precon->configure(m_HelmOp);

        m_CGOp->setPrecon(precon);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorHelmSolveImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorConjGrad<TData>> m_CGOp;
    std::shared_ptr<OperatorDirBndCond<TData>> m_DirBCOp;
    std::shared_ptr<OperatorHelmholtz<TData>> m_HelmOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProdOp;
    std::shared_ptr<OperatorNeuBndCond<TData>> m_NeuBCOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_RobBCOp;

    Field<TData, FieldState::Coeff> m_rhs;
    Field<TData, FieldState::Coeff> m_tmp;
};

} // namespace Nektar::Operators::detail
