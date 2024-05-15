///////////////////////////////////////////////////////////////////////////////
//
// File: ConjGradImplBase.hpp
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

#include "Operators/OperatorConjGrad.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ContField.h>

#include "Operators/OperatorAssmbScatr.hpp"
#include "Operators/OperatorRobBndCond.hpp"

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <memory>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorConjGradImplBase : public OperatorConjGrad<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorConjGradImplBase(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorConjGrad<TData>(expansionList),
          m_w_A(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad w_A",
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_s_A(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad s_A",
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_r_A(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad r_A",
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_wk(Field<TData, FieldState::Coeff>::template create<MemSpace>(
              "ConjGrad wk",
              GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        contfield->GetSession()->LoadParameter("NekLinSysMaxIterations",
                                               m_maxIter, 5000);
        contfield->GetSession()->LoadParameter("IterativeSolverTolerance",
                                               m_tol, 1.0E-09);

        m_nloc = contfield->GetLocalToGlobalMap()->GetNumLocalCoeffs();

        m_assmbScatrOp =
            AssmbScatr<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_robBndCondOp =
            RobBndCond<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);

        m_p_A = MemoryRegion<TData>::template create<MemSpace>("ConjGrad p_A",
                                                               m_nloc);
        m_q_A = MemoryRegion<TData>::template create<MemSpace>("ConjGrad q_A",
                                                               m_nloc);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatrOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_robBndCondOp;

    Field<TData, FieldState::Coeff> m_w_A;
    Field<TData, FieldState::Coeff> m_s_A;
    Field<TData, FieldState::Coeff> m_r_A;
    Field<TData, FieldState::Coeff> m_wk;

    MemoryRegion<TData> m_q_A;
    MemoryRegion<TData> m_p_A;

    TData m_tol;
    size_t m_nloc;
    size_t m_maxIter;
};

} // namespace Nektar::Operators::detail
