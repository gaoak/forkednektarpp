///////////////////////////////////////////////////////////////////////////////
//
// File: AUSM0SolverOpImpl.hpp
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
// Description: AUSM0 Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "RiemannSolvers/AUSMSolver/AUSM0Solver/AUSM0SolverOp.hpp"

#include "Operators/Utils/UtilsKernels.hpp"
#include "RiemannSolvers/AUSMSolver/AUSM0Solver/AUSM0SolverKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class AUSM0SolverOpImpl : public AUSM0SolverOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AUSM0SolverOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components)
        : AUSM0SolverOp<TData>(expansionList, components)
    {
        m_dimension = expansionList->GetExp(0)->GetShapeDimension();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AUSM0SolverOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_dimension;

    void v_Apply(Field<TData, FieldState::Phys> &Fwd,
                 Field<TData, FieldState::Phys> &Bwd,
                 Field<TData, FieldState::Phys> &flux) override
    {
        switch (m_dimension)
        {
            // 1D
            case 1:
            {
                this->template OperatorND<AUSM0SolverKernel, ExecSpace, 1>(
                    Fwd, Bwd, flux);
                break;
            }
            // 2D
            case 2:
            {
                this->template OperatorND<AUSM0SolverKernel, ExecSpace, 2>(
                    Fwd, Bwd, flux);
                break;
            }
            // 3D
            case 3:
            {
                this->template OperatorND<AUSM0SolverKernel, ExecSpace, 3>(
                    Fwd, Bwd, flux);
                break;
            }
        }
    }
};

} // namespace Nektar::Operators::detail
