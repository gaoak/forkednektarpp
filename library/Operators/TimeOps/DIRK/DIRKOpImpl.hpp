///////////////////////////////////////////////////////////////////////////////
//
// File: DIRKOpImpl.hpp
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

#include "Operators/TimeOps/DIRK/DIRKKernelLaunchers.hpp"
#include "Operators/TimeOps/DIRK/DIRKOp.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, unsigned int IntOrder, typename TData>
class DIRKOpImpl : public DIRKOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order
    static_assert(IntOrder >= 1 && IntOrder <= 3,
                  "The DIRKOp class is only implemented for order 1-3.");

public:
    DIRKOpImpl(const ExpListSharedPtr &expansionList)
        : DIRKOp<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const ExpListSharedPtr &expansionList)
    {
        return std::make_unique<DIRKOpImpl<ExecSpace, IntOrder, TData>>(
            expansionList);
    }

protected:
    static constexpr TData ConstSqrt2 = 1.414213562373095;
    static constexpr TData lambda2    = (2.0 - ConstSqrt2) / 2.0;
    static constexpr TData lambda3    = 0.4358665215;

    // clang-format off
    static constexpr TData m_coeff_t[3][3] = 
            {{1.0, 0.0, 0.0},
             {lambda2, 1.0, 0.0},
             {lambda3, (1.0 + lambda3) / 2.0, 1.0}};
    // clang-format on

    // clang-format off
    static constexpr TData m_lambda[3][3] =
            {{1.0, 0.0, 0.0},
             {lambda2, lambda2, 0.0},
             {lambda3, lambda3, lambda3}};
    // clang-format on

    static constexpr unsigned int NStage()
    {
        if constexpr (IntOrder == 1)
        {
            return 1;
        }
        else if constexpr (IntOrder == 2)
        {
            return 2;
        }
        else if constexpr (IntOrder == 3)
        {
            return 3;
        }
    }

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that implicit function is defined for DIRK
        ASSERTL0(this->m_implicitFunctor,
                 "DIRK schemes require a DoImplicit method. Define with "
                 "DIRKOp->DefineImplicit().");

        // Allocate memory
        if (this->m_solutions.size() == 0)
        {
            this->m_solutions.push_back(Field<TData, FieldState::Phys>::Create(
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                inout.GetNumComponents(), inout.GetNumHomoModes(),
                ExecSpace::alignment));
        }

        while (this->m_implicits.size() < NStage())
        {
            this->m_implicits.push_back(Field<TData, FieldState::Phys>::Create(
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                inout.GetNumComponents(), inout.GetNumHomoModes(),
                ExecSpace::alignment));
        }

        // Apply Runge-Kutta scheme
        this->m_solutions[0].template Copy<MemSpace>(inout);
        Staging<1>(inout);
        UpdateSolution(inout,
                       std::make_integer_sequence<unsigned int, NStage()>());

        // Increment step and time
        this->m_time += this->m_timestep;
        this->m_step++;
    }

    template <unsigned int Stage>
    void Staging(Field<TData, FieldState::Phys> &inout)
    {
        // Compute residual
        this->DoImplicit(inout, this->m_implicits[Stage - 1],
                         this->m_time + m_coeff_t[IntOrder - 1][Stage - 1] *
                                            this->m_timestep,
                         m_lambda[IntOrder - 1][Stage - 1] * this->m_timestep);
        sub<ExecSpace>(this->m_implicits[Stage - 1], inout,
                       this->m_implicits[Stage - 1]);
        mul<ExecSpace>(1.0 / m_lambda[IntOrder - 1][Stage - 1],
                       this->m_implicits[Stage - 1],
                       this->m_implicits[Stage - 1]);

        if constexpr (Stage < NStage())
        {
            // Compute stage
            UpdateStage(inout,
                        std::make_integer_sequence<unsigned int, Stage>());

            // Do next stage
            Staging<Stage + 1>(inout);
        }
    }

    template <unsigned int... ind>
    void UpdateStage(Field<TData, FieldState::Phys> &inout,
                     std::integer_sequence<unsigned int, ind...>)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            // Determine shape and type of the element.
            auto &inoutBlock = inout.GetBlocks()[blk];
            auto nelmt       = inoutBlock.GetNumElementsWithPadding();
            auto nphys =
                inoutBlock.GetNumData() * inoutBlock.GetNumComponents();

            UpdateStageKernel<ExecSpace, DIRKscheme, IntOrder>(
                nphys * nelmt,
                inoutBlock.template GetPtr<MemSpace, ReadWrite>(),
                this->m_solutions[0]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>(),
                (this->m_implicits[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>())...);
        }
    }

    template <unsigned int... ind>
    void UpdateSolution(Field<TData, FieldState::Phys> &inout,
                        std::integer_sequence<unsigned int, ind...>)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            // Determine shape and type of the element.
            auto &inoutBlock = inout.GetBlocks()[blk];
            auto nelmt       = inoutBlock.GetNumElementsWithPadding();
            auto nphys =
                inoutBlock.GetNumData() * inoutBlock.GetNumComponents();

            UpdateSolutionKernel<ExecSpace, DIRKscheme, IntOrder>(
                nphys * nelmt,
                inoutBlock.template GetPtr<MemSpace, ReadWrite>(),
                this->m_solutions[0]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>(),
                (this->m_implicits[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>())...);
        }
    }
};

} // namespace Nektar::Operators::detail
