///////////////////////////////////////////////////////////////////////////////
//
// File: IMEXOpImpl.hpp
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

#include "Operators/TimeOps/IMEX/IMEXKernelLaunchers.hpp"
#include "Operators/TimeOps/IMEX/IMEXOp.hpp"
#include "Operators/TimeOps/IMEXdirk/IMEXdirkOp.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData>
class IMEXOpImpl : public IMEXOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order.
    static_assert(IntOrder >= 1 && IntOrder <= 4,
                  "The IMEXOp class is only implemented for order 1-4.");

public:
    IMEXOpImpl(const ExpListSharedPtr &expansionList)
        : IMEXOp<TData>(expansionList)
    {
        // Initialize coefficients at construction time.
        SetCoefficients();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const ExpListSharedPtr &expansionList)
    {
        return std::make_unique<IMEXOpImpl<ExecSpace, Scheme, IntOrder, TData>>(
            expansionList);
    }

protected:
    TData m_gamma;

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that implicit function call is defined.
        ASSERTL0(this->m_implicitFunctor,
                 "IMEX schemes require a DoImplicit method. Define with "
                 "IMEXOp->DefineImplicit().");

        // Check that explicit function is defined.
        ASSERTL0(this->m_explicitFunctor,
                 "IMEX schemes require a DoExplicit method. Define with "
                 "IMEXOp->DefineExplicit().");
        ASSERTL0(this->m_projectionFunctor,
                 "IMEX schemes require a DoProjection method. Define with "
                 "IMEXOp->DefineProjection().");

        // Startup.
        if (this->m_step + 1 < IntOrder)
        {
            // Allocate new storage.
            this->m_explicits.push_front(Field<TData, FieldState::Phys>::Create(
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                inout.GetNumComponents(), inout.GetNumHomoModes(),
                ExecSpace::alignment));

            // Compute explicit terms.
            this->DoExplicit(inout, this->m_explicits[0], this->m_time,
                             this->m_timestep);

            // Save initial solution.
            this->m_solutions.push_front(Field<TData, FieldState::Phys>::Create(
                "timestep n-" + std::to_string(this->m_step + 1),
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                inout.GetNumComponents(), inout.GetNumHomoModes(),
                ExecSpace::alignment));

            this->m_solutions[0].template Copy<MemSpace>(inout);

            // Initialise IMEXdirk.
            auto maxOrder = std::min(3u, IntOrder);
            std::string variant =
                std::to_string(maxOrder) + std::to_string(maxOrder + 1);
            auto startup = IMEXdirkOp<TData>::Create(
                this->m_expansionList, maxOrder, variant, ExecSpace::name);

            // Copy functors from outer/higher-order IMEX scheme.
            startup->CopyFunctorsFrom(*this);

            // Advance in time with startup.
            startup->SetTime(this->m_time);
            startup->SetStep(this->m_step);
            startup->Apply(inout);

            // Increment step and time.
            this->m_time += this->m_timestep;
            this->m_step++;
        }
        else
        {
            // Allocate new storage.
            if (this->m_explicits.size() < IntOrder)
            {
                this->m_explicits.push_back(
                    Field<TData, FieldState::Phys>::Create(
                        GetBlockAttributes<TData>(FieldState::Phys,
                                                  this->m_expansionList),
                        inout.GetNumComponents(), inout.GetNumHomoModes(),
                        ExecSpace::alignment));
            }

            // UpdateSolution previous solutions, explicit part, and sum up.
            if constexpr (IntOrder > 1)
            {
                // Rollover previous explicit parts.
                this->RollOver(this->m_explicits);
            }

            // Ensure solution is in correct space.
            this->DoProjection(inout, inout, this->m_time);

            // Compute explicit term.
            this->DoExplicit(inout, this->m_explicits[0], this->m_time,
                             this->m_timestep);

            if constexpr (IntOrder > 1)
            {
                // Rollover previous solutions.
                this->RollOver(inout, this->m_solutions);
            }

            // Do extrapolation.
            UpdateSolution(
                inout, std::make_integer_sequence<unsigned int, IntOrder>(),
                std::make_integer_sequence<unsigned int, IntOrder - 1>());

            // Compute next time step.
            if (this->m_save_implicit)
            {
                // Allocate new storage.
                if (this->m_implicits.size() < IntOrder)
                {
                    this->m_implicits.push_back(
                        Field<TData, FieldState::Phys>::Create(
                            GetBlockAttributes<TData>(FieldState::Phys,
                                                      this->m_expansionList),
                            inout.GetNumComponents(), inout.GetNumHomoModes(),
                            ExecSpace::alignment));
                }

                // Rollover previous solutions.
                this->RollOver(inout, this->m_implicits);

                // Update solution.
                this->DoImplicit(this->m_implicits[0], inout,
                                 this->m_time + this->m_timestep,
                                 m_gamma * this->m_timestep);

                // Compute implicit terms.
                sub<ExecSpace>(inout, this->m_implicits[0],
                               this->m_implicits[0]);
                mul<ExecSpace>(1.0 / m_gamma, this->m_implicits[0],
                               this->m_implicits[0]);
            }
            else
            {
                // Update solution.
                this->DoImplicit(inout, inout, this->m_time + this->m_timestep,
                                 m_gamma * this->m_timestep);
            }

            // Increment step and time.
            this->m_time += this->m_timestep;
            this->m_step++;
        }
    }

    template <unsigned int... Ind, unsigned int... Ind2>
    void UpdateSolution(Field<TData, FieldState::Phys> &inout,
                        std::integer_sequence<unsigned int, Ind...>,
                        std::integer_sequence<unsigned int, Ind2...>)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            // Determine shape and type of the element.
            auto &inoutBlock = inout.GetBlocks()[blk];
            auto nsize       = inoutBlock.GetNumElementsWithPadding() *
                         inoutBlock.GetNumData() *
                         inoutBlock.GetNumComponents() *
                         inoutBlock.GetNumHomoModes();

            // Compute new solution.
            UpdateSolutionKernel<ExecSpace, Scheme>(
                nsize, inoutBlock.template GetPtr<MemSpace, ReadWrite>(),
                (this->m_explicits[Ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>())...,
                (this->m_solutions[Ind2]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>())...);
        }
    }

    /*
     *  Setup gamma coefficient for IMEX.
     */
    void SetCoefficients()
    {
        if constexpr (IntOrder == 1)
        {
            m_gamma = 1.0;
        }
        else if constexpr (IntOrder == 2)
        {
            m_gamma = 2.0 / 3.0;
        }
        else if constexpr (IntOrder == 3)
        {
            m_gamma = 6.0 / 11.0;
        }
        else if constexpr (IntOrder == 4)
        {
            m_gamma = 12.0 / 25.0;
        }
    }
};

} // namespace Nektar::Operators::detail
