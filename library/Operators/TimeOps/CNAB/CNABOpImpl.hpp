///////////////////////////////////////////////////////////////////////////////
//
// File: CNABOpImpl.hpp
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

#include "Operators/TimeOps/CNAB/CNABKernelLaunchers.hpp"
#include "Operators/TimeOps/CNAB/CNABOp.hpp"
#include "Operators/TimeOps/IMEX/IMEXOp.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData>
class CNABOpImpl : public CNABOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order.
    static_assert(IntOrder == 2,
                  "The CNABOp class is only implemented for order 2.");

public:
    CNABOpImpl(const ExpListSharedPtr &expansionList)
        : CNABOp<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const ExpListSharedPtr &expansionList)
    {
        return std::make_unique<CNABOpImpl<ExecSpace, Scheme, IntOrder, TData>>(
            expansionList);
    }

protected:
    static constexpr unsigned int Nimplicit(void)
    {
        if constexpr (std::is_same_v<Scheme, CNABScheme>)
        {
            return 1;
        }
        else if constexpr (std::is_same_v<Scheme, CNABModifiedScheme>)
        {
            return 2;
        }
    }

    static constexpr unsigned int Nexplicit(void)
    {
        return 2;
    }

    static constexpr TData gamma(void)
    {
        if constexpr (std::is_same_v<Scheme, CNABScheme>)
        {
            return 1.0 / 2.0;
        }
        else if constexpr (std::is_same_v<Scheme, CNABModifiedScheme>)
        {
            return 9.0 / 16.0;
        }
    }

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(this->m_explicitRhsFunctor,
                 "CNAB schemes require a DoExplicitRhs method. Define with "
                 "CNABOp->DefineExplicit().");
        ASSERTL0(this->m_projectionFunctor,
                 "CNAB schemes require a DoProjection method. Define with "
                 "CNABOp->DefineProjection().");
        ASSERTL0(this->m_implicitFunctor,
                 "CNAB schemes require a DoImplicit method. Define with "
                 "CNABOp->DefineImplicit().");

        // Startup.
        if (this->m_step + 1 <= Nimplicit())
        {
            // Initialise IMEX and hand-over the m_solutions deque.
            auto startup = IMEXOp<TData>::Create(this->m_expansionList, 1,
                                                 ExecSpace::name);
            startup->SaveImplicit(true);

            // Copy functors from outer/higher-order CNAB scheme.
            startup->CopyFunctorsFrom(*this);

            // Move solutions to startup.
            startup->SetImplicits(this->TakeImplicits());
            startup->SetExplicits(this->TakeExplicits());
            startup->SetSolutions(this->TakeSolutions());

            // Advance in time with startup.
            startup->SetTime(this->m_time);
            startup->SetStep(this->m_step);
            startup->Apply(inout);

            // Move solutions back to this CNAB.
            this->SetImplicits(startup->TakeImplicits());
            this->SetExplicits(startup->TakeExplicits());
            this->SetSolutions(startup->TakeSolutions());

            // Allocate new storage.
            if (this->m_step == 0 && Nimplicit() == 2)
            {
                this->m_implicits.push_back(
                    Field<TData, FieldState::Phys>::Create(
                        GetBlockAttributes<TData>(FieldState::Phys,
                                                  this->m_expansionList),
                        inout.GetNumComponents(), inout.GetNumHomoModes(),
                        ExecSpace::alignment));
            }

            // Increment step and time.
            this->m_time += this->m_timestep;
            this->m_step++;
        }
        // After startup.
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

            this->RollOver(this->m_explicits);

            // Ensure solution is in correct space.
            this->DoProjection(inout, inout, this->m_time);

            // Compute explicit term.
            this->DoExplicitRhs(inout, this->m_explicits[0], this->m_time,
                                this->m_timestep);

            // Do extrapolation.
            UpdateSolution(
                inout, std::make_integer_sequence<unsigned int, Nimplicit()>(),
                std::make_integer_sequence<unsigned int, Nexplicit()>());

            // Rollover previous solutions.
            this->RollOver(inout, this->m_implicits);

            // Update solution.
            this->DoImplicit(this->m_implicits[0], inout,
                             this->m_time + this->m_timestep,
                             gamma() * this->m_timestep);

            // Compute implicit terms.
            sub<ExecSpace>(inout, this->m_implicits[0], this->m_implicits[0]);
            mul<ExecSpace>(1.0 / gamma(), this->m_implicits[0],
                           this->m_implicits[0]);

            // Increment step and time.
            this->m_time += this->m_timestep;
            this->m_step++;
        }
    }

    template <unsigned int... ind, unsigned int... ind2>
    void UpdateSolution(Field<TData, FieldState::Phys> &inout,
                        std::integer_sequence<unsigned int, ind...>,
                        std::integer_sequence<unsigned int, ind2...>)
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
                (this->m_implicits[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>())...,
                (this->m_explicits[ind2]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>())...);
        }
    }
};

} // namespace Nektar::Operators::detail
