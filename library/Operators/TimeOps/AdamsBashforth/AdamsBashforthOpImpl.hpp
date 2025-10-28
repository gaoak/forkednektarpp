///////////////////////////////////////////////////////////////////////////////
//
// File: AdamsBashforthOpImpl.hpp
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

#include "Operators/TimeOps/AdamsBashforth/AdamsBashforthKernelLaunchers.hpp"
#include "Operators/TimeOps/AdamsBashforth/AdamsBashforthOp.hpp"
#include "Operators/TimeOps/RungeKutta/RungeKuttaOp.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData>
class AdamsBashforthOpImpl : public AdamsBashforthOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order.
    static_assert(
        IntOrder >= 1 && IntOrder <= 4,
        "The AdamsBashforthOp class is only implemented for order 1-4.");

public:
    AdamsBashforthOpImpl(const ExpListSharedPtr &expansionList)
        : AdamsBashforthOp<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            AdamsBashforthOpImpl<ExecSpace, Scheme, IntOrder, TData>>(
            expansionList);
    }

protected:
    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(this->m_explicitRhsFunctor,
                 "AdamsBashforth schemes require a DoExplicitRhs method. "
                 "Define with AdamsBashforthOp->DefineExplicit().");
        ASSERTL0(this->m_projectionFunctor,
                 "AdamsBashforth schemes require a DoProjection method. Define "
                 "with AdamsBashforthOp->DefineProjection().");

        // Startup.
        if (this->m_step + 1 < IntOrder)
        {
            // Allocate new storage.
            this->m_explicits.push_front(Field<TData, FieldState::Phys>(
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                inout.GetNumComponents(), inout.GetNumHomoModes(),
                ExecSpace::alignment));

            // Compute explicit terms.
            this->DoExplicitRhs(inout, this->m_explicits[0], this->m_time,
                                this->m_timestep);

            // Initialise RungeKutta scheme.
            auto startup = RungeKuttaOp<TData>::Create(
                this->m_expansionList, IntOrder, "", ExecSpace::name);

            // Copy functors from AdamsBashforth scheme.
            startup->CopyFunctorsFrom(*this);

            // Advance in time with startup.
            startup->SetTime(this->m_time);
            startup->SetStep(this->m_step);
            startup->Apply(inout);

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
                this->m_explicits.push_back(Field<TData, FieldState::Phys>(
                    GetBlockAttributes<TData>(FieldState::Phys,
                                              this->m_expansionList),
                    inout.GetNumComponents(), inout.GetNumHomoModes(),
                    ExecSpace::alignment));
            }

            if constexpr (IntOrder > 1)
            {
                this->RollOver(this->m_explicits);
            }

            // Ensure solution is in correct space.
            this->DoProjection(inout, inout, this->m_time);

            // Compute explicit terms.
            this->DoExplicitRhs(inout, this->m_explicits[0], this->m_time,
                                this->m_timestep);

            // Do extrapolation.
            UpdateSolution(
                inout, std::make_integer_sequence<unsigned int, IntOrder>());

            // Increment step and time.
            this->m_time += this->m_timestep;
            this->m_step++;
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
            auto nsize       = inoutBlock.GetNumElementsWithPadding() *
                         inoutBlock.GetNumData() *
                         inoutBlock.GetNumComponents() *
                         inoutBlock.GetNumHomoModes();

            // Compute new solution.
            UpdateSolutionKernel<ExecSpace, Scheme>(
                nsize, inoutBlock.template GetPtr<MemSpace, ReadWrite>(),
                (this->m_explicits[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>())...);
        }
    }
};

} // namespace Nektar::Operators::detail
