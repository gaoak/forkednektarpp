///////////////////////////////////////////////////////////////////////////////
//
// File: AdamsMoultonOpImpl.hpp
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

#include "Operators/TimeOps/AdamsMoulton/AdamsMoultonOp.hpp"
#include "Operators/TimeOps/DIRK/DIRKOp.hpp"

#include "Operators/TimeOps/AdamsMoulton/AdamsMoultonKernelLaunchers.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData>
class AdamsMoultonOpImpl : public AdamsMoultonOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order.
    static_assert(
        IntOrder >= 1 && IntOrder <= 4,
        "The AdamsMoultonOp class is only implemented for order 1-4.");

public:
    AdamsMoultonOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                       const std::vector<std::string> &components)
        : AdamsMoultonOp<TData>(expansionList, components)
    {
        // Initialize coefficients at construction time.
        SetCoefficients();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<TimeOp<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            AdamsMoultonOpImpl<ExecSpace, Scheme, IntOrder, TData>>(
            expansionList, components);
    }

protected:
    TData m_gamma;

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(this->m_implicitFunctor,
                 "AdamsMoulton schemes require a DoImplicit method. Define "
                 "with AdamsMoultonOp->DefineImplicit().");

        // Startup.
        if (this->m_step + 1 < IntOrder)
        {
            // Use lower-order Adams-Moulton scheme as startup.
            if constexpr (IntOrder <= 2)
            {
                // Initialise AdamsMoulton and hand-over the m_implicits deque.
                auto startup = AdamsMoultonOp<TData>::Create(
                    this->m_expansionList, this->m_components, this->m_step + 1,
                    ExecSpace::name);

                // Copy functors from outer/higher-order AdamsMoulton scheme.
                startup->CopyFunctorsFrom(*this);

                // Move implicits to startup.
                startup->SetImplicits(this->TakeImplicits());

                // Advance in time with startup.
                startup->SetTime(this->m_time);
                startup->SetStep(this->m_step);
                startup->Apply(inout);

                // Move implicits back to higher-order AdamsMoulton.
                this->SetImplicits(startup->TakeImplicits());
            }
            // Use DIRK scheme as startup.
            else
            {
                ASSERTL0(this->m_implicitRhsFunctor,
                         "AdamsMoulton schemes wiht order > 2 require a "
                         "DoImplicitRhs method. Define "
                         "with AdamsMoultonOp->DefineImplicitRhs().");

                // Allocate new storage.
                this->m_implicits.push_front(Field<TData, FieldState::Phys>(
                    GetBlockAttributes<TData, FieldState::Phys>(
                        this->m_expansionList),
                    this->m_components, inout.GetNumHomoModes()));

                // Initialise startup and hand-over the m_implicits deque.
                auto maxOrder = std::min(3u, IntOrder);
                auto startup  = DIRKOp<TData>::Create(
                    this->m_expansionList, this->m_components, maxOrder, "",
                    ExecSpace::name);

                // Copy functors from outer/higher-order AdamsMoulton scheme.
                startup->CopyFunctorsFrom(*this);

                // Advance in time with startup.
                startup->SetTime(this->m_time);
                startup->SetStep(this->m_step);
                startup->Apply(inout);

                // Compute implicit terms.
                this->DoImplicitRhs(inout, this->m_implicits[0], this->m_time,
                                    this->m_timestep);
            }

            // Increment step and time.
            this->m_time += this->m_timestep;
            this->m_step++;
        }
        // After startup.
        else
        {
            // Allocate new storage.
            if (this->m_implicits.size() < IntOrder)
            {
                this->m_implicits.push_back(Field<TData, FieldState::Phys>(
                    GetBlockAttributes<TData, FieldState::Phys>(
                        this->m_expansionList),
                    this->m_components, inout.GetNumHomoModes()));
            }

            // Do extrapolation.
            if constexpr (IntOrder > 1)
            {
                UpdateSolution(
                    inout,
                    std::make_integer_sequence<unsigned int, IntOrder - 1>());
            }

            // Use this->m_implicits[0] as temporary storage for extrapolated
            // solution.
            this->RollOver(inout, this->m_implicits);

            // Compute next time step.
            this->DoImplicit(this->m_implicits[0], inout,
                             this->m_time + this->m_timestep,
                             m_gamma * this->m_timestep);

            // Update implicit derivative.
            sub<ExecSpace>(inout, this->m_implicits[0], this->m_implicits[0]);
            mul<ExecSpace>((TData)1.0 / m_gamma, this->m_implicits[0],
                           this->m_implicits[0]);

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
            const unsigned int streamID = blk + 1;

            // Determine shape and type of the element.
            auto &inoutBlock = inout.GetBlocks()[blk];
            auto nsize       = inoutBlock.GetNumElementsWithPadding() *
                         inoutBlock.GetNumData() *
                         inoutBlock.GetNumComponents() *
                         inoutBlock.GetNumHomoModes();

            // Compute new solutions.
            UpdateSolutionKernel<ExecSpace, Scheme>(
                streamID, nsize,
                inoutBlock.template GetPtr<MemSpace, ReadWrite>(streamID),
                (this->m_implicits[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...);
        }
    }

    /*
     *  Setup gamma coefficient for AdamsMoulton.
     */
    void SetCoefficients()
    {
        if constexpr (IntOrder == 1)
        {
            m_gamma = 1.0;
        }
        else if constexpr (IntOrder == 2)
        {
            m_gamma = 1.0 / 2.0;
        }
        else if constexpr (IntOrder == 3)
        {
            m_gamma = 5.0 / 12.0;
        }
        else if constexpr (IntOrder == 4)
        {
            m_gamma = 9.0 / 24.0;
        }
    }
};

} // namespace Nektar::Operators::detail
