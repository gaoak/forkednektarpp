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

#include "Operators/MathKernels/MathKernels.hpp"
#include "Operators/TimeOps/AdamsMoulton/AdamsMoultonKernelLaunchers.hpp"
#include "Operators/TimeOps/AdamsMoulton/AdamsMoultonOp.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, unsigned int IntOrder, typename TData>
class AdamsMoultonOpImpl : public AdamsMoultonOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order
    static_assert(
        IntOrder >= 1 && IntOrder <= 4,
        "The AdamsMoultonOp class is only implemented for order 1-4.");

public:
    AdamsMoultonOpImpl(const ExpListSharedPtr &expansionList)
        : AdamsMoultonOp<TData>(expansionList)
    {
        // Initialize coefficients at construction time
        SetCoefficients();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const ExpListSharedPtr &expansionList)
    {
        return std::make_unique<AdamsMoultonOpImpl<ExecSpace, IntOrder, TData>>(
            expansionList);
    }

protected:
    TData m_gamma;

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that implicit function is defined for IMEX
        ASSERTL0(
            this->m_implicitFunctor,
            "AdamsMoulton schemes require a DoImplicit method. Define with "
            "AdamsMoultonOp->DefineImplicit().");

        // Startup
        while (this->m_step + 1 < IntOrder)
        {
            // Initialise AdamsMoulton and hand-over the m_implicits deque
            auto startup = AdamsMoultonOp<TData>::Create(
                this->m_expansionList, this->m_step + 1, ExecSpace::name);

            // Copy functors from outer/higher-order AdamsMoulton scheme
            startup->CopyFunctorsFrom(*this);

            // Move implicits to startup
            startup->SetImplicits(this->TakeImplicits());

            // Advance in time with startup
            startup->SetTime(this->m_time);
            startup->SetNumStep(this->m_step);
            startup->Apply(inout);

            // Move implicits back to higher-order AdamsMoulton
            this->SetImplicits(startup->TakeImplicits());

            // Increment step and time
            this->m_time += this->m_timestep;
            this->m_step++;
        }

        if (this->m_implicits.size() < IntOrder)
        {
            // Allocate new storage
            this->m_implicits.push_back(Field<TData, FieldState::Phys>::Create(
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                inout.GetNumComponents(), inout.GetNumHomoModes(),
                ExecSpace::alignment));
        }

        // After startup
        if (this->m_step + 1 >= IntOrder)
        {
            if constexpr (IntOrder > 1)
            {
                // Do extrapolation.
                Extrapolate(
                    inout,
                    std::make_integer_sequence<unsigned int, IntOrder - 1>());
            }

            // Use this->m_implicits[0] as temporary storage for extrapolated
            // solution
            this->RollOver(inout, this->m_implicits);

            // Compute next time step
            this->DoImplicit(this->m_implicits[0], inout,
                             m_gamma * this->m_timestep);

            // Update implicit derivative
            sub<ExecSpace>(inout, this->m_implicits[0], this->m_implicits[0]);
            mul<ExecSpace>(1.0 / (m_gamma * this->m_timestep),
                           this->m_implicits[0], this->m_implicits[0]);

            // Increment step and time
            this->m_time += this->m_timestep;
            this->m_step++;
        }
    }

    template <unsigned int... ind>
    void Extrapolate(Field<TData, FieldState::Phys> &inout,
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

            // Initialize pointer.
            auto inoutPtr = inoutBlock.template GetPtr<MemSpace, ReadWrite>();

            ExtrapolateAdamsMoultonKernel<ExecSpace>(
                nphys * nelmt, this->m_timestep, inoutPtr,
                (this->m_implicits[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>())...);
        }
    }

    /*
     *  Setup gamma coefficient for AdamsMoulton.
     *  Note the extrapolation coefficients are defined inside the
     *  ExtrapolateAdamsMoultonKernel.
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
