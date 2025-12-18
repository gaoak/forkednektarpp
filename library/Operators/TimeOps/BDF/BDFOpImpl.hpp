///////////////////////////////////////////////////////////////////////////////
//
// File: BDFOpImpl.hpp
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

#include "Operators/TimeOps/BDF/BDFOp.hpp"
#include "Operators/TimeOps/DIRK/DIRKOp.hpp"

#include "Operators/TimeOps/BDF/BDFKernelLaunchers.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData>
class BDFOpImpl : public BDFOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order.
    static_assert(IntOrder >= 1 && IntOrder <= 4,
                  "The BDFOp class is only implemented for order 1-4.");

public:
    BDFOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
              const std::vector<std::string> &components)
        : BDFOp<TData>(expansionList, components)
    {
        // Initialize coefficients at construction time.
        SetCoefficients();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<BDFOpImpl<ExecSpace, Scheme, IntOrder, TData>>(
            expansionList, components);
    }

protected:
    TData m_gamma;

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(this->m_implicitFunctor,
                 "BDF schemes require a DoImplicit method. Define with "
                 "BDFOp->DefineImplicit().");

        // Startup.
        if (this->m_step + 1 < IntOrder)
        {
            // Save initial solution.
            this->m_solutions.push_front(Field<TData, FieldState::Phys>(
                "timestep n-" + std::to_string(this->m_step + 1),
                GetBlockAttributes<TData, FieldState::Phys>(
                    this->m_expansionList),
                this->m_components, inout.GetNumHomoModes()));

            this->m_solutions[0].template Copy<MemSpace>(inout);

            // Initialise DIRK scheme.
            auto maxOrder = std::min(3u, IntOrder);
            auto startup =
                DIRKOp<TData>::Create(this->m_expansionList, this->m_components,
                                      maxOrder, "", ExecSpace::name);

            // Copy functors from outer/higher-order BDF scheme.
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
            if constexpr (IntOrder > 1)
            {
                // Rollover previous solutions in m_solutions and inout
                // (use inout as temporary storage of oldest solution)
                this->RollOver(inout, this->m_solutions);

                // Do extrapolation.
                UpdateSolution(
                    inout,
                    std::make_integer_sequence<unsigned int, IntOrder - 1>());
            }

            // Compute next time step.
            this->DoImplicit(inout, inout, this->m_time + this->m_timestep,
                             m_gamma * this->m_timestep);

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
                (this->m_solutions[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>())...);
        }
    }

    /*
     *  Setup gamma coefficient for BDF.
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
