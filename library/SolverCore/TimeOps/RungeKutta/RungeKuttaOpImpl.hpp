///////////////////////////////////////////////////////////////////////////////
//
// File: RungeKuttaOpImpl.hpp
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

#include "SolverCore/TimeOps/RungeKutta/RungeKuttaOp.hpp"

#include "SolverCore/TimeOps/RungeKutta/RungeKuttaKernelLaunchers.hpp"

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData>
class RungeKuttaOpImpl : public RungeKuttaOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    RungeKuttaOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : RungeKuttaOp<TData>(expansionList, components)
    {
        // Compile-time check for valid integration order
        if constexpr (std::is_same_v<Scheme, RungeKuttaScheme>)
        {
            static_assert(
                IntOrder >= 1 && IntOrder <= 5,
                "The RungeKutta scheme is only implemented for order 1-5.");
        }
        else if constexpr (std::is_same_v<Scheme, RungeKuttaSSPScheme>)
        {
            static_assert(
                IntOrder >= 1 && IntOrder <= 5,
                "The RungeKuttaSSP scheme is only implemented for order 1-3.");
        }

        // Allocate memory.
        this->m_solutions.push_back(
            LibUtilities::Field<TData, FieldState::Phys>(
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    this->m_expansionList),
                this->m_components, 1));

        for (unsigned int i = 0; i < NStage(); i++)
        {
            this->m_explicits.push_back(
                LibUtilities::Field<TData, FieldState::Phys>(
                    MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                        this->m_expansionList),
                    this->m_components, 1));
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<TimeOp<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            RungeKuttaOpImpl<ExecSpace, Scheme, IntOrder, TData>>(expansionList,
                                                                  components);
    }

protected:
    static constexpr unsigned int NStage()
    {
        if constexpr (std::is_same_v<Scheme, RungeKuttaScheme>)
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
            else if constexpr (IntOrder == 4)
            {
                return 4;
            }
            else if constexpr (IntOrder == 5)
            {
                return 6;
            }
        }
        else if constexpr (std::is_same_v<Scheme, RungeKuttaSSPScheme>)
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
    }

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(
            this->m_explicitRhsFunctor,
            "RungeKutta schemes require a DoExplicitRhs method. Define with "
            "RungeKuttaOp->DefineExplicit().");
        ASSERTL0(
            this->m_projectionFunctor,
            "RungeKutta schemes require a DoProjection method. Define with "
            "RungeKuttaOp->DefineProjection().");

        // Apply Runge-Kutta scheme.
        Staging<1>(inout);
        UpdateSolution(inout,
                       std::make_integer_sequence<unsigned int, NStage()>());

        // Increment step and time.
        this->m_time += this->m_timestep;
        this->m_step++;
    }

    TData v_GetTimeStability(void) const override
    {
        switch (IntOrder)
        {
            case 1:
                return 2.0;
            case 2:
                return 2.0;
            case 3:
                return 2.51274532661833;
            case 4:
                return 2.784;
            case 5:
                return 3.21704786664011;
            default:
                return 2.0;
        }
    }

    template <unsigned int Stage>
    void Staging(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        constexpr auto coeff =
            GetRungeKuttaTimeCoefficients<Scheme, IntOrder, TData>()[Stage - 1];

        // Ensure solution is in correct space.
        this->DoProjection(inout, inout,
                           this->m_time + coeff * this->m_timestep);

        if constexpr (Stage == 1)
        {
            this->m_solutions[0].template Copy<MemSpace>(inout);
        }

        // Compute explicit term.
        this->DoExplicitRhs(inout, this->m_explicits[Stage - 1],
                            this->m_time + coeff * this->m_timestep,
                            this->m_timestep);

        // Recursive loop over stages.
        if constexpr (Stage < NStage())
        {
            // Compute stage.
            UpdateStage(inout,
                        std::make_integer_sequence<unsigned int, Stage>());

            // Do next stage.
            Staging<Stage + 1>(inout);
        }
    }

    template <unsigned int... ind>
    void UpdateStage(LibUtilities::Field<TData, FieldState::Phys> &out,
                     std::integer_sequence<unsigned int, ind...>)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            // Determine shape and type of the element.
            auto &outBlock = out.GetBlocks()[blk];
            auto nsize     = outBlock.GetNumElementsWithPadding() *
                         outBlock.GetNumData() * outBlock.GetNumComponents() *
                         outBlock.GetNumHomoModes();

            // Compute stage solution.
            UpdateStageKernel<ExecSpace, Scheme, IntOrder>(
                streamID, nsize,
                outBlock.template GetPtr<MemSpace, WriteOnly>(streamID),
                this->m_solutions[0]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>(streamID),
                (this->m_explicits[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...);
        }
    }

    template <unsigned int... ind>
    void UpdateSolution(LibUtilities::Field<TData, FieldState::Phys> &out,
                        std::integer_sequence<unsigned int, ind...>)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            // Determine shape and type of the element.
            auto &outBlock = out.GetBlocks()[blk];
            auto nsize     = outBlock.GetNumElementsWithPadding() *
                         outBlock.GetNumData() * outBlock.GetNumComponents() *
                         outBlock.GetNumHomoModes();

            // Compute new solution.
            UpdateSolutionKernel<ExecSpace, Scheme, IntOrder>(
                streamID, nsize,
                outBlock.template GetPtr<MemSpace, WriteOnly>(streamID),
                this->m_solutions[0]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>(streamID),
                (this->m_explicits[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...);
        }
    }
};

} // namespace Nektar::SolverCore::detail
