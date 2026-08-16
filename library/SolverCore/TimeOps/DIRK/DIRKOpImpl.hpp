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

#include "SolverCore/TimeOps/DIRK/DIRKOp.hpp"

#include "SolverCore/TimeOps/DIRK/DIRKKernelLaunchers.hpp"

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData>
class DIRKOpImpl : public DIRKOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DIRKOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
               const std::vector<std::string> &components)
        : DIRKOp<TData>(expansionList, components)
    {
        // Compile-time check for valid integration order.
        if constexpr (std::is_same_v<Scheme, DIRKScheme>)
        {
            static_assert(IntOrder >= 1 && IntOrder <= 3,
                          "The DIRK scheme is only implemented for order 1-3.");
        }
        else if constexpr (std::is_same_v<Scheme, DIRK_ESScheme>)
        {
            static_assert(
                IntOrder >= 2 && IntOrder <= 4,
                "The DIRK_ES scheme is only implemented for order 1-4.");
        }

        // Allocate memory.
        this->m_solutions.push_back(
            LibUtilities::Field<TData, FieldState::Phys>(
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    this->m_expansionList),
                this->m_components, 1));

        for (unsigned int i = 0; i < NStage(); i++)
        {
            this->m_implicits.push_back(
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
        return std::make_unique<DIRKOpImpl<ExecSpace, Scheme, IntOrder, TData>>(
            expansionList, components);
    }

protected:
    static constexpr unsigned int NStage()
    {
        if constexpr (std::is_same_v<Scheme, DIRKScheme>)
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
        else if constexpr (std::is_same_v<Scheme, DIRK_ESScheme>)
        {
            if constexpr (IntOrder == 2)
            {
                return 3;
            }
            else if constexpr (IntOrder == 3)
            {
                return 5;
            }
            else if constexpr (IntOrder == 4)
            {
                return 6;
            }
        }
    }

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(this->m_implicitFunctor,
                 "DIRK schemes require a DoImplicit method. Define with "
                 "DIRKOp->DefineImplicit().");
        if constexpr (std::is_same_v<Scheme, DIRK_ESScheme>)
        {
            ASSERTL0(this->m_implicitRhsFunctor,
                     "DIRK_ES schemes require a DoImplicitRhs method. Define "
                     "with DIRKOp->DefineImplicitRhs().");
            ASSERTL0(this->m_projectionFunctor,
                     "DIRK_ES schemes require a DoProjection method. Define "
                     "with DIRKOp->DefineProjection().");
        }

        // Ensure solution is in correct space.
        constexpr auto lambda =
            GetDIRKLambdaCoefficients<Scheme, IntOrder, TData>()[0];
        if constexpr (lambda == 0.0)
        {
            this->DoProjection(inout, this->m_solutions[0], this->m_time);
        }
        else
        {
            this->m_solutions[0].template Copy<MemSpace>(inout);
        }

        // Apply Runge-Kutta scheme.
        Staging<1>(inout);
        UpdateSolution(inout,
                       std::make_integer_sequence<unsigned int, NStage()>());

        // Increment step and time.
        this->m_time += this->m_timestep;
        this->m_step++;
    }

    template <unsigned int Stage>
    void Staging(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        // Compute explicit/implicit terms.
        constexpr auto coeff =
            GetDIRKTimeCoefficients<Scheme, IntOrder, TData>()[Stage - 1];
        constexpr auto lambda =
            GetDIRKLambdaCoefficients<Scheme, IntOrder, TData>()[Stage - 1];
        if constexpr (lambda == 0.0)
        {
            this->DoImplicitRhs(inout, this->m_implicits[Stage - 1],
                                this->m_time + coeff * this->m_timestep,
                                this->m_timestep);
        }
        else
        {
            // Update solution.
            this->DoImplicit(inout, this->m_implicits[Stage - 1],
                             this->m_time + coeff * this->m_timestep,
                             lambda * this->m_timestep);

            // Compute implicit terms.
            Math::sub<ExecSpace>(this->m_implicits[Stage - 1], inout,
                                 this->m_implicits[Stage - 1]);
            Math::mul<ExecSpace>((TData)1.0 / lambda,
                                 this->m_implicits[Stage - 1],
                                 this->m_implicits[Stage - 1]);
        }

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
                (this->m_implicits[ind]
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
                (this->m_implicits[ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...);
        }
    }
};

} // namespace Nektar::SolverCore::detail
