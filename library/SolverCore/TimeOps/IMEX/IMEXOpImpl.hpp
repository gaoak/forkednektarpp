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

#include "SolverCore/TimeOps/IMEX/IMEXOp.hpp"
#include "SolverCore/TimeOps/IMEXdirk/IMEXdirkOp.hpp"

#include "SolverCore/TimeOps/IMEX/IMEXKernelLaunchers.hpp"

using namespace Nektar;

namespace Nektar::SolverCore::detail
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
    IMEXOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
               const std::vector<std::string> &components)
        : IMEXOp<TData>(expansionList, components)
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
        return std::make_unique<IMEXOpImpl<ExecSpace, Scheme, IntOrder, TData>>(
            expansionList, components);
    }

protected:
    TData m_gamma;

    std::vector<TData> v_GetExtrapolationCoefficients(
        const TimeOpExtrapolationType type,
        const unsigned int historySize) override
    {
        ASSERTL0(historySize >= 1 && historySize <= IntOrder,
                 "Requested extrapolation history is incompatible with the "
                 "time integration order.");

        switch (type)
        {
            case TimeOpExtrapolationType::StateExtrapolation:
            {
                switch (historySize)
                {
                    case 1:
                        return {1.0};
                    case 2:
                        return {2.0, -1.0};
                    case 3:
                        return {3.0, -3.0, 1.0};
                    case 4:
                        return {4.0, -6.0, 4.0, -1.0};
                    default:
                        break;
                }
                break;
            }
            case TimeOpExtrapolationType::BdfHistory:
            {
                ASSERTL0(historySize == IntOrder,
                         "BDF history extrapolation requires a full time "
                         "integration history.");
                constexpr auto coeff = GetIMEXCoefficients<IntOrder, TData>();
                std::vector<TData> out(historySize);
                for (unsigned int i = 0; i < historySize; ++i)
                {
                    out[i] = coeff[IntOrder + i];
                }
                return out;
            }
            case TimeOpExtrapolationType::ExplicitContribution:
            {
                ASSERTL0(historySize == IntOrder,
                         "Explicit contribution extrapolation requires a "
                         "full time integration history.");
                constexpr auto coeff = GetIMEXCoefficients<IntOrder, TData>();
                std::vector<TData> out(historySize);
                for (unsigned int i = 0; i < historySize; ++i)
                {
                    out[i] = coeff[i];
                }
                return out;
            }
        }

        ASSERTL0(false, "Unsupported IMEX extrapolation coefficient request.");
        return {};
    }

    void v_Apply(MultiRegions::Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(this->m_explicitRhsFunctor,
                 "IMEX schemes require a DoExplicitRhs method. Define with "
                 "IMEXOp->DefineExplicit().");
        ASSERTL0(this->m_implicitFunctor,
                 "IMEX schemes require a DoImplicit method. Define with "
                 "IMEXOp->DefineImplicit().");

        // Startup.
        if (this->m_step + 1 < IntOrder)
        {
            // Allocate new storage.
            this->m_explicits.push_front(
                MultiRegions::Field<TData, FieldState::Phys>(
                    MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                        this->m_expansionList),
                    this->m_components, inout.GetNumHomoModes()));

            // Compute explicit terms.
            this->DoExplicitRhs(inout, this->m_explicits[0], this->m_time,
                                this->m_timestep);

            // Save initial solution.
            this->m_solutions.push_front(
                MultiRegions::Field<TData, FieldState::Phys>(
                    "timestep n-" + std::to_string(this->m_step + 1),
                    MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                        this->m_expansionList),
                    this->m_components, inout.GetNumHomoModes()));

            this->m_solutions[0].template Copy<MemSpace>(inout);

            // Initialise IMEXdirk.
            auto maxOrder = std::min(3u, IntOrder);
            std::string variant =
                std::to_string(maxOrder) + std::to_string(maxOrder + 1);
            auto startup = IMEXdirkOp<TData>::Create(
                this->m_expansionList, this->m_components, maxOrder, variant,
                ExecSpace::name);

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
                    MultiRegions::Field<TData, FieldState::Phys>(
                        MultiRegions::GetBlockAttributes<
                            TData, FieldState::Phys>(this->m_expansionList),
                        this->m_components, inout.GetNumHomoModes()));
            }
            // UpdateSolution previous solutions, explicit part, and sum up.
            if constexpr (IntOrder > 1)
            {
                // Rollover previous explicit parts.
                this->RollOver(this->m_explicits);
            }

            // Compute explicit term.
            this->DoExplicitRhs(inout, this->m_explicits[0], this->m_time,
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
                        MultiRegions::Field<TData, FieldState::Phys>(
                            MultiRegions::GetBlockAttributes<
                                TData, FieldState::Phys>(this->m_expansionList),
                            this->m_components, inout.GetNumHomoModes()));
                }

                // Rollover previous solutions.
                this->RollOver(inout, this->m_implicits);

                if (this->m_useExplicitContributionExtrapolation)
                {
                    this->SetExplicitContributionCoefficients(
                        v_GetExtrapolationCoefficients(
                            TimeOpExtrapolationType::ExplicitContribution,
                            IntOrder));
                }
                this->DoImplicit(this->m_implicits[0], inout,
                                 this->m_time + this->m_timestep,
                                 m_gamma * this->m_timestep);
                if (this->m_useExplicitContributionExtrapolation)
                {
                    this->ClearExplicitContributionCoefficients();
                }

                // Compute implicit terms.
                Math::sub<ExecSpace>(inout, this->m_implicits[0],
                                     this->m_implicits[0]);
                Math::mul<ExecSpace>((TData)1.0 / m_gamma, this->m_implicits[0],
                                     this->m_implicits[0]);
            }
            else
            {
                if (this->m_useExplicitContributionExtrapolation)
                {
                    this->SetExplicitContributionCoefficients(
                        v_GetExtrapolationCoefficients(
                            TimeOpExtrapolationType::ExplicitContribution,
                            IntOrder));
                }
                this->DoImplicit(inout, inout, this->m_time + this->m_timestep,
                                 m_gamma * this->m_timestep);
                if (this->m_useExplicitContributionExtrapolation)
                {
                    this->ClearExplicitContributionCoefficients();
                }
            }

            // Increment step and time.
            this->m_time += this->m_timestep;
            this->m_step++;
        }
    }

    template <unsigned int... Ind, unsigned int... Ind2>
    void UpdateSolution(MultiRegions::Field<TData, FieldState::Phys> &inout,
                        std::integer_sequence<unsigned int, Ind...>,
                        std::integer_sequence<unsigned int, Ind2...>)
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

            // Compute new solution.
            UpdateSolutionKernel<ExecSpace, Scheme>(
                streamID, nsize,
                inoutBlock.template GetPtr<MemSpace, ReadWrite>(streamID),
                (this->m_explicits[Ind]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...,
                (this->m_solutions[Ind2]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...);
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

} // namespace Nektar::SolverCore::detail
