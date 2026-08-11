///////////////////////////////////////////////////////////////////////////////
//
// File: IMEXdirkOpImpl.hpp
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

#include "Operators/TimeOps/IMEXdirk/IMEXdirkOp.hpp"

#include "Operators/TimeOps/IMEXdirk/IMEXdirkKernelLaunchers.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Scheme, unsigned int ImpStage,
          unsigned int ExpStage, unsigned int IntOrder, typename TData>
class IMEXdirkOpImpl : public IMEXdirkOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IMEXdirkOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : IMEXdirkOp<TData>(expansionList, components)
    {
        // Compile-time check for valid integration order.
        constexpr auto check =
            // IMEX Dirk 1 1 1 : Forward - Backward Euler IMEX
            (ImpStage == 1 && ExpStage == 1 && IntOrder == 1) ||
            // IMEX Dirk 1 2 1 : Forward - Backward Euler IMEX w/B implicit
            // == B explicit
            (ImpStage == 1 && ExpStage == 2 && IntOrder == 1) ||
            // IMEX Dirk 1 2 2 : Implict-Explicit Midpoint IMEX
            (ImpStage == 1 && ExpStage == 2 && IntOrder == 2) ||
            // IMEX Dirk 2 2 2 : L Stable, two stage, second order IMEX
            (ImpStage == 2 && ExpStage == 2 && IntOrder == 2) ||
            // IMEX Dirk 2 3 2 : L Stable, two stage, second order IMEX
            (ImpStage == 2 && ExpStage == 3 && IntOrder == 2) ||
            // IMEX Dirk 2 3 3 : L Stable, two stage, third order IMEX
            (ImpStage == 2 && ExpStage == 3 && IntOrder == 3) ||
            // IMEX Dirk 3 4 3 : L Stable, three stage, third order IMEX
            (ImpStage == 3 && ExpStage == 4 && IntOrder == 3) ||
            // IMEX Dirk 4 4 3 : L Stable, four stage, third order IMEX
            (ImpStage == 4 && ExpStage == 4 && IntOrder == 3);

        ASSERTL0(check, "The unknow IMEXdirk(" + std::to_string(ImpStage) +
                            ", " + std::to_string(ExpStage) + ", " +
                            std::to_string(IntOrder) + ") schemes.");

        // Allocate memory.
        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                this->m_expansionList);
        this->m_solutions.push_back(
            MultiRegions::Field<TData, FieldState::Phys>(
                blockAttr, this->m_components, 1));

        for (unsigned int i = 0; i < ImpStage; i++)
        {
            this->m_implicits.push_back(
                MultiRegions::Field<TData, FieldState::Phys>(
                    blockAttr, this->m_components, 1));
        }

        for (unsigned int i = 0; i < ExpStage; i++)
        {
            this->m_explicits.push_back(
                MultiRegions::Field<TData, FieldState::Phys>(
                    blockAttr, this->m_components, 1));
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<TimeOp<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<IMEXdirkOpImpl<ExecSpace, Scheme, ImpStage,
                                               ExpStage, IntOrder, TData>>(
            expansionList, components);
    }

protected:
    void v_Apply(MultiRegions::Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(this->m_explicitRhsFunctor,
                 "IMEXdirk schemes require a DoExplicitRhs method. Define with "
                 "IMEXdirkOp->DefineExplicit().");
        ASSERTL0(this->m_implicitFunctor,
                 "IMEXdirk schemes require a DoImplicit method. Define with "
                 "IMEXdirkOp->DefineImplicit().");

        this->m_solutions[0].template Copy<MemSpace>(inout);

        // Apply IMEX dirk scheme.
        Staging<1>(inout);
        UpdateSolution(inout,
                       std::make_integer_sequence<unsigned int, ImpStage>(),
                       std::make_integer_sequence<unsigned int, ExpStage>());

        // Increment step and time.
        this->m_time += this->m_timestep;
        this->m_step++;
    }

    template <unsigned int Stage>
    void Staging(MultiRegions::Field<TData, FieldState::Phys> &inout)
    {
        constexpr auto coeff0 =
            GetIMEXdirkTimeCoefficients<ImpStage, ExpStage, IntOrder,
                                        TData>()[Stage - 1];

        // Compute explicit terms.
        this->DoExplicitRhs(inout, this->m_explicits[Stage - 1],
                            this->m_time + coeff0 * this->m_timestep,
                            this->m_timestep);

        // Compute implicit terms.
        if constexpr (Stage <= ImpStage)
        {
            // Compute stage.
            UpdateStage(this->m_implicits[Stage - 1],
                        std::make_integer_sequence<unsigned int, Stage - 1>(),
                        std::make_integer_sequence<unsigned int, Stage>());

            constexpr auto coeff1 =
                GetIMEXdirkTimeCoefficients<ImpStage, ExpStage, IntOrder,
                                            TData>()[Stage];
            constexpr auto lambda1 =
                GetIMEXdirkLambdaCoefficients<ImpStage, ExpStage, IntOrder,
                                              TData>()[Stage - 1];

            // Update solution.
            if (this->m_useExplicitContributionExtrapolation)
            {
                SetStageExplicitContributionCoefficients<Stage>();
            }
            this->DoImplicit(this->m_implicits[Stage - 1], inout,
                             this->m_time + coeff1 * this->m_timestep,
                             lambda1 * this->m_timestep);
            if (this->m_useExplicitContributionExtrapolation)
            {
                this->ClearExplicitContributionCoefficients();
            }

            // Compute implicit terms.
            sub<ExecSpace>(inout, this->m_implicits[Stage - 1],
                           this->m_implicits[Stage - 1]);
            mul<ExecSpace>((TData)1.0 / lambda1, this->m_implicits[Stage - 1],
                           this->m_implicits[Stage - 1]);
        }

        // Recursive loop over stages.
        if constexpr (Stage < ExpStage)
        {
            // Do next stage.
            Staging<Stage + 1>(inout);
        }
    }

    template <unsigned int Stage>
    void SetStageExplicitContributionCoefficients()
    {
        constexpr auto coeff =
            GetIMEXdirkStageCoefficients<ImpStage, ExpStage, IntOrder, TData>();

        const unsigned int indStart =
            Stage * (Stage - 1) / 2 + (Stage - 1) * (Stage - 2) / 2;
        const unsigned int explicitStart = indStart + Stage - 1;

        std::vector<TData> outCoeff(Stage);
        for (unsigned int i = 0; i < Stage; ++i)
        {
            outCoeff[i] = coeff[explicitStart + Stage - 1 - i];
        }

        this->SetExplicitContributionCoefficients(std::move(outCoeff));
    }

    template <unsigned int... ind0, unsigned int... ind1>
    void UpdateStage(MultiRegions::Field<TData, FieldState::Phys> &inout,
                     std::integer_sequence<unsigned int, ind0...>,
                     std::integer_sequence<unsigned int, ind1...>)
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

            // Compute stage solution.
            UpdateStageKernel<ExecSpace, Scheme, ImpStage, ExpStage, IntOrder>(
                streamID, nsize,
                inoutBlock.template GetPtr<MemSpace, WriteOnly>(streamID),
                this->m_solutions[0]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>(streamID),
                (this->m_implicits[ind0]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...,
                (this->m_explicits[ind1]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...);
        }
    }

    template <unsigned int... ind0, unsigned int... ind1>
    void UpdateSolution(MultiRegions::Field<TData, FieldState::Phys> &inout,
                        std::integer_sequence<unsigned int, ind0...>,
                        std::integer_sequence<unsigned int, ind1...>)
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
            UpdateSolutionKernel<ExecSpace, Scheme, ImpStage, ExpStage,
                                 IntOrder>(
                streamID, nsize,
                inoutBlock.template GetPtr<MemSpace, WriteOnly>(streamID),
                this->m_solutions[0]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>(streamID),
                (this->m_implicits[ind0]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...,
                (this->m_explicits[ind1]
                     .GetBlocks()[blk]
                     .template GetPtr<MemSpace, ReadOnly>(streamID))...);
        }
    }
};

} // namespace Nektar::Operators::detail
