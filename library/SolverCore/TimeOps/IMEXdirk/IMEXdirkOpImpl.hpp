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

#include "SolverCore/TimeOps/IMEXdirk/IMEXdirkOp.hpp"

#include "SolverCore/TimeOps/IMEXdirk/IMEXdirkKernelLaunchers.hpp"

using namespace Nektar;

namespace Nektar::SolverCore::detail
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

        static_assert(check, "Unknown IMEXdirk (ImpStage, ExpStage, IntOrder) "
                             "scheme.");

        // Allocate memory.
        this->m_solutions.push_back(
            LibUtilities::Field<TData, FieldState::Phys>(
                this->m_blockAttr, this->m_components, 1));

        for (unsigned int i = 0; i < ImpStage; i++)
        {
            this->m_implicits.push_back(
                LibUtilities::Field<TData, FieldState::Phys>(
                    this->m_blockAttr, this->m_components, 1));
        }

        for (unsigned int i = 0; i < ExpStage; i++)
        {
            this->m_explicits.push_back(
                LibUtilities::Field<TData, FieldState::Phys>(
                    this->m_blockAttr, this->m_components, 1));
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
    /*
     *  True if the scheme is stiffly accurate in both parts: the last stage
     *  solution is the new solution, so the last implicit term and the final
     *  update are not computed.
     */
    static constexpr bool LastStageIsSolution()
    {
        if constexpr (ImpStage != ExpStage)
        {
            return false;
        }
        else
        {
            constexpr unsigned int nStage = ImpStage;
            constexpr auto lambda =
                GetIMEXdirkLambdaCoefficients<ImpStage, ExpStage, IntOrder,
                                              TData>();
            constexpr auto coeff =
                GetIMEXdirkCoefficients<ImpStage, ExpStage, IntOrder, TData>();
            constexpr auto stageCoeff =
                GetIMEXdirkStageCoefficients<ImpStage, ExpStage, IntOrder,
                                             TData>();
            constexpr unsigned int indStart =
                nStage * (nStage - 1) / 2 + (nStage - 1) * (nStage - 2) / 2;

            bool isSolution = lambda[nStage - 1] != TData(0) &&
                              coeff[nStage - 1] == lambda[nStage - 1];
            if constexpr (nStage > 1)
            {
                for (unsigned int i = 0; i < nStage - 1; ++i)
                {
                    isSolution =
                        isSolution && coeff[i] == stageCoeff[indStart + i];
                }
            }
            for (unsigned int i = 0; i < nStage; ++i)
            {
                isSolution =
                    isSolution && coeff[ImpStage + i] ==
                                      stageCoeff[indStart + nStage - 1 + i];
            }
            return isSolution;
        }
    }

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &inout) override
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
        if constexpr (!LastStageIsSolution())
        {
            UpdateSolution(
                inout, std::make_integer_sequence<unsigned int, ImpStage>(),
                std::make_integer_sequence<unsigned int, ExpStage>());
        }

        // Increment step and time.
        this->m_time += this->m_timestep;
        this->m_step++;
    }

    template <unsigned int Stage>
    void Staging(LibUtilities::Field<TData, FieldState::Phys> &inout)
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

            // Compute implicit terms, unless inout is the new solution.
            if constexpr (Stage < ImpStage || !LastStageIsSolution())
            {
                Math::daxpby<ExecSpace>(
                    (TData)1.0 / lambda1, inout, -(TData)1.0 / lambda1,
                    this->m_implicits[Stage - 1], this->m_implicits[Stage - 1]);
            }
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
    void UpdateStage(LibUtilities::Field<TData, FieldState::Phys> &inout,
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
    void UpdateSolution(LibUtilities::Field<TData, FieldState::Phys> &inout,
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

} // namespace Nektar::SolverCore::detail
