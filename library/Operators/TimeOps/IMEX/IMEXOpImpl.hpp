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

#include "Operators/TimeOps/IMEX/IMEXOp.hpp"

#include "Operators/TimeOps/IMEX/IMEXDeviceKernels.hpp"
#include "Operators/TimeOps/IMEX/IMEXSerialAVXKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData, unsigned int IntOrder>
class IMEXOpImpl : public IMEXOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order
    static_assert(IntOrder >= 1 && IntOrder <= 4,
                  "The IMEXOp class is only implemented for order 1-4.");

public:
    IMEXOpImpl(const ExpListSharedPtr &expansionList)
        : IMEXOp<TData>(expansionList)
    {
        // ASSERTL0(
        //     TimeOp<TData>::m_intMethod == "IMEX",
        //     "This class must be called for the method 'IMEX'. Instead it "
        //     "was called with " + TimeOp<TData>::m_intMethod + ".");
        //
        // ASSERTL0(1 <= TimeOp<TData>::m_intOrder &&
        //              TimeOp<TData>::m_intOrder <= 4,
        //          "The IMEXOp class is only implemented for order 1-4.");

        // Initialize coefficients at construction time
        SetCoefficients();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const ExpListSharedPtr &expansionList)
    {
        auto order = expansionList->GetSession()->GetTimeIntScheme().order;
        switch (order)
        {
            case 1:
                return std::make_unique<IMEXOpImpl<ExecSpace, TData, 1>>(
                    expansionList);
            case 2:
                return std::make_unique<IMEXOpImpl<ExecSpace, TData, 2>>(
                    expansionList);
            case 3:
                return std::make_unique<IMEXOpImpl<ExecSpace, TData, 3>>(
                    expansionList);
            case 4:
                return std::make_unique<IMEXOpImpl<ExecSpace, TData, 4>>(
                    expansionList);
            default:
                NEKERROR(ErrorUtil::efatal,
                         "IMEX order must be between 1 and 4. "
                         "Instead it is " +
                             std::to_string(order));
                return nullptr;
        }
    }

    // Move‐in
    void SetSolutions(std::deque<Field<TData, FieldState::Phys>> &&solutions,
                      std::deque<Field<TData, FieldState::Phys>> &&explicits)
    // Field<TData, FieldState::Phys> &&wsp_explicit)
    {
        ASSERTL0(m_solutions.empty(),
                 "Do not call SetSolutions() if m_solutions is "
                 "already defined in IMEX operator.")
        m_solutions = std::move(solutions);

        ASSERTL0(m_explicits.empty(),
                 "Do not call SetSolutions() if m_explicits is "
                 "already defined in IMEX operator.")
        m_explicits = std::move(explicits);
    }

    void InitMemoryRegionsSolution()
    {
        ASSERTL0(!m_solutions.empty(),
                 "Do not call InitMemoryRegions() if m_solutions is not "
                 "defined in IMEX operator.")

        m_solutions_mr = MemoryRegion<const TData *>::Create(
            m_solutions.size(), ExecSpace::alignment);
    }

    void InitMemoryRegionsExplicit()
    {
        ASSERTL0(!m_explicits.empty(),
                 "Do not call InitMemoryRegions() if m_explicits is not "
                 "defined in IMEX operator.")

        m_explicits_mr = MemoryRegion<const TData *>::Create(
            m_explicits.size(), ExecSpace::alignment);
    }

    // Move‐out
    std::deque<Field<TData, FieldState::Phys>> TakeSolutions()
    {
        return std::move(m_solutions);
    }

    std::deque<Field<TData, FieldState::Phys>> TakeExplicits()
    {
        return std::move(m_explicits);
    }

protected:
    // Extrapolation coefficient of implicit scheme
    TData m_gamma;

    // Storage for previous solutions in Fields
    // and memory region for pointer access on device
    std::deque<Field<TData, FieldState::Phys>> m_solutions;
    MemoryRegion<const TData *> m_solutions_mr;

    // Storage for previous explicit parts in Fields
    // and memory region for pointer access on device
    std::deque<Field<TData, FieldState::Phys>> m_explicits;
    MemoryRegion<const TData *> m_explicits_mr;

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that implicit function call is defined for IMEX
        ASSERTL0(TimeOp<TData>::m_implicitFunctor,
                 "IMEX schemes require a DoImplicit method. Define with "
                 "IMEXOp->DefineImplicit().");

        // Check that explicit function is defined for IMEX
        ASSERTL0(TimeOp<TData>::m_explicitFunctor,
                 "IMEX schemes require a DoExplicit method. Define with "
                 "IMEXOp->DefineExplicit().");

        // Compute explicit part for current timestep
        TimeOp<TData>::DoExplicit(inout, m_explicits[0], m_gamma);

        // For first order: inout = 1 / \Delta t * u^n + explicit part
        if constexpr (IntOrder == 1)
        {
            mul<ExecSpace, TData>(1. / TimeOp<TData>::m_timestep, inout, inout);
            add<ExecSpace, TData>(inout, m_explicits[0], inout);
        }

        // Extrapolate previous solutions, explicit part, and sum up
        if constexpr (IntOrder > 1)
        {
            // Rollover previous solutions and explicit parts
            // (use inout as temporary storage of oldest solution)
            RollOver(inout, m_solutions);
            RollOver(m_explicits);

            // Loop over the blocks.
            for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
            {
                // Determine shape and type of the element.
                auto &inoutBlock = inout.GetBlocks()[blk];
                auto nelmt       = inoutBlock.GetNumElements();
                auto nphys       = inoutBlock.GetNumData();

                // Initialize pointer.
                auto inoutPtr =
                    inoutBlock.template GetPtr<MemSpace, ReadWrite>();

                // Gather pointers for previous solutions on Host and setup
                // read-only pointers for Device.
                auto solutionsHost =
                    m_solutions_mr
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned int ns = 0; ns < m_solutions.size(); ++ns)
                {
                    solutionsHost[ns] =
                        m_solutions[ns]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, ReadOnly>();
                }
                auto solutionsDevice =
                    m_solutions_mr.template GetPtr<MemSpace, ReadOnly>();

                // Gather pointers for explicit parts on Host and setup
                // read-only pointers for Device.
                auto explicitsHost =
                    m_explicits_mr
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned int ne = 0; ne < m_explicits.size(); ++ne)
                {
                    explicitsHost[ne] =
                        m_explicits[ne]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, ReadOnly>();
                }
                auto explicitsDevice =
                    m_explicits_mr.template GetPtr<MemSpace, ReadOnly>();

                // Do extrapolation.
                ExtrapolateIMEXKernel<ExecSpace, TData, IntOrder>(
                    nphys * nelmt, TimeOp<TData>::m_timestep, solutionsDevice,
                    explicitsDevice, inoutPtr);
            }
        }

        // Compute next time step
        TimeOp<TData>::DoImplicit(inout, m_gamma);
    }

    // Initialise the time-stepping array
    void v_Initialise(Field<TData, FieldState::Phys> &initial, TData &time,
                      size_t &step) override
    {
        // Initialse workspace for explicit part as first entry in deque
        m_explicits.push_back(Field<TData, FieldState::Phys>::Create(
            "explicit n",
            GetBlockAttributes<TData>(FieldState::Phys, this->m_expansionList),
            initial.GetNumComponents(), initial.GetNumHomoModes(),
            ExecSpace::alignment));
        TimeOp<TData>::DoExplicit(initial, m_explicits[0], m_gamma);

        // Initialise all previous solutions to initial condition
        // and compute explicit parts for initial condition
        if constexpr (IntOrder > 1)
        {
            m_solutions.push_back(Field<TData, FieldState::Phys>::Create(
                "timestep n-1",
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                initial.GetNumComponents(), initial.GetNumHomoModes(),
                ExecSpace::alignment));
            m_solutions[0].template Copy<MemSpace>(initial);

            m_explicits.push_back(Field<TData, FieldState::Phys>::Create(
                "explicit n-1",
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                initial.GetNumComponents(), initial.GetNumHomoModes(),
                ExecSpace::alignment));
            TimeOp<TData>::DoExplicit(initial, m_explicits[1], m_gamma);
        }
        if constexpr (IntOrder > 2)
        {
            m_solutions.push_back(Field<TData, FieldState::Phys>::Create(
                "timestep n-2",
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                initial.GetNumComponents(), initial.GetNumHomoModes(),
                ExecSpace::alignment));
            m_solutions[1].template Copy<MemSpace>(initial);

            m_explicits.push_back(Field<TData, FieldState::Phys>::Create(
                "explicit n-2",
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                initial.GetNumComponents(), initial.GetNumHomoModes(),
                ExecSpace::alignment));
            TimeOp<TData>::DoExplicit(initial, m_explicits[2], m_gamma);
        }
        if constexpr (IntOrder > 3)
        {
            m_solutions.push_back(Field<TData, FieldState::Phys>::Create(
                "timestep n-3",
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                initial.GetNumComponents(), initial.GetNumHomoModes(),
                ExecSpace::alignment));
            m_solutions[2].template Copy<MemSpace>(initial);

            m_explicits.push_back(Field<TData, FieldState::Phys>::Create(
                "explicit n-3",
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                initial.GetNumComponents(), initial.GetNumHomoModes(),
                ExecSpace::alignment));
            TimeOp<TData>::DoExplicit(initial, m_explicits[3], m_gamma);
        }
        if constexpr (IntOrder > 1)
        {
            InitMemoryRegionsSolution();
        }
        InitMemoryRegionsExplicit();

        // Restart from lower IMEX
        InitFromLowerOrderScheme(initial, time, step);
    }

    /*
     *  Setup gamma coefficient for IMEX.
     *  Note the extrapolation coefficients are defined inside the
     *  ExtrapolateKernel.
     */
    void SetCoefficients()
    {
        if constexpr (IntOrder == 1)
        {
            m_gamma = 1.0;
        }
        if constexpr (IntOrder == 2)
        {
            m_gamma = 3.0 / 2.0;
        }
        if constexpr (IntOrder == 3)
        {
            m_gamma = 11.0 / 6.0;
        }
        if constexpr (IntOrder == 4)
        {
            m_gamma = 25.0 / 12.0;
        }
    }

    /*
     *  Restart initialisation from lower order IMEX.
     *
     *  This is a workaround to avoid:
     *  instantiate another IMEXOpImpl, and add integration
     *  order to either Create() call or as template parameter. Also,
     *  we would need to give m_solutions to lower order IMEX via some
     *  SetSolutions(m_solutions) like call to avoid excessive memory
     *  allocations.
     */
    void InitFromLowerOrderScheme(Field<TData, FieldState::Phys> &initial,
                                  TData &time, size_t &step)
    {
        // Restart with IMEX1 if order > 1
        if constexpr (IntOrder > 1)
        {
            // Initialise 1st order IMEX and hand-over the m_solutions deque
            auto imex_one = std::make_unique<IMEXOpImpl<ExecSpace, TData, 1>>(
                this->m_expansionList);

            // Copy functors from outer/higher-order IMEX scheme
            imex_one->CopyFunctorsFrom(*this);

            // Move solutions to IMEX1
            imex_one->SetSolutions(this->TakeSolutions(),
                                   this->TakeExplicits());
            imex_one->InitMemoryRegionsSolution();
            imex_one->InitMemoryRegionsExplicit();

            // Advance in time with IMEX1
            imex_one->Apply(initial);

            // Move solutions back to this IMEX
            this->SetSolutions(imex_one->TakeSolutions(),
                               imex_one->TakeExplicits());

            // Increment step and time
            time += TimeOp<TData>::m_timestep;
            step++;
        }
        if constexpr (IntOrder > 2)
        {
            // Initialise 1st order IMEX and hand-over the m_solutions deque
            auto imex_two = std::make_unique<IMEXOpImpl<ExecSpace, TData, 2>>(
                this->m_expansionList);

            // Copy functors from outer/higher-order IMEX scheme
            imex_two->CopyFunctorsFrom(*this);

            // Move solutions to IMEX2
            imex_two->SetSolutions(this->TakeSolutions(),
                                   this->TakeExplicits());
            imex_two->InitMemoryRegionsSolution();
            imex_two->InitMemoryRegionsExplicit();

            // Advance in time with IMEX2
            imex_two->Apply(initial);

            // Move solutions back to higher-order IMEX
            this->SetSolutions(imex_two->TakeSolutions(),
                               imex_two->TakeExplicits());

            // Increment step and time
            time += TimeOp<TData>::m_timestep;
            step++;
        }
        if constexpr (IntOrder > 3)
        {
            // Initialise 1st order IMEX and hand-over the m_solutions deque
            auto imex_three = std::make_unique<IMEXOpImpl<ExecSpace, TData, 3>>(
                this->m_expansionList);

            // Copy functors from outer/higher-order IMEX scheme
            imex_three->CopyFunctorsFrom(*this);

            // Move solutions to IMEX3
            imex_three->SetSolutions(this->TakeSolutions(),
                                     this->TakeExplicits());
            imex_three->InitMemoryRegionsSolution();
            imex_three->InitMemoryRegionsExplicit();

            // Advance in time with IMEX3
            imex_three->Apply(initial);

            // Move solutions back to higher-order IMEX
            this->SetSolutions(imex_three->TakeSolutions(),
                               imex_three->TakeExplicits());

            // Increment step and time
            time += TimeOp<TData>::m_timestep;
            step++;
        }
    }

    /*
     *  Roll over solutions for next time step.
     *  The operation for 3rd order works as follows:
     *  Upon input:
     *  param inout:        u^n
     *  param m_solutions: [u^{n-1}, u^{n-2}]
     *  Upon output:
     *  param inout:        u^{n-2}
     *  param m_solutions: [u^{n}, u^{n-1}]
     *  Note that the first order schemes do not use this.
     */
    void RollOver(Field<TData, FieldState::Phys> &inout,
                  std::deque<Field<TData, FieldState::Phys>> &fieldDeque)
    {
        // Save last solution
        auto tmp = std::move(fieldDeque.back());

        // Remove last solution from deque
        fieldDeque.pop_back();

        // Add current solution to front
        fieldDeque.push_front(std::move(inout));

        // Move saved solution to inout
        inout = std::move(tmp);
    }

    /*
     *  Roll over explicit parts for next time step.
     *  The operation for 3rd order works as follows:
     *  Upon input:
     *  param m_explicits: [u^{n}, u^{n-1}, u^{n-2}]
     *  Upon output:
     *  param m_explicits: [u^{n-2}, u^{n}, u^{n-1}]
     *  Note that the first order schemes do not use this.
     */
    void RollOver(std::deque<Field<TData, FieldState::Phys>> &fieldDeque)
    {
        // Save last solution
        auto tmp = std::move(fieldDeque.back());

        // Remove last solution from back
        fieldDeque.pop_back();

        // Add last solution to front
        // (use inout as temporary storage of oldest solution)
        fieldDeque.push_front(std::move(tmp));
    }
};

} // namespace Nektar::Operators::detail
