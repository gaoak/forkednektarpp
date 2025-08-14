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

#include "Operators/TimeOps/BDF/BDFDeviceKernels.hpp"
#include "Operators/TimeOps/BDF/BDFSerialAVXKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData, unsigned int IntOrder>
class BDFOpImpl : public BDFOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order
    static_assert(IntOrder >= 1 && IntOrder <= 4,
                  "The BDFOp class is only implemented for order 1-4.");

public:
    BDFOpImpl(const ExpListSharedPtr &expansionList)
        : BDFOp<TData>(expansionList)
    {
        // ASSERTL0(
        //     TimeOp<TData>::m_intMethod == "BDFImplicit" ||
        //         TimeOp<TData>::m_intMethod == "BDF" ||
        //         (TimeOp<TData>::m_intMethod == "Euler" &&
        //          TimeOp<TData>::m_intVariant == "Backward"),
        //     "This class must be called for the method 'BDF', 'BDFImplicit' or
        //     "
        //     "'Euler' with variant 'Backward'. Instead it "
        //     "was called with " +
        //         TimeOp<TData>::m_intMethod + " and variant " +
        //         TimeOp<TData>::m_intVariant + ".");
        //
        // ASSERTL0(1 <= TimeOp<TData>::m_intOrder &&
        //              TimeOp<TData>::m_intOrder <= 4,
        //          "The BDFOp class is only implemented for order 1-4.");

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
                return std::make_unique<BDFOpImpl<ExecSpace, TData, 1>>(
                    expansionList);
            case 2:
                return std::make_unique<BDFOpImpl<ExecSpace, TData, 2>>(
                    expansionList);
            case 3:
                return std::make_unique<BDFOpImpl<ExecSpace, TData, 3>>(
                    expansionList);
            case 4:
                return std::make_unique<BDFOpImpl<ExecSpace, TData, 4>>(
                    expansionList);
            default:
                NEKERROR(ErrorUtil::efatal,
                         "BDF order must be between 1 and 4. "
                         "Instead it is " +
                             std::to_string(order));
                return nullptr;
        }
    }

    // Move‐in
    void SetSolutions(std::deque<Field<TData, FieldState::Phys>> &&solutions)
    {
        ASSERTL0(m_solutions.empty(),
                 "Do not call SetSolutions() if m_solutions is "
                 "already defined in BDF operator.")
        m_solutions = std::move(solutions);
    }

    void InitMemoryRegions()
    {
        ASSERTL0(!m_solutions.empty(),
                 "Do not call InitMemoryRegions() if m_solutions is not "
                 "defined in BDF operator.")

        m_solutions_mr = MemoryRegion<const TData *>::Create(
            m_solutions.size(), ExecSpace::alignment);
    }

    // Move‐out
    std::deque<Field<TData, FieldState::Phys>> TakeSolutions()
    {
        return std::move(m_solutions);
    }

protected:
    // Extrapolation coefficient of implicit scheme
    TData m_gamma;

    // Storage for previous solutions in Fields
    // and memory region for pointer access on device
    std::deque<Field<TData, FieldState::Phys>> m_solutions;
    MemoryRegion<const TData *> m_solutions_mr;

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that implicit function call is defined for BDF
        ASSERTL0(TimeOp<TData>::m_implicitFunctor,
                 "BDF schemes "
                 "require a DoImplicit method. Define with "
                 "BDFOp->DefineImplicit().");

        // Check that explicit function is not defined, because it will be
        // ignored
        ASSERTL0(!TimeOp<TData>::m_explicitFunctor,
                 "BDF schemes "
                 "use only a DoImplicit method. Any method defined with "
                 "BDFOp->DefineExplicit() will be ignored.");

        // For first order: inout = 1 / \Delta t * u^n
        if constexpr (IntOrder == 1)
        {
            mul<ExecSpace, TData>(1. / TimeOp<TData>::m_timestep, inout, inout);
        }

        // Update next time step in solution array
        // The if condition ensures that the restart from lower order BDF works
        if constexpr (IntOrder > 1)
        {
            // Rollover previous solutions in m_solutions and inout
            // (use inout as temporary storage of oldest solution)
            RollOver(inout);

            // Extrapolate previous solutions.
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

                // Gather pointers for previous solutions.
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

                // Do extrapolation.
                ExtrapolateKernel<ExecSpace, TData, IntOrder>(
                    nphys * nelmt, TimeOp<TData>::m_timestep, solutionsDevice,
                    inoutPtr);
            }
        }

        // Compute next time step
        TimeOp<TData>::DoImplicit(inout, m_gamma);
    }

    // Initialise the time-stepping array
    void v_Initialise(Field<TData, FieldState::Phys> &initial, TData &time,
                      size_t &step) override
    {

        // Initialise all previous time steps to initial condition
        if constexpr (IntOrder > 1)
        {
            m_solutions.push_back(Field<TData, FieldState::Phys>::Create(
                "timestep n-1",
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList),
                initial.GetNumComponents(), initial.GetNumHomoModes(),
                ExecSpace::alignment));
            m_solutions[0].template Copy<MemSpace>(initial);
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
        }
        if constexpr (IntOrder > 1)
        {
            InitMemoryRegions();
        }

        // Restart from lower BDF
        InitFromLowerOrderScheme(initial, time, step);
    }

    /*
     *  Setup gamma coefficient for BDF.
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
     *  Restart initialisation from lower order BDF.
     *
     *  This is a workaround to avoid:
     *  instantiate another BDFOpImpl, and add integration
     *  order to either Create() call or as template parameter. Also,
     *  we would need to give m_solutions to lower order BDF via some
     *  SetSolutions(m_solutions) like call to avoid excessive memory
     *  allocations.
     */
    void InitFromLowerOrderScheme(Field<TData, FieldState::Phys> &initial,
                                  TData &time, size_t &step)
    {
        // Restart with BDF1 if order > 1
        if constexpr (IntOrder > 1)
        {
            // Initialise 1st order BDF and hand-over the m_solutions deque
            auto bdf_one = std::make_unique<BDFOpImpl<ExecSpace, TData, 1>>(
                this->m_expansionList);

            // Copy functors from outer/higher-order BDF scheme
            bdf_one->CopyFunctorsFrom(*this);

            // Note Move solutions and InitMemoryRegions not required for 1st
            // order because we do not call extrapolate.

            // Advance in time with BDF1
            bdf_one->Apply(initial);

            // Increment step and time
            time += TimeOp<TData>::m_timestep;
            step++;
        }
        if constexpr (IntOrder > 2)
        {
            // Initialise 1st order BDF and hand-over the m_solutions deque
            auto bdf_two = std::make_unique<BDFOpImpl<ExecSpace, TData, 2>>(
                this->m_expansionList);

            // Copy functors from outer/higher-order BDF scheme
            bdf_two->CopyFunctorsFrom(*this);

            // Move solutions to BDF2
            bdf_two->SetSolutions(this->TakeSolutions());
            bdf_two->InitMemoryRegions();

            // Advance in time with BDF2
            bdf_two->Apply(initial);

            // Move solutions back to higher-order BDF
            this->SetSolutions(bdf_two->TakeSolutions());

            // Increment step and time
            time += TimeOp<TData>::m_timestep;
            step++;
        }
        if constexpr (IntOrder > 3)
        {
            // Initialise 1st order BDF and hand-over the m_solutions deque
            auto bdf_three = std::make_unique<BDFOpImpl<ExecSpace, TData, 3>>(
                this->m_expansionList);

            // Copy functors from outer/higher-order BDF scheme
            bdf_three->CopyFunctorsFrom(*this);

            // Move solutions to BDF3
            bdf_three->SetSolutions(this->TakeSolutions());
            bdf_three->InitMemoryRegions();

            // Advance in time with BDF3
            bdf_three->Apply(initial);

            // Move solutions back to higher-order BDF
            this->SetSolutions(bdf_three->TakeSolutions());

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
     *  Note that the first order BDF does not use this.
     */
    void RollOver(Field<TData, FieldState::Phys> &inout)
    {
        // Save last solution
        auto tmp = std::move(m_solutions.back());

        // Remove last solution from deque
        m_solutions.pop_back();

        // Add current solution to front
        m_solutions.push_front(std::move(inout));

        // Move saved solution to inout
        // (use inout as temporary storage of oldest solution)
        inout = std::move(tmp);
    }
};

} // namespace Nektar::Operators::detail
