///////////////////////////////////////////////////////////////////////////////
//
// File: MCNABOpImpl.hpp
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
#include "Operators/TimeOps/MCNAB/MCNABKernelLaunchers.hpp"
#include "Operators/TimeOps/MCNAB/MCNABOp.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, unsigned int IntOrder, typename TData>
class MCNABOpImpl : public MCNABOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

    // Compile-time check for valid integration order
    static_assert(IntOrder == 2,
                  "The MCNABOp class is only implemented for order 2.");

public:
    MCNABOpImpl(const ExpListSharedPtr &expansionList)
        : MCNABOp<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const ExpListSharedPtr &expansionList)
    {
        return std::make_unique<MCNABOpImpl<ExecSpace, IntOrder, TData>>(
            expansionList);
    }

protected:
    TData m_gamma = 9.0 / 16.0;

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that implicit function call is defined for MCNAB
        ASSERTL0(this->m_implicitFunctor,
                 "MCNAB schemes require a DoImplicit method. Define with "
                 "MCNABOp->DefineImplicit().");

        // Check that explicit function is defined for MCNAB
        ASSERTL0(this->m_explicitFunctor,
                 "MCNAB schemes require a DoExplicit method. Define with "
                 "MCNABOp->DefineExplicit().");

        // Startup
        if (this->m_step <= 1)
        {
            // Initialise IMEX and hand-over the m_solutions deque
            auto startup = IMEXOp<TData>::Create(this->m_expansionList, 1,
                                                 ExecSpace::name);
            startup->SaveImplicit(true);

            // Copy functors from outer/higher-order CNAB scheme
            startup->CopyFunctorsFrom(*this);

            // Move solutions to startup
            startup->SetImplicits(this->TakeImplicits());
            startup->SetExplicits(this->TakeExplicits());
            startup->SetSolutions(this->TakeSolutions());

            // Advance in time with startup
            startup->SetTime(this->m_time);
            startup->SetStep(this->m_step);
            startup->Apply(inout);

            // Move solutions back to this MCNAB
            this->SetImplicits(startup->TakeImplicits());
            this->SetExplicits(startup->TakeExplicits());
            this->SetSolutions(startup->TakeSolutions());

            // Allocate new storage
            if (this->m_step == 0)
            {
                this->m_implicits.push_back(
                    Field<TData, FieldState::Phys>::Create(
                        GetBlockAttributes<TData>(FieldState::Phys,
                                                  this->m_expansionList),
                        inout.GetNumComponents(), inout.GetNumHomoModes(),
                        ExecSpace::alignment));
            }

            // Increment step and time
            this->m_time += this->m_timestep;
            this->m_step++;
        }
        // After startup
        else
        {
            // After startup
            if (this->m_explicits.size() < IntOrder)
            {
                // Allocate new storage
                this->m_explicits.push_back(
                    Field<TData, FieldState::Phys>::Create(
                        GetBlockAttributes<TData>(FieldState::Phys,
                                                  this->m_expansionList),
                        inout.GetNumComponents(), inout.GetNumHomoModes(),
                        ExecSpace::alignment));
            }

            this->RollOver(this->m_explicits);

            this->DoExplicit(inout, this->m_explicits[0], this->m_time,
                             this->m_timestep);

            // Do extrapolation.
            UpdateSolution(inout);

            // Rollover previous solutions
            this->RollOver(this->m_implicits);

            // Compute next time step
            this->m_implicits[0].template Copy<MemSpace>(inout);

            this->DoImplicit(this->m_implicits[0], inout,
                             this->m_time + this->m_timestep,
                             m_gamma * this->m_timestep);
            sub<ExecSpace>(inout, this->m_implicits[0], this->m_implicits[0]);
            mul<ExecSpace>(1.0 / m_gamma, this->m_implicits[0],
                           this->m_implicits[0]);

            // Increment step and time
            this->m_time += this->m_timestep;
            this->m_step++;
        }
    }

    void UpdateSolution(Field<TData, FieldState::Phys> &inout)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            // Determine shape and type of the element.
            auto &inoutBlock = inout.GetBlocks()[blk];
            auto nelmt       = inoutBlock.GetNumElementsWithPadding();
            auto nphys =
                inoutBlock.GetNumData() * inoutBlock.GetNumComponents();

            UpdateSolutionKernel<ExecSpace, MCNABscheme>(
                nphys * nelmt,
                inoutBlock.template GetPtr<MemSpace, ReadWrite>(),
                this->m_implicits[0]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>(),
                this->m_implicits[1]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>(),
                this->m_explicits[0]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>(),
                this->m_explicits[1]
                    .GetBlocks()[blk]
                    .template GetPtr<MemSpace, ReadOnly>());
        }
    }
};

} // namespace Nektar::Operators::detail
