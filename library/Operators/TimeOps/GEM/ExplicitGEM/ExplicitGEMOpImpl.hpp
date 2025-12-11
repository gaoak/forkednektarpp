///////////////////////////////////////////////////////////////////////////////
//
// File: ExplicitGEMOpImpl.hpp
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

#include "Operators/TimeOps/GEM/ExplicitGEM/ExplicitGEMOp.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class ExplicitGEMOpImpl : public ExplicitGEMOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ExplicitGEMOpImpl(const ExpListSharedPtr &expansionList)
        : ExplicitGEMOp<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const ExpListSharedPtr &expansionList)
    {
        return std::make_unique<ExplicitGEMOpImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(
            this->m_explicitRhsFunctor,
            "ExplicitGEM schemes require a DoExplicitRhs method. Define with "
            "ExplicitGEMOp->DefineExplicitRhs().");
        ASSERTL0(
            this->m_projectionFunctor,
            "ExplicitGEM schemes require a DoProjection method. Define with "
            "ExplicitGEMOp->DefineProjection().");

        // Initialize.
        if (!this->m_initialized)
        {
            auto blockAttr = GetBlockAttributes<TData, FieldState::Phys>(
                this->m_expansionList);
            unsigned int n = (this->m_variant == "Midpoint") ? this->m_order / 2
                                                             : this->m_order;
            for (unsigned int m = 0; m < n; ++m)
            {
                this->m_T.push_back(Field<TData, FieldState::Phys>(
                    blockAttr, inout.GetNumComponents(),
                    inout.GetNumHomoModes()));

                this->m_T0.push_back(Field<TData, FieldState::Phys>(
                    blockAttr, inout.GetNumComponents(),
                    inout.GetNumHomoModes()));
            }

            for (unsigned int m = 0; m < this->m_order; ++m)
            {
                this->m_solutions.push_back(Field<TData, FieldState::Phys>(
                    blockAttr, inout.GetNumComponents(),
                    inout.GetNumHomoModes()));
            }

            this->m_explicits.push_back(Field<TData, FieldState::Phys>(
                blockAttr, inout.GetNumComponents(), inout.GetNumHomoModes()));

            this->m_explicits.push_back(Field<TData, FieldState::Phys>(
                blockAttr, inout.GetNumComponents(), inout.GetNumHomoModes()));

            this->m_initialized = true;
        }

        this->DoExplicitRhs(inout, this->m_explicits[0], this->m_time,
                            this->m_timestep);

        if (this->m_variant == "")
        {
            // Compute first order approximation.
            for (unsigned int m = 1; m <= this->m_order; ++m)
            {
                for (unsigned int k = 1; k <= m; ++k)
                {
                    // For the first stage, used pre-computed rhs.
                    if (k == 1)
                    {
                        daxpy<ExecSpace>((TData)1.0 / m, this->m_explicits[0],
                                         inout, this->m_solutions[k - 1]);
                    }
                    // For other stages, compute new rhs.
                    else
                    {
                        this->DoExplicitRhs(
                            this->m_solutions[k - 2], this->m_explicits[1],
                            this->m_time + (k - 1) * (this->m_timestep / m),
                            this->m_timestep);
                        daxpy<ExecSpace>((TData)1.0 / m, this->m_explicits[1],
                                         this->m_solutions[k - 2],
                                         this->m_solutions[k - 1]);
                    }

                    this->DoProjection(
                        this->m_solutions[k - 1], this->m_solutions[k - 1],
                        this->m_time + k * (this->m_timestep / m));
                }

                // Save solution to m_T0.
                this->m_T0[m - 1].template Copy<MemSpace>(
                    this->m_solutions[m - 1]);
            }
        }
        else if (this->m_variant == "Midpoint")
        {
            // Compute second order approximation
            for (unsigned int m = 1; m <= this->m_order / 2; ++m)
            {
                // Use precomputed rhs for initial Euler stage
                daxpy<ExecSpace>((TData)1.0 / (2 * m), this->m_explicits[0],
                                 inout, this->m_solutions[0]);
                this->DoProjection(this->m_solutions[0], this->m_solutions[0],
                                   this->m_time + this->m_timestep / (2 * m));

                // Compute new rhs for midpoint stage
                for (unsigned int k = 2; k <= 2 * m; ++k)
                {
                    this->DoExplicitRhs(
                        this->m_solutions[k - 2], this->m_explicits[1],
                        this->m_time + (k - 1) * (this->m_timestep / (2 * m)),
                        this->m_timestep);
                    daxpy<ExecSpace>((TData)1.0 / m, this->m_explicits[1],
                                     (k == 2) ? inout
                                              : this->m_solutions[k - 3],
                                     this->m_solutions[k - 1]);
                    this->DoProjection(
                        this->m_solutions[k - 1], this->m_solutions[k - 1],
                        this->m_time + k * (this->m_timestep / (2 * m)));
                }

                // Save solution to m_T0
                this->m_T0[m - 1].template Copy<MemSpace>(
                    this->m_solutions[2 * m - 1]);
            }
        }

        // Extrapolate solution.
        this->template ExtrapolateSolution<ExecSpace>(inout);

        // Increment step and time.
        this->m_time += this->m_timestep;
        this->m_step++;
    }
};

} // namespace Nektar::Operators::detail
