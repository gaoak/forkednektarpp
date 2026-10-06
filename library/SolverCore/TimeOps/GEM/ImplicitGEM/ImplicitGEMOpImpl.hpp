///////////////////////////////////////////////////////////////////////////////
//
// File: ImplicitGEMOpImpl.hpp
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

#include "SolverCore/TimeOps/GEM/ImplicitGEM/ImplicitGEMOp.hpp"

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class ImplicitGEMOpImpl : public ImplicitGEMOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ImplicitGEMOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components,
                      const unsigned int &order, const std::string &variant)
        : ImplicitGEMOp<TData>(expansionList, components, order, variant)
    {
        unsigned int n =
            (this->m_variant == "Midpoint") ? this->m_order / 2 : this->m_order;

        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                this->m_expansionList,
                MultiRegions::Operator<TData>::GetDefaultInterleaveWidth(
                    this->m_expansionList->GetSession()));
        for (unsigned int m = 0; m < n; ++m)
        {
            this->m_T.push_back(LibUtilities::Field<TData, FieldState::Phys>(
                blockAttr, this->m_components, 1));

            this->m_T0.push_back(LibUtilities::Field<TData, FieldState::Phys>(
                blockAttr, this->m_components, 1));
        }

        for (unsigned int m = 0; m < this->m_order; ++m)
        {
            this->m_solutions.push_back(
                LibUtilities::Field<TData, FieldState::Phys>(
                    blockAttr, this->m_components, 1));
        }

        this->m_implicits.push_back(
            LibUtilities::Field<TData, FieldState::Phys>(
                blockAttr, this->m_components, 1));
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<TimeOp<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components, const unsigned int &order,
        const std::string &variant)
    {
        return std::make_unique<ImplicitGEMOpImpl<ExecSpace, TData>>(
            expansionList, components, order, variant);
    }

protected:
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(this->m_implicitFunctor,
                 "ImplicitGEM schemes require a DoImplicit method. Define with "
                 "ImplicitGEMOp->DefineImplicit().");

        if (this->m_variant == "")
        {
            // Compute first order approximation.
            for (unsigned int m = 1; m <= this->m_order; ++m)
            {
                this->DoImplicit(inout, this->m_solutions[0], this->m_time,
                                 this->m_timestep / m);
                for (unsigned int k = 2; k <= m; ++k)
                {
                    this->DoImplicit(this->m_solutions[k - 2],
                                     this->m_solutions[k - 1],
                                     this->m_time + k * (this->m_timestep / m),
                                     this->m_timestep / m);
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
                for (unsigned int k = 1; k <= m; ++k)
                {
                    if (k == 1)
                    {
                        this->DoImplicit(inout, this->m_solutions[0],
                                         this->m_time +
                                             0.25 * this->m_timestep / m,
                                         0.25 * this->m_timestep / m);
                        Math::mul<ExecSpace>((TData)2.0, this->m_solutions[0],
                                             this->m_solutions[1]);
                        Math::daxpy<ExecSpace>(-(TData)1.0, inout,
                                               this->m_solutions[1],
                                               this->m_solutions[1]);
                    }
                    else
                    {
                        this->DoImplicit(this->m_solutions[2 * k - 3],
                                         this->m_solutions[2 * k - 2],
                                         this->m_time + (k - 1 + 0.25) *
                                                            this->m_timestep /
                                                            m,
                                         0.25 * this->m_timestep / m);
                        Math::mul<ExecSpace>((TData)2.0,
                                             this->m_solutions[2 * k - 2],
                                             this->m_solutions[2 * k - 1]);
                        Math::daxpy<ExecSpace>(-(TData)1.0,
                                               this->m_solutions[2 * k - 3],
                                               this->m_solutions[2 * k - 1],
                                               this->m_solutions[2 * k - 1]);
                    }
                    this->DoImplicit(
                        this->m_solutions[2 * k - 1], this->m_implicits[0],
                        this->m_time + (k - 0.25) * (this->m_timestep / m),
                        0.25 * this->m_timestep / m);
                    Math::sub<ExecSpace>(this->m_implicits[0],
                                         this->m_solutions[2 * k - 1],
                                         this->m_implicits[0]);
                    Math::daxpy<ExecSpace>((TData)2.0, this->m_implicits[0],
                                           this->m_solutions[2 * k - 1],
                                           this->m_solutions[2 * k - 1]);
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

} // namespace Nektar::SolverCore::detail
