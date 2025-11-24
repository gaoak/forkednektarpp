///////////////////////////////////////////////////////////////////////////////
//
// File: IMEXGEMOpImpl.hpp
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

#include "Operators/TimeOps/GEM/IMEXGEM/IMEXGEMOp.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class IMEXGEMOpImpl : public IMEXGEMOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IMEXGEMOpImpl(const ExpListSharedPtr &expansionList)
        : IMEXGEMOp<TData>(expansionList)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const ExpListSharedPtr &expansionList)
    {
        return std::make_unique<IMEXGEMOpImpl<ExecSpace, TData>>(expansionList);
    }

protected:
    std::shared_ptr<TimeOp<TData>> m_stepper;

    void v_Apply(Field<TData, FieldState::Phys> &inout) override
    {
        // Initialize.
        if (!this->m_initialized)
        {
            const unsigned int npts = (this->m_variant == "Midpoint")
                                          ? this->m_order / 2
                                          : this->m_order;
            for (unsigned int m = 0; m < npts; ++m)
            {
                this->m_T.push_back(Field<TData, FieldState::Phys>(
                    GetBlockAttributes<TData>(FieldState::Phys,
                                              this->m_expansionList),
                    inout.GetNumComponents(), inout.GetNumHomoModes()));

                this->m_T0.push_back(Field<TData, FieldState::Phys>(
                    GetBlockAttributes<TData>(FieldState::Phys,
                                              this->m_expansionList),
                    inout.GetNumComponents(), inout.GetNumHomoModes()));
            }

            // Initialise IMEXdirk scheme.
            const std::string variant =
                (this->m_variant == "Midpoint") ? "12" : "11";
            const unsigned int order = (this->m_variant == "Midpoint") ? 2 : 1;
            this->m_stepper          = TimeOp<TData>::Create(
                this->m_expansionList, "IMEXdirk", order, variant,
                std::vector<TData>{}, ExecSpace::name);
            this->m_stepper->CopyFunctorsFrom(*this);

            this->m_initialized = true;
        }

        // Compute approximation.
        const unsigned int npts =
            (this->m_variant == "Midpoint") ? this->m_order / 2 : this->m_order;
        for (unsigned int m = 1; m <= npts; ++m)
        {
            this->m_stepper->SetTime(this->m_time);
            this->m_stepper->SetTimeStep(this->m_timestep / m);
            this->m_stepper->SetStep(0);
            this->m_T0[m - 1].template Copy<MemSpace>(inout);
            for (unsigned int k = 1; k <= m; ++k)
            {
                this->m_stepper->Apply(this->m_T0[m - 1]);
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
