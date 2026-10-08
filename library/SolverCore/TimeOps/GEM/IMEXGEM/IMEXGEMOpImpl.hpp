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

#include "SolverCore/TimeOps/GEM/IMEXGEM/IMEXGEMOp.hpp"

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class IMEXGEMOpImpl : public IMEXGEMOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IMEXGEMOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components,
                  const unsigned int &order, const std::string &variant)
        : IMEXGEMOp<TData>(expansionList, components, order, variant)
    {
        const unsigned int npts =
            (this->m_variant == "Midpoint") ? this->m_order / 2 : this->m_order;

        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                this->m_expansionList,
                MultiRegions::Operator<TData>::GetDefaultInterleaveWidth(
                    this->m_expansionList->GetSession()));
        for (unsigned int m = 0; m < npts; ++m)
        {
            this->m_T.push_back(LibUtilities::Field<TData, FieldState::Phys>(
                blockAttr, this->m_components, 1));

            this->m_T0.push_back(LibUtilities::Field<TData, FieldState::Phys>(
                blockAttr, this->m_components, 1));
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<TimeOp<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components, const unsigned int &order,
        const std::string &variant)
    {
        return std::make_unique<IMEXGEMOpImpl<ExecSpace, TData>>(
            expansionList, components, order, variant);
    }

protected:
    std::shared_ptr<TimeOp<TData>> m_stepper = nullptr;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &inout) override
    {
        // Initialise IMEXdirk scheme.
        if (!this->m_stepper)
        {
            const std::string stepperVariant =
                (this->m_variant == "Midpoint") ? "12" : "11";
            const unsigned int stepperOrder =
                (this->m_variant == "Midpoint") ? 2 : 1;
            this->m_stepper =
                TimeOp<TData>::Create(this->m_expansionList, this->m_components,
                                      "IMEXdirk", stepperOrder, stepperVariant,
                                      std::vector<TData>{}, ExecSpace::name);
            this->m_stepper->CopyFunctorsFrom(*this);
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

} // namespace Nektar::SolverCore::detail
