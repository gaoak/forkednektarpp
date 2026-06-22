///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionIPOp.hpp
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

#include "Operators/SolverUtilsOps/Diffusion/DiffusionOp.hpp"

namespace Nektar::Operators
{

// DiffusionIP base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class DiffusionIPOp : public DiffusionOp<TData>
{
public:
    static std::shared_ptr<DiffusionIPOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operator<TData>::template Create<DiffusionIPOp>(
            expansionList, components, execStr);
    }

    static inline const std::string name = "DiffusionIP";

    void SetAppend(const bool &append)
    {
        v_SetAppend(append);
    }

protected:
    DiffusionIPOp(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : DiffusionOp<TData>(expansionList, components)
    {
        auto session = expansionList->GetSession();

        session->LoadParameter("IPSymmFluxCoeff", m_IPSymmFluxCoeff, 0.0);
        session->LoadParameter("IP2ndDervCoeff", m_IP2ndDervCoeff, 0.0);
        session->LoadParameter("IPPenaltyCoeff", m_IPPenaltyCoeff, 4.0);
    }

    ~DiffusionIPOp() override = default;

    virtual void v_SetAppend(const bool &append) = 0;

    double m_IPSymmFluxCoeff = 0.0;
    double m_IP2ndDervCoeff  = 0.0;
    double m_IPPenaltyCoeff  = 4.0;
};

} // namespace Nektar::Operators
