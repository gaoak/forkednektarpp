///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionScalarIPTraceFluxOp.hpp
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
// Description: Scalar IP diffusion trace flux operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/SolverUtilsOps/Diffusion/DiffusionTraceFluxOp.hpp"

namespace Nektar::Operators
{

template <typename TData>
class DiffusionScalarIPTraceFluxOp : public ScalarIPDiffusionTraceFluxOp<TData>
{
public:
    static std::shared_ptr<DiffusionScalarIPTraceFluxOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return std::dynamic_pointer_cast<DiffusionScalarIPTraceFluxOp<TData>>(
            FluxOp<TData>::Create(expansionList, components, name, execStr));
    }

    static inline const std::string name = "DiffusionScalarIPTraceFlux";

protected:
    DiffusionScalarIPTraceFluxOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : ScalarIPDiffusionTraceFluxOp<TData>(expansionList, components)
    {
        auto session = expansionList->GetSession();
        session->LoadParameter("IPPenaltyCoeff", m_IPPenaltyCoeff, 4.0);
    }

    ~DiffusionScalarIPTraceFluxOp() override = default;
    TData m_IPPenaltyCoeff                   = 4.0;
};

} // namespace Nektar::Operators
