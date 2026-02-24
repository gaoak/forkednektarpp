///////////////////////////////////////////////////////////////////////////////
//
// File: HLLSolverOp.hpp
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

#include "CompressibleFlowSolverRedesign/RiemannSolvers/CompressibleSolverOp.hpp"

namespace Nektar::Operators
{

// HLL base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class HLLSolverOp : public CompressibleSolverOp<TData>
{
public:
    static std::shared_ptr<HLLSolverOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return std::dynamic_pointer_cast<HLLSolverOp<TData>>(
            CompressibleSolverOp<TData>::template Create<HLLSolverOp>(
                expansionList, components, name, execStr));
    }

    static inline const std::string name = "HLL";

protected:
    HLLSolverOp(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : CompressibleSolverOp<TData>(expansionList, components)
    {
    }

    ~HLLSolverOp() override = default;
};

} // namespace Nektar::Operators
