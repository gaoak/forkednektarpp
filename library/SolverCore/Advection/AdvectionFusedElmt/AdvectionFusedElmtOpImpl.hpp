///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionFusedElmtOpImpl.hpp
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
// Description: Whole-field implementation of the fused element-local advection
// operator, registered per execution space.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtOp.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief The whole-field fused element-local advection operator registered
 * with the operator factory, one per execution space; the work is in
 * AdvectionFusedElmtOp and its block operators.
 */
template <typename ExecSpace, typename TData>
class AdvectionFusedElmtOpImpl : public AdvectionFusedElmtOp<TData>
{
public:
    AdvectionFusedElmtOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : AdvectionFusedElmtOp<TData>(expansionList, components)
    {
    }

    static std::string className;

    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AdvectionFusedElmtOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }
};

} // namespace Nektar::SolverCore::detail
