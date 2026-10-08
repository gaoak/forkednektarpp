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
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Description: Interface of the interior-penalty diffusion trace flux for
// the ADR solver.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "SolverCore/TraceFlux/TraceFluxOp.hpp"

namespace Nektar
{

/**
 * @brief The interior-penalty diffusion trace flux for a scalar field.
 *
 * @details
 * Holds the operator's registration name, which the generated factory
 * declarations build their key from, and the Create() a solver calls to ask
 * for the implementation.
 */
template <typename TData>
class DiffusionScalarIPTraceFluxOp : public SolverCore::TraceFluxOp<TData>
{
public:
    /**
     * @brief Create the implementation for the requested execution space.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components.
     * @param   execStr         Execution space; empty takes the session's.
     */
    static std::shared_ptr<DiffusionScalarIPTraceFluxOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return std::dynamic_pointer_cast<DiffusionScalarIPTraceFluxOp<TData>>(
            SolverCore::TraceFluxOp<TData>::Create(expansionList, components,
                                                   name, execStr));
    }

    static inline const std::string name = "DiffusionScalarIPTraceFlux";

protected:
    DiffusionScalarIPTraceFluxOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : SolverCore::TraceFluxOp<TData>(expansionList, components)
    {
    }

    ~DiffusionScalarIPTraceFluxOp() override = default;
};

} // namespace Nektar
