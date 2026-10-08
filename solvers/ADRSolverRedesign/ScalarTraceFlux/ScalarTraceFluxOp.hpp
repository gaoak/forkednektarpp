///////////////////////////////////////////////////////////////////////////////
//
// File: ScalarTraceFluxOp.hpp
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
// Description: Interface of the scalar trace flux for the ADR solver.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_ADRSOLVERREDESIGN_SCALARTRACEFLUXOP_HPP
#define NEKTAR_SOLVERS_ADRSOLVERREDESIGN_SCALARTRACEFLUXOP_HPP

#include "SolverCore/TraceFlux/TraceFluxOp.hpp"

namespace Nektar
{

/**
 * @brief The scalar equation's trace flux.
 *
 * @details
 * Holds the operator's registration name, which the generated factory
 * declarations build their key from, and the Create() a solver calls to ask
 * for the implementation of a given Riemann solver.
 */
template <typename TData>
class ScalarTraceFluxOp : public SolverCore::TraceFluxOp<TData>
{
public:
    /**
     * @brief Create the implementation registered for @p solverType.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components.
     * @param   solverType      Riemann solver to use; empty takes the
     *                          session's UpwindType.
     * @param   execStr         Execution space; empty takes the session's.
     */
    static std::shared_ptr<ScalarTraceFluxOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &solverType = "", const std::string &execStr = "")
    {
        // The base resolves the session's UpwindType only for an empty key,
        // which the name makes this one never be, so resolve it here.
        std::string solverType0 = solverType;
        auto session            = expansionList->GetSession();
        if (solverType0 == "" && session->DefinesSolverInfo("UpwindType"))
        {
            solverType0 = session->GetSolverInfo("UpwindType");
        }

        return std::dynamic_pointer_cast<ScalarTraceFluxOp<TData>>(
            SolverCore::TraceFluxOp<TData>::Create(
                expansionList, components, name + solverType0, execStr));
    }

    static inline const std::string name = "ScalarTraceFlux";

protected:
    ScalarTraceFluxOp(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components)
        : SolverCore::TraceFluxOp<TData>(expansionList, components)
    {
    }

    ~ScalarTraceFluxOp() override = default;
};

} // namespace Nektar

#endif
