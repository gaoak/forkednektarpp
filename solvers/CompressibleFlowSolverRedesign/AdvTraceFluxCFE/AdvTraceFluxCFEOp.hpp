///////////////////////////////////////////////////////////////////////////////
//
// File: AdvTraceFluxCFEOp.hpp
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
// Description: Interface of the inviscid trace flux for the compressible
// Euler equations.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "SolverCore/TraceFlux/TraceFluxOp.hpp"

namespace Nektar
{

/**
 * @brief The compressible Euler equations' inviscid trace flux.
 *
 * @details
 * Holds the operator's registration name, which the generated factory
 * declarations build their key from, the Create() a solver calls to ask for
 * the implementation of a given Riemann solver and equation of state, and the
 * single-shot evaluation below.
 */
template <typename TData>
class AdvTraceFluxCFEOp : public SolverCore::TraceFluxOp<TData>
{
public:
    /**
     * @brief Create the implementation registered for @p solverType.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components.
     * @param   solverType      Riemann solver and equation of state to use,
     *                          as the generated registration spells them.
     * @param   execStr         Execution space; empty takes the session's.
     */
    static std::shared_ptr<AdvTraceFluxCFEOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &solverType = "", const std::string &execStr = "")
    {
        return std::dynamic_pointer_cast<AdvTraceFluxCFEOp<TData>>(
            SolverCore::TraceFluxOp<TData>::Create(expansionList, components,
                                                   name + solverType, execStr));
    }

    /**
     * @brief Apply the Riemann solve to trace states given directly; for the
     * unit tests.
     *
     * Runs the Riemann solve, and the rotation into and out of the trace
     * normal frame around it, on @p fwd and @p bwd as supplied, writing
     * @p flux. Unlike Apply() it neither gathers the states off the field nor
     * scatters the result back, which is what the Riemann unit tests need:
     * a flux for a pair of states they chose. The name says who it is for;
     * a solver applies the operator through Apply().
     *
     * @param   fwd     Forward-side conserved states on the trace.
     * @param   bwd     Backward-side conserved states on the trace.
     * @param   flux    Flux on the trace, in Cartesian components.
     */
    void ApplyUnitTest(LibUtilities::Field<TData, FieldState::Phys> &fwd,
                       LibUtilities::Field<TData, FieldState::Phys> &bwd,
                       LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_ApplyUnitTest(fwd, bwd, flux);
    }

    static inline const std::string name = "AdvTraceFluxCFE";

protected:
    AdvTraceFluxCFEOp(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components)
        : SolverCore::TraceFluxOp<TData>(expansionList, components)
    {
    }

    ~AdvTraceFluxCFEOp() override = default;

    virtual void v_ApplyUnitTest(
        LibUtilities::Field<TData, FieldState::Phys> &fwd,
        LibUtilities::Field<TData, FieldState::Phys> &bwd,
        LibUtilities::Field<TData, FieldState::Phys> &flux) = 0;
};

} // namespace Nektar
