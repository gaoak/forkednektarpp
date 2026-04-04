///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyAdvection.h
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
// Description: Unsteady advection solve routines
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_REDESIGNSOLVER_EQUATIONSYSTEMS_UNSTEADYADVECTION_H
#define NEKTAR_SOLVERS_REDESIGNSOLVER_EQUATIONSYSTEMS_UNSTEADYADVECTION_H

#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysOp.hpp"
#include "Operators/Math/Math.hpp"
#include <Operators/ElmtOps/Expression/ExpressionOp.hpp>
#include <Operators/Field/Field.hpp>
#include <Operators/SolverUtilsOps/Advection/AdvectionWeakDG/AdvectionWeakDGOp.hpp>
#include <Operators/SolverUtilsOps/Advection/VolumeFluxOp.hpp>
#include <Operators/SolverUtilsOps/RiemannSolvers/RiemannSolverOp.hpp>
#include <SolverUtils/EquationSystem.h>

namespace Nektar
{
using namespace SolverUtils;
using namespace Operators;

class UnsteadyAdvection : public EquationSystem
{
public:
    friend class MemoryManager<UnsteadyAdvection>;

    /// Creates an instance of this class
    static EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        EquationSystemSharedPtr p =
            MemoryManager<UnsteadyAdvection>::AllocateSharedPtr(pSession,
                                                                pGraph);
        p->InitObject();
        return p;
    }

    /// Name of class
    static std::string className;

protected:
    // Save variable strings and number for verbose output and looping
    std::vector<std::string> m_variables;
    unsigned int m_nVariables;
    unsigned int m_ndim;

    // Setup workspaces
    Field<double, FieldState::Phys> m_in;
    Field<double, FieldState::Phys> m_advectVel;
    Field<double, FieldState::Phys> m_traceAdvectVel;

    // Declare math
    Math m_math;

    // Time-integration
    std::shared_ptr<TimeOp<double>> m_timeOp;

    // Initialise operators
    std::shared_ptr<AdvectionWeakDGOp<double>> m_advectionWeakDGOp;
    std::shared_ptr<VolumeFluxOp<double>> m_volumeFluxOp;
    std::shared_ptr<RiemannSolverOp<double>> m_riemannSolverOp;
    std::shared_ptr<GetFwdBwdTracePhysOp<double>> m_getFwdBwdTracePhysOp;
    std::shared_ptr<ExpressionOp<double>> m_initialOp;
    std::shared_ptr<ExpressionOp<double>> m_expressionOp;

    UnsteadyAdvection(const LibUtilities::SessionReaderSharedPtr &pSession,
                      const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~UnsteadyAdvection() override = default;

    void v_InitObject(bool DeclareFields = true) override;

    void v_DoSolve() override;

    void v_GenerateSummary(SummaryList &s) override;

    void DoAdvection(Field<double, FieldState::Phys> &in,
                     Field<double, FieldState::Phys> &out,
                     [[maybe_unused]] const double &time,
                     [[maybe_unused]] const double &factor);

    void DoProjection(Field<double, FieldState::Phys> &in,
                      Field<double, FieldState::Phys> &out, const double time);

    void InitialiseOperators();

    void InitialiseFields();

    void SetInitialConditionsField(Field<double, FieldState::Phys> &field);
};

} // namespace Nektar

#endif
