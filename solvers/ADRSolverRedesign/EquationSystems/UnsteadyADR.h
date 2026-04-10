///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyADR.h
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
// Description: Unsteady Advection Diffusion problem solve routines for new
// operators
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Math/Math.hpp"
#include <Operators/ElmtOps/Advection/AdvectionOp.hpp>
#include <Operators/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <Operators/ElmtOps/Expression/ExpressionOp.hpp>
#include <Operators/Field/Field.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/FwdTrans/FwdTransOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp>
#include <SolverUtils/EquationSystem.h>

namespace Nektar
{
using namespace SolverUtils;
using namespace Operators;

class UnsteadyADR : public EquationSystem
{
public:
    friend class MemoryManager<UnsteadyADR>;

    /// Creates an instance of this class
    static EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        EquationSystemSharedPtr p =
            MemoryManager<UnsteadyADR>::AllocateSharedPtr(pSession, pGraph);
        p->InitObject();
        return p;
    }

    /// Name of class
    static std::string className;

protected:
    // Diffusion coefficient
    double m_epsilon;
    std::vector<double> m_diffCoeff;

    // Time stepping coefficient
    double m_lambda;
    bool m_explicitAdvection = true;

    // Save variable strings and number for verbose output and looping
    std::vector<std::string> m_variables;
    unsigned int m_nVariables;

    // Setup workspaces
    Field<double, FieldState::Phys> m_in, m_wsp_phys, m_advectionVelocity;
    Field<double, FieldState::Coeff> m_wsp_coeff;

    // Declare math
    Math m_math;

    // Time-integration
    std::shared_ptr<TimeOp<double>> m_timeOp;

    // Initialise operators
    std::shared_ptr<AdvectionOp<double>> m_advectionOp;
    std::shared_ptr<BwdTransOp<double>> m_bwdTransOp;
    std::shared_ptr<HelmSolveOp<double>> m_helmSolveOp;
    std::shared_ptr<LinearADRSolveOp<double>> m_linearADRSolveOp;
    std::shared_ptr<LinearSolverOp<double>> m_linearSolverOp;
    std::map<double, std::shared_ptr<PreconOp<double>>> m_preconOp;
    std::shared_ptr<ExpressionOp<double>> m_forcingOp;
    std::shared_ptr<FwdTransOp<double>> m_fwdTransOp;

    UnsteadyADR(const LibUtilities::SessionReaderSharedPtr &pSession,
                const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~UnsteadyADR() override = default;

    void v_InitObject(bool DeclareFields = true) override;

    void v_DoSolve() override;

    void v_GenerateSummary(SummaryList &s) override;

    void DoDiffusion(Field<double, FieldState::Phys> &inout,
                     Field<double, FieldState::Phys> &out,
                     [[maybe_unused]] const double &time, const double &lambda);

    void DoExplicitRhs(Field<double, FieldState::Phys> &in,
                       Field<double, FieldState::Phys> &out, const double &time,
                       const double &dt);

    void DoProjection(Field<double, FieldState::Phys> &in,
                      Field<double, FieldState::Phys> &out, const double time);

    void InitialiseOperators();

    void InitialiseFields();

    void SetDiffusionCoeff();

    void SetAdvectionVel();

    void SetInitialConditionsField(Field<double, FieldState::Phys> &field);
};

} // namespace Nektar
