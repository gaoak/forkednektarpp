///////////////////////////////////////////////////////////////////////////////
//
// File: SteadyADR.h
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
// Description: Steady Advection Diffusion problem solve routines for new
// operators
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Math/Math.hpp"
#include <Operators/ElmtOps/Expression/ExpressionOp.hpp>
#include <Operators/Field/Field.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/LinearSystemOp.hpp>
#include <SolverUtils/EquationSystem.h>

namespace Nektar
{
using namespace SolverUtils;
using namespace Operators;

class SteadyADR : public EquationSystem
{
public:
    friend class MemoryManager<SteadyADR>;

    /// Creates an instance of this class
    static EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        EquationSystemSharedPtr p =
            MemoryManager<SteadyADR>::AllocateSharedPtr(pSession, pGraph);
        p->InitObject();
        return p;
    }

    /// Name of class
    static std::string className1;
    static std::string className2;
    static std::string className3;

protected:
    // Diffusion coefficient
    double m_epsilon;
    std::vector<double> m_diffCoeff;

    // Save variable strings and number for verbose output and looping
    std::vector<std::string> m_variables;
    unsigned int m_nVariables;

    // Setup workspaces
    Field<double, FieldState::Phys> m_in;
    Field<double, FieldState::Phys> m_advectionVel;
    Field<double, FieldState::Phys> m_wsp_fce;
    Field<double, FieldState::Coeff> m_wsp_coeff;

    // Declare math
    Math m_math;

    // Initialise operators
    std::shared_ptr<LinearSystemOp<double>> m_linearSystemOp;
    std::shared_ptr<LinearSolverOp<double>> m_linearSolverOp;
    std::shared_ptr<PreconOp<double>> m_preconOp;
    std::shared_ptr<ExpressionOp<double>> m_forcingOp;

    SteadyADR(const LibUtilities::SessionReaderSharedPtr &pSession,
              const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~SteadyADR() override = default;

    void v_InitObject(bool DeclareFields = true) override;

    void v_DoSolve() override;

    void v_GenerateSummary(SummaryList &s) override;

    void InitialiseOperators();

    void InitialiseFields();

    void SetDiffusionCoeff();

    void SetAdvectionVel();
};

} // namespace Nektar
