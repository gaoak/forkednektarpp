///////////////////////////////////////////////////////////////////////////////
//
// File: VelocityCorrectionScheme.h
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
// Description: Velocity correction scheme for solving the
//              incompressible Navier-Stokes equations.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Operators/ElmtOps/Advection/AdvectionOp.hpp>
#include <Operators/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <Operators/ElmtOps/Divergence/DivergenceOp.hpp>
#include <Operators/ElmtOps/PhysDeriv/PhysDerivOp.hpp>
#include <Operators/Field/Field.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/FwdTrans/FwdTransOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/HelmSolve/HelmSolveOp.hpp>
#include <Operators/GlobalLinSysOps/LinearSystems/PoissonSolve/PoissonSolveOp.hpp>
#include <Operators/Norm/NormL2/NormL2Op.hpp>
#include <Operators/Norm/NormLinf/NormLinfOp.hpp>
#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/EquationSystems/UnsteadySystem.h>
#include <SolverCore/Forcing/Forcing.h>

namespace Nektar
{
using namespace SolverCore;
using namespace Operators;

class VelocityCorrectionScheme : public UnsteadySystem
{
public:
    friend class MemoryManager<VelocityCorrectionScheme>;

    /// Creates an instance of this class
    static EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        EquationSystemSharedPtr equ =
            MemoryManager<VelocityCorrectionScheme>::AllocateSharedPtr(pSession,
                                                                       pGraph);
        equ->InitObject();
        return equ;
    }

    /// Name of class
    static std::string className;

protected:
    // Kinematic viscosity
    double m_kinvis;

    // Diffusion coefficient
    std::vector<double> m_diffCoeff;

    // Time stepping coefficient
    double m_lambda;

    // Save variable strings and number for verbose output and looping
    unsigned int m_pressureIndex;
    std::vector<std::string> m_variablesVel;
    std::vector<std::string> m_variablesFields;
    std::vector<std::string> m_variablesAddScalars;
    std::vector<std::string> m_variablesPressure;
    std::vector<std::string> m_variablesTotal;

    // Setup workspaces
    Field<double, FieldState::Phys> m_pressure;
    Field<double, FieldState::Phys> m_advVel;
    Field<double, FieldState::Phys> m_wsp_phys;
    Field<double, FieldState::Phys> m_wsp_phys_deriv_pressure;
    Field<double, FieldState::Phys> m_wsp_phys_1c;
    Field<double, FieldState::Coeff> m_pressure_coeff;

    // Initialise operators
    std::shared_ptr<BwdTransOp<double>> m_bwdTransPressureOp;
    std::shared_ptr<AdvectionOp<double>> m_advectionOp;
    std::shared_ptr<PhysDerivOp<double>> m_physDerivPressureOp;
    std::shared_ptr<HelmSolveOp<double>> m_helmSolveOp;
    std::shared_ptr<LinearSolverOp<double>> m_linearSolverFieldsOp;
    std::shared_ptr<LinearSolverOp<double>> m_linearSolverPressureOp;
    std::map<double, std::shared_ptr<PreconOp<double>>> m_preconFieldsOpMap;
    std::map<double, std::shared_ptr<PreconOp<double>>> m_preconPressureOpMap;
    std::shared_ptr<PoissonSolveOp<double>> m_poissonSolveOp;
    std::vector<ForcingSharedPtr> m_forcing;
    std::shared_ptr<DivergenceOp<double>> m_divergenceOp;

    VelocityCorrectionScheme(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~VelocityCorrectionScheme() override = default;

    void SolveUnsteadyStokesSystem(Field<double, FieldState::Phys> &in,
                                   Field<double, FieldState::Phys> &out,
                                   [[maybe_unused]] const double &time,
                                   const double &lambda);

    void EvaluateAdvection_SetPressureBCs(Field<double, FieldState::Phys> &in,
                                          Field<double, FieldState::Phys> &out,
                                          const double &time, const double &dt);

    void v_InitObject(bool declareExpansionLists = true) override;
    void v_PrintNorms(std::ostream &out) override;

    void v_GenerateSummary(SummaryList &s) override;

    std::vector<bool> v_GetSystemSingularChecks() override;

    void v_SetInitialConditions([[maybe_unused]] double initialTime) override;

    void v_InitialiseFields() override;

    void v_InitialiseOperators() override;

    void v_WriteFld(const std::string &outname) override;

    void SetDiffusionCoeff();

    void InitialiseParameters();
};

} // namespace Nektar
