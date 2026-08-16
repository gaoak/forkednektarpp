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

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <SolverCore/EquationSystems/UnsteadySystem.h>
#include <SolverCore/Forcing/Forcing.h>

#include <deque>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Nektar
{
namespace Operators
{
template <typename TData> class AdvectionOp;
template <typename TData> class BwdTransOp;
template <typename TData> class DivergenceOp;
template <typename TData> class PhysDerivOp;
template <typename TData> class LinearSystemOp;
template <typename TData> class LinearSolverOp;
template <typename TData> class PoissonSolveOp;
template <typename TData> class PreconOp;
} // namespace Operators

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
    // true = linear-implicit, false = semi-implicit scheme
    bool m_implicitAdvection = false;

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
    LibUtilities::Field<double, FieldState::Phys> m_pressure;
    LibUtilities::Field<double, FieldState::Phys> m_advVel;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_phys;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_fields_rhs;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_explicit_adv_rhs;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_phys_deriv_pressure;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_phys_1c;
    LibUtilities::Field<double, FieldState::Coeff> m_pressure_coeff;
    std::map<unsigned int,
             std::deque<LibUtilities::Field<double, FieldState::Phys>>>
        m_advectionRhsHistories;

    // Initialise operators
    std::shared_ptr<BwdTransOp<double>> m_bwdTransPressureOp;
    std::shared_ptr<AdvectionOp<double>> m_advectionOp;
    std::shared_ptr<PhysDerivOp<double>> m_physDerivPressureOp;
    std::shared_ptr<LinearSystemOp<double>> m_fieldsSolveOp;
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

    void SolveUnsteadyStokesSystem(
        LibUtilities::Field<double, FieldState::Phys> &in,
        LibUtilities::Field<double, FieldState::Phys> &out,
        [[maybe_unused]] const double &time, const double &lambda);

    void EvaluateAdvection_SetPressureBCs(
        LibUtilities::Field<double, FieldState::Phys> &in,
        LibUtilities::Field<double, FieldState::Phys> &out, const double &time,
        const double &dt);

    void EvaluateAdvectionContribution(
        LibUtilities::Field<double, FieldState::Phys> &in,
        LibUtilities::Field<double, FieldState::Phys> &out, const double &time,
        const double &dt);
    std::deque<LibUtilities::Field<double, FieldState::Phys>> &
    GetAdvectionRhsHistory(unsigned int historyId);

    void UpdateAdvectionRhsHistory(
        LibUtilities::Field<double, FieldState::Phys> &advRhs);

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
