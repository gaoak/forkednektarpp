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

#include <IncNavierStokesSolverRedesign/BoundaryConditions/HOPressureBndCond/HOPressureBndCondOp.hpp>
#include <IncNavierStokesSolverRedesign/BoundaryConditions/HOutflowBndCond/HOutflowBndCondOp.hpp>
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
namespace MultiRegions
{
template <typename TData> class AdvectionOp;
template <typename TData> class AdvectionDealiasOp;
template <typename TData> class BwdTransOp;
template <typename TData> class DivergenceOp;
template <FieldState TFieldOut, typename TData> class IProductWRTDerivBaseOp;
template <typename TData> class PhysDerivOp;
} // namespace MultiRegions

namespace SolverCore
{
template <typename TData> class MeanRemovalOp;
template <typename TData> class PreconOp;
template <typename TData> class LinearSystemOp;
template <typename TData> class LinearSolverOp;
template <typename TData> class PoissonSolveOp;
} // namespace SolverCore

using namespace SolverCore;
using namespace MultiRegions;

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

    // true = evaluate explicit advection with 3/2-rule spectral/hp
    // dealiasing (local over-integration), false = native-order evaluation
    bool m_specHPDealiasing = false;

    // true = weak pressure form, forced by the inner product of the
    // pressure RHS vector with the derivative of the basis; false = strong
    // pressure form, forced by the divergence of that vector
    bool m_weakPressure = false;

    // true = this rank has a region with the higher-order pressure or the
    // high-order outflow condition, whose velocity history it keeps; the
    // linear-implicit formulation keeps it on every rank
    bool m_hasHighOrderPressureBCs = false;

    // true = this rank has a higher-order pressure region with a velocity
    // component that is not Dirichlet, whose explicit forcing history it
    // keeps
    bool m_hasPressureBCExplicitRhs = false;

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
    // Allocated only with m_implicitAdvection: the advection velocity of the
    // velocity solve.
    LibUtilities::Field<double, FieldState::Phys> m_advVel;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_phys;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_fields_rhs;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_explicit_adv_rhs;
    // Allocated only with m_hasHighOrderPressureBCs.
    LibUtilities::Field<double, FieldState::Phys> m_pressureBCVelocity;
    // Allocated only with m_hasPressureBCExplicitRhs.
    LibUtilities::Field<double, FieldState::Phys> m_pressureBCExplicitRhs;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_phys_deriv_pressure;
    LibUtilities::Field<double, FieldState::Phys> m_wsp_phys_1c;
    LibUtilities::Field<double, FieldState::Coeff> m_pressure_coeff;
    // Allocated only with m_weakPressure.
    LibUtilities::Field<double, FieldState::Coeff> m_wsp_coeff_1c;
    std::map<unsigned int,
             std::deque<LibUtilities::Field<double, FieldState::Phys>>>
        m_advectionRhsHistories;
    std::map<unsigned int,
             std::deque<LibUtilities::Field<double, FieldState::Phys>>>
        m_velocityHistories;
    std::map<unsigned int,
             std::deque<LibUtilities::Field<double, FieldState::Phys>>>
        m_pressureBCRhsHistories;

    // Initialise operators
    std::shared_ptr<BwdTransOp<double>> m_bwdTransPressureOp;
    std::shared_ptr<AdvectionOp<double>> m_advectionOp;
    std::shared_ptr<AdvectionDealiasOp<double>> m_advectionDealiasOp;
    std::shared_ptr<PhysDerivOp<double>> m_physDerivPressureOp;
    std::shared_ptr<LinearSystemOp<double>> m_fieldsSolveOp;
    std::shared_ptr<LinearSolverOp<double>> m_linearSolverFieldsOp;
    std::shared_ptr<LinearSolverOp<double>> m_linearSolverPressureOp;
    // Semi-implicit only: a preconditioner of the velocity solve per step
    // size.
    std::map<double, std::shared_ptr<PreconOp<double>>> m_preconFieldsOpMap;
    // Linear-implicit only: the preconditioner of the velocity solve,
    // configured again every step.
    std::shared_ptr<PreconOp<double>> m_preconFieldsOp;
    std::map<double, std::shared_ptr<PreconOp<double>>> m_preconPressureOpMap;
    std::shared_ptr<PoissonSolveOp<double>> m_poissonSolveOp;
    // Built only when a region of any rank has the condition.
    std::shared_ptr<HOPressureBndCondOp<double>> m_hoPressureBCOp;
    std::shared_ptr<HOutflowBndCondOp<double>> m_hOutflowBCOp;
    std::vector<ForcingSharedPtr> m_forcing;
    // Strong pressure form only: the forcing div(f).
    std::shared_ptr<DivergenceOp<double>> m_divergenceOp;
    // Weak pressure form only: the forcing (grad(phi), f).
    std::shared_ptr<IProductWRTDerivBaseOp<FieldState::Coeff, double>>
        m_iprodDerivPressureOp;

    // Lazily constructed on first use in v_PrintNorms. Execution-space
    // dispatch (Serial/AVX/Device) is resolved once by
    // MeanRemovalOp::Create(), so this is used identically regardless of
    // opExecSpace.
    std::shared_ptr<MeanRemovalOp<double>> m_meanRemovalOp;

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

    void SetUpWeakPressureForcing(
        LibUtilities::Field<double, FieldState::Phys> &in,
        LibUtilities::Field<double, FieldState::Phys> &fieldRhs,
        const double dt_inv_gamma);

    void EvaluateAdvectionContribution(
        LibUtilities::Field<double, FieldState::Phys> &in,
        LibUtilities::Field<double, FieldState::Phys> &out, const double &time,
        const double &dt);
    std::deque<LibUtilities::Field<double, FieldState::Phys>> &
    GetAdvectionRhsHistory(unsigned int historyId, unsigned int historySize);
    void FreeStartupHistories(unsigned int historyId);
    std::deque<LibUtilities::Field<double, FieldState::Phys>> &GetVelocityHistory(
        unsigned int historyId, unsigned int historySize);
    std::deque<LibUtilities::Field<double, FieldState::Phys>> &
    GetPressureBCRhsHistory(unsigned int historyId, unsigned int historySize);

    void UpdateAdvectionRhsHistory(
        LibUtilities::Field<double, FieldState::Phys> &advRhs);
    void UpdateVelocityHistory(
        LibUtilities::Field<double, FieldState::Phys> &velocity);
    void UpdatePressureBCRhsHistory(
        LibUtilities::Field<double, FieldState::Phys> &rhs, const double dt);

    void v_InitObject(bool declareExpansionLists = true) override;

    /// Offer the advection velocity to the Courant estimate.
    LibUtilities::Field<double, FieldState::Phys> &v_GetCFLVelocityField()
        override;

    void v_PrintNorms(std::ostream &out) override;

    void v_GenerateSummary(SummaryList &s) override;

    std::vector<bool> v_GetSystemSingularChecks() override;

    void v_SetInitialConditions([[maybe_unused]] double initialTime) override;

    void v_InitialiseFields() override;

    void v_InitialiseOperators() override;

    void v_WriteFld(const std::string &outname) override;

    void SetDiffusionCoeff();

    void InitialiseParameters();

    bool HasUserDefinedType(const std::string &userDefinedType,
                            const std::vector<std::string> &variables) const;
};

} // namespace Nektar
