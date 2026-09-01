///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadySystem.h
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
// Description: Generic timestepping for Unsteady solvers
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Operators/CFL/MaxStdVelocity/MaxStdVelocityOp.hpp>
#include <Operators/Norm/NormL2/NormL2Op.hpp>
#include <SolverCore/EquationSystems/EquationSystem.h>
#include <SolverCore/Filters/Filter.h>
#include <SolverCore/TimeOps/TimeOp.hpp>

namespace Nektar::SolverCore
{
/// Base class for unsteady solvers.
class UnsteadySystem : public EquationSystem
{
public:
    /// Destructor
    SOLVER_CORE_EXPORT ~UnsteadySystem() override = default;

    void InitialiseTimeOp()
    {
        ASSERTL0(m_fields.GetNumComponents() == m_variables.size(),
                 "Number of fields in EquationSystem::m_fields "
                 "must be equal to number of fields for TimeOp.")
        v_InitialiseTimeOp();
    }

    void SetInitialConditions(double initialTime = 0.0)
    {
        v_SetInitialConditions(initialTime);
    }

    SOLVER_CORE_EXPORT double GetTimeStep()
    {
        return m_timestep;
    }

    SOLVER_CORE_EXPORT void SetTimeStep(const double timestep)
    {
        m_timestep = timestep;
    }

    static std::string cmdSetStartTime;
    static std::string cmdSetStartChkNum;

protected:
    double m_fintime         = 0.0;
    double m_timestep        = 0.0;
    unsigned int m_steps     = 0;
    unsigned int m_infosteps = 0;
    std::shared_ptr<TimeOp<double>> m_timeOp;

    /// Courant number the timestep is reported against. Zero, the default,
    /// switches the estimate off entirely.
    double m_cflSafetyFactor = 0.0;
    /// Reciprocal of the tightest advective time scale in the mesh - the
    /// largest standard element velocity times \f$(P-1)^2\f$, as returned
    /// by MaxStdVelocityOp. Kept from the last estimate for reporting.
    double m_cflInvTimeScale = 0.0;
    std::shared_ptr<Operators::MaxStdVelocityOp<double>> m_maxStdVelocityOp;

    /// Cadence, in steps, of the in-flight NaN and abort-file tests; zero
    /// disables both. Session parameter CheckAbortSteps.
    unsigned int m_abortSteps = 1;
    /// A file of this name beside the run requests a clean stop; it is
    /// consumed when seen. Session property CheckAbortFile.
    std::string m_abortFile = "abort";
    std::shared_ptr<Operators::NormL2Op<double>> m_abortNormOp;

    /// Test the two conditions under which a run should stop early.
    SOLVER_CORE_EXPORT bool CheckAbortConditions();

    /// Timestep the Courant condition permits for the current state.
    SOLVER_CORE_EXPORT double GetCFLTimeStep();

    /// The field the Courant estimate reads.
    SOLVER_CORE_EXPORT virtual LibUtilities::Field<double, FieldState::Phys> &
    v_GetCFLVelocityField();

    /// Weight on the wave-speed component of the field above.
    SOLVER_CORE_EXPORT virtual double v_GetSoundSpeedFactor();
    std::vector<std::pair<std::string, FilterSharedPtr>> m_filters;

    /// Initialises UnsteadySystem class members.
    SOLVER_CORE_EXPORT UnsteadySystem(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph);

    /// Init object for UnsteadySystem class.
    SOLVER_CORE_EXPORT void v_InitObject(
        bool declareExpansionLists = true) override;

    // Initialise Time-integration operator
    SOLVER_CORE_EXPORT virtual void v_InitialiseTimeOp();

    // Parse initial conditions
    SOLVER_CORE_EXPORT virtual void v_SetInitialConditions(
        double initialTime = 0.0);

    /// Solves an unsteady problem.
    SOLVER_CORE_EXPORT void v_DoSolve() override;

    /// Print Status Information
    SOLVER_CORE_EXPORT virtual void v_PrintStatusInformation();

    /// Print Summary Statistics
    SOLVER_CORE_EXPORT virtual void v_PrintSummaryStatistics(double intTime);

    /// Sets up initial conditions.
    SOLVER_CORE_EXPORT void v_DoInitialise(
        bool dumpInitialConditions = true) override;

    /// Print a summary of time stepping parameters.
    SOLVER_CORE_EXPORT void v_GenerateSummary(SummaryList &s) override;

    SOLVER_CORE_EXPORT virtual bool v_PreIntegrate();

    SOLVER_CORE_EXPORT virtual bool v_PostIntegrate();

    SOLVER_CORE_EXPORT void CheckForRestartTime(double &time);

    SOLVER_CORE_EXPORT void WriteInitialConditions();

    SOLVER_CORE_EXPORT LibUtilities::FunctionType GetInitialConditionType(
        std::vector<std::string> &variables);

    SOLVER_CORE_EXPORT void PrintInitialConditions(
        const MultiRegions::ExpListSharedPtr expList,
        const std::vector<std::string> &variables);
};

} // namespace Nektar::SolverCore
