///////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadySystem.cpp
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
// Description: Generic timestepping for time-dependent equation systems
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/algorithm/string/predicate.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/EquationSystems/UnsteadySystem.h>

namespace Nektar::SolverCore
{
// Spencer, Numerical Methods for Fluid Dynamics, p317.
constexpr double cLambda = 0.2;

std::string UnsteadySystem::cmdSetStartTime =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "set-start-time", "", "Set the starting time of the simulation.");

/**
 * @class UnsteadySystem
 *
 * Provides the underlying timestepping framework for unsteady solvers
 * including the general timestepping routines. This class is not
 * intended to be directly instantiated, but rather is a base class
 * on which to define unsteady solvers.
 *
 * For details on implementing unsteady solvers see
 * \ref sectionADRSolverModuleImplementation here
 */

/**
 * Processes SolverInfo parameters from the session file and sets up
 * timestepping-specific code.
 * @param   pSession        Session object to read parameters from.
 */
UnsteadySystem::UnsteadySystem(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph)
{
}

/**
 * Initialization object for UnsteadySystem class.
 */
void UnsteadySystem::v_InitObject(bool declareExpansionLists)
{
    EquationSystem::v_InitObject(declareExpansionLists);

    // Ensure time integration is defined
    ASSERTL0(
        m_session->DefinesSolverInfo("TimeIntegrationMethod") ||
            m_session->DefinesTimeIntScheme(),
        "TimeIntegration not defined. Please check User-Guide section 3.4.2")

    // Load time-stepping parameters.
    m_session->LoadParameter("IO_InfoSteps", m_infosteps, 0);
    // Safety factor on the permitted timestep; see GetCFLTimeStep().
    m_session->LoadParameter("CFL", m_cflSafetyFactor, 1.0);
    // Cadence of the Courant number report; see GetCFLNumber().
    m_session->LoadParameter("IO_CFLSteps", m_cflSteps, 0);
    m_session->LoadParameter("Time", m_time, 0.0);
    m_session->LoadParameter("TimeStep", m_timestep, 0.0);
    m_session->LoadParameter("NumSteps", m_steps, 0);
    m_session->LoadParameter("FinTime", m_fintime, 0.0);

    // Checkpoints are written only by the Checkpoint filter.
    WARNINGL0(!m_session->DefinesParameter("IO_CheckSteps") &&
                  !m_session->DefinesParameter("IO_CheckTime"),
              "IO_CheckSteps and IO_CheckTime are ignored and no checkpoints "
              "are written for them. Define a Checkpoint filter instead.");

    // Cadence of the in-flight NaN and abort-file tests.
    m_session->LoadParameter("CheckAbortSteps", m_abortSteps, 1);
    if (m_session->DefinesSolverInfo("CheckAbortFile"))
    {
        m_abortFile = m_session->GetSolverInfo("CheckAbortFile");
    }

    ASSERTL0(m_timestep > 0, "m_timestep < 0");

    // Resolve the start time now, from a restart file or --set-start-time,
    // so that the summary and everything set up before DoInitialise() see
    // the time the run actually starts from.
    CheckForRestartTime(m_time);

    // Set up time to dump in output field metadata.
    m_fieldMetaDataMap["Time"] = std::to_string(m_time);

    // Setup all filters defined in session file
    for (const auto &filter : m_session->GetFilters())
    {
        m_filters.emplace_back(
            filter.name,
            GetFilterFactory().CreateInstance(
                filter.name, m_session, shared_from_this(), filter.params));
    }
}

/*
 * Set initial conditions in m_fields and m_fields_coeff using m_variables.
 */
void UnsteadySystem::v_SetInitialConditions(double initialTime)
{
    m_time = initialTime;

    // Print to log/console
    // Note this is useful for a user to see what Nektar is doing
    if (m_session->GetComm()->GetRank() == 0)
    {
        std::cout << "Initial Conditions:" << std::endl;
    }

    // Default: Zero solution in Phys and Coeff space
    m_math.zero(m_fields);
    m_math.zero(m_fields_coeff);

    // Set initial conditions
    auto vType = GetInitialConditionType(m_variables);
    SessionFunction initialConditions(m_session, m_expansionLists[0],
                                      "InitialConditions");
    if (vType == LibUtilities::eFunctionTypeExpression)
    {
        initialConditions.EvaluateExpression(m_variables, m_fields, m_time);

        // Continuous Galerkin: C0 projection
        // Discontinuous Galerkin: Copy
        DoProjection(m_fields, m_fields, m_time);

        // The expression was sampled at the quadrature points, which carries
        // more information than the expansion can hold - there are more
        // quadrature points than modes - so the sampled field is not in the
        // polynomial space. DoProjection() is a copy for a discontinuous
        // field and so leaves it that way, and every later update adds only
        // in-space increments, so the surplus survives the whole run. It has
        // no representation in the modal coefficients a field file stores,
        // which is why a discontinuous restart cannot reproduce the run it
        // resumed. Projecting once here confines the state to the space, and
        // a restart is then exact to every digit.
        //
        // Projecting unconditionally would change every discontinuous
        // answer, so it is opt-in through `ProjectInitialConditions`.
        // DoProjection() itself is left alone: the time integrator calls it
        // every step, where this would be a costly no-op.
        if (m_projectInitialConditions &&
            m_projectionType == MultiRegions::eDiscontinuous)
        {
            ProjectPhysToCoeffs(m_fields);
            m_bwdTransOp->Apply(m_fields_coeff, m_fields);
        }
        else if (m_projectionType == MultiRegions::eDiscontinuous)
        {
            // Warn here rather than on the restart that suffers for it. This
            // is the run whose state stops being representable: what a later
            // restart loses was already lost when this run's field file was
            // written. By then nothing can tell - a state read back from a
            // file is in the polynomial space whatever produced it.
            NEKERROR(ErrorUtil::ewarning,
                     "This discontinuous state is sampled at the quadrature "
                     "points and so is not confined to the polynomial space. "
                     "The modal coefficients in a field file cannot represent "
                     "it, so a run restarted from one written here will not "
                     "reproduce this one. Set ProjectInitialConditions to "
                     "True to confine the state and make a restart exact.");
        }
    }
    // Set initial conditions from file
    else if (vType == LibUtilities::eFunctionTypeFile)
    {
        // Field files store modal coefficients. Load coefficient space first
        // and reconstruct physical values with the device support BwdTrans
        // operator.
        //
        // Nothing to warn about here: whatever wrote the file, the state
        // coming back is a set of modal coefficients and so is in the
        // polynomial space by construction. Whether it is the state the
        // earlier run actually held was decided when that run was set up,
        // which is where the warning lives.
        initialConditions.EvaluateFld(m_variables, m_fields_coeff, m_time);
        m_bwdTransOp->Apply(m_fields_coeff, m_fields);
    }
    PrintInitialConditions(m_expansionLists[0], m_variables);
}

/**
 * @brief Initialise time-integration operator.
 */
void UnsteadySystem::v_InitialiseTimeOp()
{
    m_timeOp = TimeOp<double>::Create(m_expansionLists[0], m_variables);
    m_timeOp->DefineProjection(&EquationSystem::DoProjection, this);
}

/**
 * @brief Initialises the time integration scheme (as specified in the
 * session file), and perform the time integration.
 */
void UnsteadySystem::v_DoSolve()
{
    Nektar::LibUtilities::Timer timer;
    double intTime = 0.0;
    double elapsed = 0.0;

    // Time-integration loop
    while (m_timeOp->GetStep() < m_steps ||
           m_timeOp->GetTime() < m_fintime - NekConstants::kNekZeroTol)
    {
        timer.Start();
        // Perform any solver-specific pre-integration steps.
        if (v_PreIntegrate())
        {
            break;
        }

        // Do time integration
        m_timeOp->Apply(m_fields);
        timer.Stop();
        elapsed = timer.TimePerTest(1);
        intTime += elapsed;

        // Update EquationSystem::m_time
        m_time = m_timeOp->GetTime();

        // Update time in field info if required
        m_fieldMetaDataMap["Time"] = std::to_string(m_time);

        // Write out status information.
        v_PrintStatusInformation();

        // Courant number report, on its own cadence. Computed on every rank
        // - the reduction is collective - and printed on one.
        if (m_cflSteps && !(m_timeOp->GetStep() % m_cflSteps))
        {
            const double cfl = GetCFLNumber();
            if (cfl > 0.0 &&
                m_session->GetComm()->GetSpaceComm()->GetRank() == 0)
            {
                std::cout << "CFL: " << cfl << std::endl;
            }
        }

        // Perform any solver-specific post-integration steps.
        const bool stopIntegration = v_PostIntegrate();

        // Evaluate all filters
        for (auto &filter : m_filters)
        {
            filter.second->Apply(m_time);
        }

        // Test for the abort conditions (NaN, or an abort file).
        if (stopIntegration ||
            (m_abortSteps && !(m_timeOp->GetStep() % m_abortSteps) &&
             CheckAbortConditions()))
        {
            break;
        }
    }

    // Keep time consistent with TimeOp state for
    // exact-solution evaluation and output metadata.
    m_time = m_timeOp->GetTime();

    // Print out summary statistics.
    v_PrintSummaryStatistics(intTime);

    // Finalise all filters
    // TODO move this into UnsteadySystem~ destructor?
    for (auto &filter : m_filters)
    {
        filter.second->Finalise(m_time);
    }
}

/**
 * @brief Timestep the Courant condition permits for the current state.
 *
 * \f[ \Delta t = \frac{\text{CFL} \; \alpha}{c_\lambda \,
 * \text{invTimeScale}} \f]
 *
 * with invTimeScale from MaxStdVelocityOp, \f$c_\lambda = 0.2\f$ and
 * \f$\alpha\f$ the stability limit of the time integration scheme.
 * Returns zero when the solver does not offer a velocity, or when the flow
 * is at rest and no finite timestep is implied.
 *
 * Nothing acts on this yet: it is reported beside the timestep in use so
 * that a run can be judged before the estimate is trusted to set one.
 */
double UnsteadySystem::GetCFLTimeStep()
{
    const double invTimeScale = ComputeCFLInvTimeScale();
    if (invTimeScale <= 0.0)
    {
        return 0.0;
    }

    // The stability limit of the configured time integration scheme,
    // supplied by the scheme itself.
    const double alpha = m_timeOp->GetTimeStability();

    return m_cflSafetyFactor * alpha / (cLambda * invTimeScale);
}

/**
 * @brief Courant number of the timestep in use, for the current state.
 *
 * \f[ \text{CFL} = \Delta t \, c_\lambda \, \text{invTimeScale} \f]
 *
 * with invTimeScale from MaxStdVelocityOp and \f$c_\lambda = 0.2\f$.
 * Returns zero when the solver offers no velocity or the flow is at rest.
 */
double UnsteadySystem::GetCFLNumber()
{
    const double invTimeScale = ComputeCFLInvTimeScale();
    if (invTimeScale <= 0.0)
    {
        return 0.0;
    }

    return m_timestep * cLambda * invTimeScale;
}

/**
 * @brief Reciprocal of the tightest advective time scale in the mesh.
 *
 * Zero when the solver offers no velocity or the flow is at rest - a
 * statement about the flow rather than a failure, so callers say nothing
 * rather than divide by it.
 */
double UnsteadySystem::ComputeCFLInvTimeScale()
{
    auto &velocity = v_GetCFLVelocityField();
    if (!velocity)
    {
        return 0.0;
    }

    if (!m_maxStdVelocityOp)
    {
        m_maxStdVelocityOp =
            MaxStdVelocityOp<double>::Create(m_expansionLists[0], m_variables);
    }

    m_maxStdVelocityOp->SetSoundSpeedFactor(v_GetSoundSpeedFactor());
    m_cflInvTimeScale = m_maxStdVelocityOp->Apply(velocity);

    return m_cflInvTimeScale;
}

/**
 * @brief The field the Courant estimate reads.
 *
 * A solver holds its velocity in its own way: as a field of its own for
 * scalar advection and the incompressible equations, or reconstructed from
 * conserved variables for a compressible one, where the speed of sound
 * follows it as an extra component. The leading components are the velocity
 * and, when v_GetSoundSpeedFactor() is non-zero, the one after them is a
 * wave speed. A solver with no velocity to offer - this default - returns a
 * field that was never instantiated, and GetCFLTimeStep() skips the
 * estimate.
 */
LibUtilities::Field<double, FieldState::Phys> &UnsteadySystem::
    v_GetCFLVelocityField()
{
    // Never instantiated, which GetCFLTimeStep() reads as "this solver
    // offers no velocity" and skips the estimate.
    static LibUtilities::Field<double, FieldState::Phys> noVelocity;
    return noVelocity;
}

/**
 * @brief Weight on the wave-speed component of the CFL velocity field:
 * zero, this default, when the flow carries no acoustic wave.
 */
double UnsteadySystem::v_GetSoundSpeedFactor()
{
    return 0.0;
}

void UnsteadySystem::v_PrintStatusInformation()
{
    if (m_infosteps && !(m_timeOp->GetStep() % m_infosteps) &&
        m_session->GetComm()->GetSpaceComm()->GetRank() == 0)
    {
        std::cout
            // << std::scientific
            << "Steps: " << std::setw(8) << std::left << m_timeOp->GetStep()
            << " Time: " << std::setw(12) << std::left << m_timeOp->GetTime();

        std::cout << std::endl;
    }
}

void UnsteadySystem::v_PrintSummaryStatistics(double intTime)
{
    if (m_session->GetComm()->GetRank() == 0)
    {
        if (boost::iequals(
                m_session->GetCmdLineArgument<std::string>("opExecSpace"),
                "Device"))
        {
            std::cout << "Time-integration  : " << "NO TIMER IMPLEMENTED"
                      << std::endl;
        }
        else
        {
            std::cout << "Time-integration  : " << intTime << std::endl;
        }
    }
}

/**
 * @brief Sets the initial conditions.
 */
void UnsteadySystem::v_DoInitialise(bool dumpInitialConditions)
{
    SetInitialConditions(m_time);

    // The time operator starts at zero when it is created, so start it here
    // from the initial time resolved in v_InitObject(), which may be a
    // restart time.
    if (m_timeOp)
    {
        m_timeOp->SetTime(m_time);
    }

    if (dumpInitialConditions)
    {
        WriteInitialConditions();
    }

    // Initialise all filters
    for (auto &filter : m_filters)
    {
        filter.second->Initialise(m_time);
    }
}

/**
 * @brief Prints a summary with some information regards the
 * time-stepping.
 */
void UnsteadySystem::v_GenerateSummary(SummaryList &s)
{
    EquationSystem::v_GenerateSummary(s);

    AddSummaryItem(s, "Integration Scheme",
                   m_session->GetTimeIntScheme().method);
    AddSummaryItem(s, "Integration Order",
                   std::to_string(m_session->GetTimeIntScheme().order));

    AddSummaryItem(s, "Time Step", m_timestep);
    AddSummaryItem(s, "No. of Steps", m_steps);
    AddSummaryItem(s, "Initial time", m_time);
    AddSummaryItem(s, "Final time", m_time + m_steps * m_timestep);
}

/**
 *
 */
void UnsteadySystem::CheckForRestartTime(double &time)
{
    if (m_session->DefinesFunction("InitialConditions"))
    {
        for (unsigned int i = 0; i < m_expansionLists.size(); ++i)
        {
            LibUtilities::FunctionType vType;

            vType = m_session->GetFunctionType("InitialConditions",
                                               m_session->GetVariable(i));

            if (vType == LibUtilities::eFunctionTypeFile ||
                vType == LibUtilities::eFunctionTypeTransientFile)
            {
                std::string filename = m_session->GetFunctionFilename(
                    "InitialConditions", m_session->GetVariable(i));

                fs::path pfilename(filename);

                // Redefine path for parallel file which is in directory.
                if (fs::is_directory(pfilename))
                {
                    fs::path metafile("Info.xml");
                    fs::path fullpath = pfilename / metafile;
                    filename          = LibUtilities::PortablePath(fullpath);
                }
                LibUtilities::FieldIOSharedPtr fld =
                    LibUtilities::FieldIO::CreateForFile(m_session, filename);
                fld->ImportFieldMetaData(filename, m_fieldMetaDataMap);

                // Check to see if time defined.
                if (m_fieldMetaDataMap != LibUtilities::NullFieldMetaDataMap)
                {
                    auto iter = m_fieldMetaDataMap.find("Time");
                    if (iter != m_fieldMetaDataMap.end())
                    {
                        time = std::stod(iter->second);
                    }
                }

                break;
            }
        }
    }
    if (m_session->DefinesCmdLineArgument("set-start-time"))
    {
        time = std::stod(
            m_session->GetCmdLineArgument<std::string>("set-start-time")
                .c_str());
    }
    ASSERTL0(time >= 0, "Starting time should be >= 0");
}

/**
 *
 */
bool UnsteadySystem::v_PreIntegrate()
{
    return false;
}

/**
 *
 */
bool UnsteadySystem::v_PostIntegrate()
{
    return false;
}

/**
 * @brief Test the two conditions under which a run should stop early.
 *
 * A non-finite solution norm means the integration cannot recover, so it
 * stops with an error. An abort file placed beside the run asks for a clean
 * stop instead: the file is consumed and true is returned, so the time loop
 * is left through its normal exit and the summary, filters and final output
 * still happen.
 */
bool UnsteadySystem::CheckAbortConditions()
{
    // The L2 norm is a sum of squares, which propagates a NaN or an Inf
    // where a max reduction may drop it, and the reduction inside the
    // operator makes the verdict identical on every rank.
    if (!m_abortNormOp)
    {
        m_abortNormOp =
            NormL2Op<double>::Create(m_expansionLists[0], m_variables);
    }
    m_abortNormOp->Apply(m_fields);
    for (const double &norm : m_abortNormOp->GetNorms())
    {
        ASSERTL0(std::isfinite(norm), "NaN found during time integration.");
    }

    // Rank zero looks for the abort file and consumes it.
    int abortFile = 0;
    if (m_comm->GetRank() == 0 && fs::exists(m_abortFile))
    {
        fs::remove(m_abortFile);
        abortFile = 1;
    }
    m_comm->AllReduce(abortFile, LibUtilities::ReduceMax);
    return abortFile != 0;
}

/*
 * Write initial conditions to file.
 */
void UnsteadySystem::WriteInitialConditions()
{
    m_fieldMetaDataMap["Time"] = std::to_string(m_time);
    WriteFld(m_sessionName + "_initial.fld");
}

LibUtilities::FunctionType UnsteadySystem::GetInitialConditionType(
    std::vector<std::string> &variables)
{
    LibUtilities::FunctionType vType = LibUtilities::eFunctionTypeNone;
    if (m_session->DefinesFunction("InitialConditions"))
    {
        vType = m_session->GetFunctionType("InitialConditions", variables[0]);

        // Check all variables use same FunctionType
        for (unsigned int i = 1; i < variables.size(); ++i)
        {
            LibUtilities::FunctionType vType2 =
                m_session->GetFunctionType("InitialConditions", variables[i]);
            ASSERTL0(vType == vType2,
                     "Currently support either all InitialConditions from file "
                     "<F> or all from expression <E>.");
        }
    }

    return vType;
}

void UnsteadySystem::PrintInitialConditions(
    const MultiRegions::ExpListSharedPtr expList,
    const std::vector<std::string> &variables)
{
    if (m_session->GetComm()->GetRank() == 0)
    {
        SessionFunction initialConditions(m_session, expList,
                                          "InitialConditions");
        for (const auto &varName : variables)
        {
            std::cout << "  - Field " << varName << ": "
                      << initialConditions.Describe(varName) << std::endl;
        }
    }
}
} // namespace Nektar::SolverCore
