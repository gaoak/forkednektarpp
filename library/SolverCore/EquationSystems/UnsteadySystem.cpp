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

#include <iomanip>
#include <iostream>

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/EquationSystems/UnsteadySystem.h>

#include "Operators/ElmtOps/BwdTrans/BwdTransOp.hpp"
#include "Operators/GlobalLinSysOps/LinearSystems/FwdTrans/FwdTransOp.hpp"

namespace Nektar::SolverCore
{
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
    m_session->LoadParameter("Time", m_time, 0.0);
    m_session->LoadParameter("TimeStep", m_timestep, 0.0);
    m_session->LoadParameter("NumSteps", m_steps, 0);
    m_session->LoadParameter("FinTime", m_fintime, 0.0);

    ASSERTL0(m_timestep > 0, "m_timestep < 0");

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
    }
    // Set initial conditions from file
    else if (vType == LibUtilities::eFunctionTypeFile)
    {
        // Field files store modal coefficients. Load coefficient space first
        // and reconstruct physical values with the device support BwdTrans
        // operator.
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
    // Time-integration loop
    while (m_timeOp->GetStep() < m_steps ||
           m_timeOp->GetTime() < m_fintime - NekConstants::kNekZeroTol)
    {
        // Perform any solver-specific pre-integration steps.
        if (v_PreIntegrate())
        {
            break;
        }

        // Do time integration
        m_timeOp->Apply(m_fields);

        // Update EquationSystem::m_time
        m_time = m_timeOp->GetTime();

        // Update time in field info if required
        m_fieldMetaDataMap["Time"] = std::to_string(m_time);

        // Write out status information.
        v_PrintStatusInformation();

        // Perform any solver-specific post-integration steps.
        const bool stopIntegration = v_PostIntegrate();

        // Evaluate all filters
        for (auto &filter : m_filters)
        {
            filter.second->Apply(m_time);
        }

        if (stopIntegration)
        {
            break;
        }
    }

    // Keep time consistent with TimeOp state for
    // exact-solution evaluation and output metadata.
    m_time = m_timeOp->GetTime();

    // Print out summary statistics.
    v_PrintSummaryStatistics();

    // Finalise all filters
    // TODO move this into UnsteadySystem~ destructor?
    for (auto &filter : m_filters)
    {
        filter.second->Finalise(m_time);
    }
}

void UnsteadySystem::v_PrintStatusInformation()
{
    if (m_infosteps && !(m_timeOp->GetStep() % m_infosteps) &&
        m_session->GetComm()->GetSpaceComm()->GetRank() == 0)
    {
        std::cout
            // << std::scientific
            << "Steps: " << std::setw(8) << std::left << m_timeOp->GetStep()
            << " Time: " << std::setw(12) << std::left << m_timeOp->GetTime()
            << std::endl;
    }
}

void UnsteadySystem::v_PrintSummaryStatistics()
{
    if (m_session->GetComm()->GetRank() == 0)
    {
        std::cout << "Time-integration  : "
                  << "NO TIMER IMPLEMENTED" << std::endl;
    }
}

/**
 * @brief Sets the initial conditions.
 */
void UnsteadySystem::v_DoInitialise(bool dumpInitialConditions)
{
    CheckForRestartTime(m_time);
    SetInitialConditions(m_time);
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
