///////////////////////////////////////////////////////////////////////////////
//
// File: FilterCheckpoint.cpp
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
// Description: Outputs solution fields during time-stepping.
//
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <sstream>

#include <LibUtilities/BasicUtils/Equation.h>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <SolverCore/Filters/FilterCheckpoint.h>

namespace Nektar::SolverCore
{

std::string FilterCheckpoint::cmdSetStartFilterFileNum =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "set-filter-checkpoint-start-number", "",
        "Set the starting number of the check point filter file number.");

std::string FilterCheckpoint::className =
    GetFilterFactory().RegisterCreatorFunction("Checkpoint",
                                               FilterCheckpoint::create);

FilterCheckpoint::FilterCheckpoint(
    const LibUtilities::SessionReaderSharedPtr &session,
    const EquationSystemSharedPtr &equation, const ParamMap &params)
    : Filter(session, equation)
{
    m_outputFile = Filter::SetupOutput(".chk", params);

    auto it = params.find("OutputFrequency");
    ASSERTL0(it != params.end(), "Missing parameter 'OutputFrequency'.");
    LibUtilities::Equation frequency(m_session->GetInterpreter(), it->second);
    m_outputFrequency =
        static_cast<unsigned int>(std::round(frequency.Evaluate()));
    ASSERTL0(m_outputFrequency > 0,
             "FilterCheckpoint requires OutputFrequency > 0.");

    it = params.find("OutputStartTime");
    if (it != params.end())
    {
        LibUtilities::Equation startTime(m_session->GetInterpreter(),
                                         it->second);
        m_outputStartTime = startTime.Evaluate();
    }
}

void FilterCheckpoint::v_Initialise(const double time)
{
    m_index       = 0;
    m_outputIndex = 0;

    auto eqsys = m_equ.lock();
    ASSERTL0(eqsys, "FilterCheckpoint has no EquationSystem.");

    if (eqsys->HasFieldMetaData("FilterFileNum"))
    {
        m_outputIndex = std::stoul(eqsys->GetFieldMetaData("FilterFileNum"));
    }

    // Note command line overwrites EquationSystem::m_fieldMetaDataMap
    if (m_session->DefinesCmdLineArgument("set-filter-checkpoint-start-number"))
    {
        m_outputIndex = std::stoul(m_session->GetCmdLineArgument<std::string>(
            "set-filter-checkpoint-start-number"));
    }

    // Evaluate filter for initialisation conditions
    if (m_applyOnInitialise)
    {
        v_Apply(time);
    }
}

void FilterCheckpoint::v_Apply(const double time)
{
    if (m_index++ % m_outputFrequency > 0 ||
        (time - m_outputStartTime) < -NekConstants::kNekMachineEpsilon)
    {
        return;
    }

    auto equation = m_equ.lock();
    ASSERTL0(equation, "FilterCheckpoint has no EquationSystem.");

    const std::string ext = fs::path(m_outputFile).extension().string();

    std::stringstream filename;
    filename << fs::path(m_outputFile).replace_extension("").string() << "_"
             << m_outputIndex << ".chk";

    equation->SetFieldMetaData("FilterFileNum", std::to_string(m_outputIndex));
    equation->WriteFld(SetupOutput(ext, filename.str()));
    ++m_outputIndex;
}

void FilterCheckpoint::v_Finalise([[maybe_unused]] const double time)
{
}

bool FilterCheckpoint::v_IsTimeDependent()
{
    return true;
}

} // namespace Nektar::SolverCore
