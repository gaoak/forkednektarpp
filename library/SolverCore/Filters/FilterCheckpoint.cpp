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
    const LibUtilities::SessionReaderSharedPtr &session, const ParamMap &params)
    : Filter(session)
{
    m_outputFile = Filter::SetupOutput(".chk", params);
    FilterParameters parameters(session, params, "FilterCheckpoint");
    parameters.Consume("OutputFile");
    m_outputFrequency = parameters.PositiveFrequency("OutputFrequency", 0);
    m_schedule = StepSchedule(m_outputFrequency, true, parameters.StartTime());
    parameters.ValidateAllConsumed();
    m_series = std::make_unique<OutputSeries>(m_outputFile);
}

void FilterCheckpoint::v_Initialise(FilterContext &context)
{
    const auto &metadata = context.output.getMetadata();
    if (metadata.count("FilterFileNum") != 0)
    {
        m_series->SetIndex(std::stoul(metadata.at("FilterFileNum")));
    }

    // Note command line overwrites EquationSystem::m_fieldMetaDataMap
    if (m_session->DefinesCmdLineArgument("set-filter-checkpoint-start-number"))
    {
        m_series->SetIndex(
            std::stoul(m_session->GetCmdLineArgument<std::string>(
                "set-filter-checkpoint-start-number")));
    }
    v_Apply(context);
}

void FilterCheckpoint::v_Apply(FilterContext &context)
{
    if (!m_schedule.Due(context))
    {
        return;
    }
    const auto index = m_series->GetIndex();
    context.output.setMetadata("FilterFileNum", std::to_string(index));
    const auto path = m_series->Next();
    context.output.writeFields(SetupOutput(".chk", path));
}

void FilterCheckpoint::v_Finalise([[maybe_unused]] FilterContext &context)
{
}

} // namespace Nektar::SolverCore
