///////////////////////////////////////////////////////////////////////////////
//
// File: FilterHistoryPoints.cpp
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
// Description: Samples physical fields at a set of points during time-
// stepping.
//
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <SolverCore/Filters/FilterHistoryPoints.h>
#include <SolverCore/Filters/PointSet.h>

namespace Nektar::SolverCore
{

std::string FilterHistoryPoints::cmdSetStartFilterFileNum =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "set-filter-historypoints-start-number", "",
        "Set the starting number of the history points filter file number.");

std::string FilterHistoryPoints::className =
    GetFilterFactory().RegisterCreatorFunction("HistoryPoints",
                                               FilterHistoryPoints::create);

FilterHistoryPoints::FilterHistoryPoints(
    const LibUtilities::SessionReaderSharedPtr &session, const ParamMap &params)
    : Filter(session)
{
    m_outputFile = Filter::SetupOutput(".his", params);
    FilterParameters parameters(session, params, "FilterHistoryPoints");
    parameters.Consume("OutputFile");
    m_outputFrequency = parameters.PositiveFrequency("OutputFrequency", 1);
    m_outputOneFile   = parameters.StrictBoolean("OutputOneFile", true);
    m_schedule = StepSchedule(m_outputFrequency, true, parameters.StartTime());
    m_points   = PointSet::Parse(parameters).GetPoints();
    parameters.ValidateAllConsumed();
    m_series = std::make_unique<OutputSeries>(m_outputFile);
}

void FilterHistoryPoints::v_Initialise(FilterContext &context)
{
    ASSERTL0(!m_session->DefinesSolverInfo("HOMOGENEOUS"),
             "Redesigned HistoryPoints does not yet support homogeneous "
             "expansions.");

    const auto &fields = context.fields.GetPhysicalGroups();
    ASSERTL0(!fields.empty(), "FilterHistoryPoints found no physical fields.");
    m_historyOps.clear();
    m_componentCounts.clear();
    m_variables.clear();
    for (const auto &view : fields)
    {
        ASSERTL0(view.field.get().GetNumHomoModes() == 1,
                 "Redesigned HistoryPoints does not yet support homogeneous "
                 "fields.");
        ASSERTL0(view.field.get().GetNumComponents() == view.components.size(),
                 "HistoryPoints physical field component mismatch.");
        m_historyOps.push_back(HistoryPointsOp<double>::Create(
            view.expansion, ComponentNames(view.components)));
        m_historyOps.back()->Configure(m_points);
        m_componentCounts.push_back(view.components.size());
        const auto names = ComponentNames(view.components);
        m_variables.insert(m_variables.end(), names.begin(), names.end());
    }

    if (m_session->DefinesCmdLineArgument(
            "set-filter-historypoints-start-number"))
    {
        m_series->SetIndex(
            std::stoul(m_session->GetCmdLineArgument<std::string>(
                "set-filter-historypoints-start-number")));
    }

    m_writer = std::make_unique<RootTableWriter>(m_session->GetComm());
    if (m_outputOneFile)
    {
        m_writer->Open(m_outputFile);
        WriteHeader();
    }
    v_Apply(context);
}

void FilterHistoryPoints::OpenOutputFile()
{
    const std::string output = m_series->Next();
    m_writer->Open(SetupOutput(fs::path(output).extension().string(), output));
    WriteHeader();
}

void FilterHistoryPoints::WriteHeader()
{
    if (!m_writer->IsWriter())
    {
        return;
    }
    auto &stream = m_writer->Stream();
    stream << "# History data for variables:";
    for (const auto &variable : m_variables)
    {
        stream << " " << variable;
    }
    m_writer->EndRow();
    for (size_t point = 0; point < m_points.size(); ++point)
    {
        stream << "# Point " << point << " " << m_points[point][0] << " "
               << m_points[point][1] << " " << m_points[point][2];
        m_writer->EndRow();
    }
    // Deliberately un-commented for:
    // pd.read_csv(filename, sep=r"\s+", comment="#")
    stream << "Time Point";
    for (const auto &variable : m_variables)
    {
        stream << " " << variable;
    }
    m_writer->EndRow();
}

void FilterHistoryPoints::v_Apply(FilterContext &context)
{
    if (!m_schedule.Due(context))
    {
        return;
    }

    const auto &fields = context.fields.GetPhysicalGroups();
    ASSERTL0(fields.size() == m_historyOps.size(),
             "HistoryPoints physical field count changed.");

    // Each operator returns its group's values point-major,
    // values[point * nComp + comp], already reduced to rank zero. Interleave
    // the groups into one point-major row per point across all variables.
    m_values.assign(m_points.size() * m_variables.size(), 0.0);
    size_t componentOffset = 0;
    for (size_t field = 0; field < fields.size(); ++field)
    {
        ASSERTL0(fields[field].components.size() == m_componentCounts[field],
                 "HistoryPoints physical field components changed.");
        // Collective: every rank must call this so each operator's internal
        // reduction completes, even though only rank zero keeps a result.
        (*m_historyOps[field])(fields[field].field.get());
        if (m_writer->IsWriter())
        {
            const auto values = m_historyOps[field]->GetValues();
            ASSERTL1(values.size() ==
                         m_points.size() * m_componentCounts[field],
                     "HistoryPoints result size mismatch.");
            for (size_t point = 0; point < m_points.size(); ++point)
            {
                std::copy_n(values.begin() + point * m_componentCounts[field],
                            m_componentCounts[field],
                            m_values.begin() + point * m_variables.size() +
                                componentOffset);
            }
        }
        componentOffset += m_componentCounts[field];
    }
    WriteData(context.time);
}

void FilterHistoryPoints::WriteData(const double time)
{
    if (!m_writer->IsWriter())
    {
        return;
    }

    if (!m_outputOneFile)
    {
        OpenOutputFile();
    }

    // One row per point, flushed once for the whole step.
    auto &stream = m_writer->Stream();
    for (size_t point = 0; point < m_points.size(); ++point)
    {
        stream << time << " " << point;
        for (size_t component = 0; component < m_variables.size(); ++component)
        {
            stream << " " << m_values[point * m_variables.size() + component];
        }
        stream << '\n';
    }
    m_writer->Flush();

    if (!m_outputOneFile)
    {
        m_writer->Close();
    }
}

void FilterHistoryPoints::v_Finalise([[maybe_unused]] FilterContext &context)
{
    if (m_writer)
    {
        m_writer->Close();
    }
}

} // namespace Nektar::SolverCore
