///////////////////////////////////////////////////////////////////////////////
//
// File: FilterUtils.cpp
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
// Description: Shared parsing, scheduling and output utilities for filters.
//
///////////////////////////////////////////////////////////////////////////////

#include <SolverCore/Filters/FilterUtils.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <system_error>

#include <LibUtilities/BasicUtils/Equation.h>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/StringUtils.hpp>

namespace Nektar::SolverCore
{

FilterParameters::FilterParameters(
    const LibUtilities::SessionReaderSharedPtr &session,
    const ParamMap &parameters, std::string filterName)
    : m_session(session), m_parameters(parameters),
      m_filterName(std::move(filterName))
{
}

bool FilterParameters::Has(const std::string &name) const
{
    return m_parameters.count(name) != 0;
}

std::string FilterParameters::RequiredString(const std::string &name)
{
    const auto it = m_parameters.find(name);
    ASSERTL0(it != m_parameters.end() && !it->second.empty(),
             m_filterName + " requires non-empty parameter '" + name + "'.");
    m_consumed.insert(name);
    return it->second;
}

std::string FilterParameters::OptionalString(const std::string &name,
                                             std::string defaultValue)
{
    const auto it = m_parameters.find(name);
    if (it == m_parameters.end())
    {
        return defaultValue;
    }
    ASSERTL0(!it->second.empty(),
             m_filterName + " parameter '" + name + "' must not be empty.");
    m_consumed.insert(name);
    return it->second;
}

double FilterParameters::Evaluate(const std::string &name,
                                  const std::string &value) const
{
    LibUtilities::Equation equation(m_session->GetInterpreter(), value);
    const double result = equation.Evaluate();
    ASSERTL0(std::isfinite(result),
             m_filterName + " parameter '" + name + "' must be finite.");
    return result;
}

double FilterParameters::Finite(const std::string &name,
                                const double defaultValue)
{
    const auto it = m_parameters.find(name);
    if (it == m_parameters.end())
    {
        return defaultValue;
    }
    m_consumed.insert(name);
    return Evaluate(name, it->second);
}

unsigned int FilterParameters::PositiveFrequency(const std::string &name,
                                                 unsigned int defaultValue)
{
    const auto it = m_parameters.find(name);
    if (it == m_parameters.end())
    {
        ASSERTL0(defaultValue > 0,
                 m_filterName + " requires parameter '" + name + "'.");
        return defaultValue;
    }
    m_consumed.insert(name);
    const double value = Evaluate(name, it->second);
    double integer     = 0.0;
    ASSERTL0(value > 0.0 && std::modf(value, &integer) == 0.0 &&
                 value <= std::numeric_limits<unsigned int>::max(),
             m_filterName + " parameter '" + name +
                 "' must be a positive integer.");
    return static_cast<unsigned int>(integer);
}

bool FilterParameters::StrictBoolean(const std::string &name,
                                     const bool defaultValue)
{
    const auto it = m_parameters.find(name);
    if (it == m_parameters.end())
    {
        return defaultValue;
    }
    m_consumed.insert(name);
    if (StringUtils::IEquals(it->second, "true"))
    {
        return true;
    }
    if (StringUtils::IEquals(it->second, "false"))
    {
        return false;
    }
    ASSERTL0(false, m_filterName + " parameter '" + name +
                        "' must be 'true' or 'false'.");
    return false;
}

double FilterParameters::StartTime(const double defaultValue)
{
    const bool hasStart = Has("StartTime");
    const bool hasAlias = Has("OutputStartTime");
    ASSERTL0(!(hasStart && hasAlias),
             m_filterName +
                 " cannot specify both StartTime and OutputStartTime.");
    if (hasStart)
    {
        return Finite("StartTime", defaultValue);
    }
    if (hasAlias)
    {
        if (m_session->GetComm()->TreatAsRankZero())
        {
            std::cerr << "WARNING: " << m_filterName
                      << " parameter OutputStartTime is deprecated; use "
                         "StartTime."
                      << std::endl;
        }
        return Finite("OutputStartTime", defaultValue);
    }
    return defaultValue;
}

void FilterParameters::Consume(const std::string &name)
{
    if (Has(name))
    {
        m_consumed.insert(name);
    }
}

void FilterParameters::ValidateAllConsumed() const
{
    for (const auto &[name, value] : m_parameters)
    {
        static_cast<void>(value);
        ASSERTL0(m_consumed.count(name) != 0,
                 m_filterName + " does not recognise parameter '" + name +
                     "'.");
    }
}

StepSchedule::StepSchedule(const unsigned int frequency,
                           const bool includeInitial, const double startTime)
    : m_frequency(frequency), m_includeInitial(includeInitial),
      m_startTime(startTime)
{
    ASSERTL0(m_frequency > 0, "A filter schedule frequency must be positive.");
    ASSERTL0(std::isfinite(m_startTime),
             "A filter schedule start time must be finite.");
}

bool StepSchedule::Due(const FilterContext &context) const
{
    if (context.initial && !m_includeInitial)
    {
        return false;
    }
    return context.step % m_frequency == 0 &&
           context.time >= m_startTime - NekConstants::kNekMachineEpsilon;
}

std::string SetupFilterOutput(
    const LibUtilities::SessionReaderSharedPtr &session,
    const std::string &extension, const FilterParameters::ParamMap &parameters)
{
    std::string name = session->GetSessionName();
    const auto it    = parameters.find("OutputFile");
    if (it != parameters.end())
    {
        ASSERTL0(!it->second.empty(), "OutputFile must not be empty.");
        name = it->second;
    }
    return SetupFilterOutput(session, extension, name);
}

std::string SetupFilterOutput(
    const LibUtilities::SessionReaderSharedPtr &session,
    const std::string &extension, const std::string &name)
{
    fs::path path = fs::path(name).replace_extension(extension);
    if (session->GetBackups() && session->GetComm()->TreatAsRankZero() &&
        fs::exists(path))
    {
        unsigned int count = 0;
        fs::path backup;
        do
        {
            backup = path.parent_path() /
                     (path.stem().string() + ".bak" + std::to_string(count++) +
                      path.extension().string());
        } while (fs::exists(backup));

        std::cout << "renaming " << path << " -> " << backup << std::endl;
        try
        {
            fs::rename(path, backup);
        }
        catch (const fs::filesystem_error &error)
        {
            ASSERTL0(error.code() == std::errc::no_such_file_or_directory,
                     "Filesystem error: " + std::string(error.what()));
        }
    }
    return LibUtilities::PortablePath(path);
}

OutputSeries::OutputSeries(std::string basePath, const unsigned int index)
    : m_basePath(std::move(basePath)), m_index(index)
{
}

std::string OutputSeries::At(const unsigned int index) const
{
    const fs::path path(m_basePath);
    return LibUtilities::PortablePath(path.parent_path() /
                                      (path.stem().string() + "_" +
                                       std::to_string(index) +
                                       path.extension().string()));
}

std::string OutputSeries::Next()
{
    return At(m_index++);
}

unsigned int OutputSeries::GetIndex() const
{
    return m_index;
}

void OutputSeries::SetIndex(const unsigned int index)
{
    m_index = index;
}

RootTableWriter::RootTableWriter(LibUtilities::CommSharedPtr communicator)
    : m_writer(communicator->TreatAsRankZero())
{
}

RootTableWriter::~RootTableWriter()
{
    Close();
}

void RootTableWriter::Open(const std::string &path)
{
    if (!m_writer)
    {
        return;
    }
    Close();
    m_stream.open(path);
    ASSERTL0(m_stream.good(), "Unable to open filter output file: " + path);
    m_stream << std::scientific << std::setprecision(16);
}

void RootTableWriter::Close()
{
    if (m_stream.is_open())
    {
        m_stream.close();
    }
}

bool RootTableWriter::IsWriter() const
{
    return m_writer;
}

std::ostream &RootTableWriter::Stream()
{
    ASSERTL1(m_writer && m_stream.is_open(),
             "Filter table output is not open on this rank.");
    return m_stream;
}

/**
 * @brief End the current row of the table and flush it to the file.
 *
 * A line of a table is ended here rather than with a bare newline, so that
 * each row reaches the file as soon as it is complete and a run that
 * terminates abnormally leaves every completed row on disk. A filter that
 * writes many rows at once ends them with a newline and calls Flush() after
 * the last. Like Stream(), it is called on the writing rank only.
 */
void RootTableWriter::EndRow()
{
    Stream() << '\n' << std::flush;
}

/**
 * @brief Flush the rows written since the last flush to the file.
 */
void RootTableWriter::Flush()
{
    Stream() << std::flush;
}

void RootTableWriter::Metadata(const std::string &text)
{
    if (m_writer)
    {
        Stream() << "# " << text;
        EndRow();
    }
}

void RootTableWriter::Columns(const std::vector<std::string> &columns)
{
    if (!m_writer)
    {
        return;
    }
    for (size_t i = 0; i < columns.size(); ++i)
    {
        Stream() << (i == 0 ? "" : " ") << columns[i];
    }
    EndRow();
}

} // namespace Nektar::SolverCore
