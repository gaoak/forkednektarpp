///////////////////////////////////////////////////////////////////////////////
//
// File: FilterUtils.h
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

#pragma once

#include <fstream>
#include <set>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverCore/Filters/FilterContext.h>

namespace Nektar::SolverCore
{

class SOLVER_CORE_EXPORT FilterParameters
{
public:
    using ParamMap = std::map<std::string, std::string>;

    FilterParameters(const LibUtilities::SessionReaderSharedPtr &session,
                     const ParamMap &parameters, std::string filterName);

    bool Has(const std::string &name) const;
    std::string RequiredString(const std::string &name);
    std::string OptionalString(const std::string &name,
                               std::string defaultValue = {});
    double Finite(const std::string &name, double defaultValue);
    unsigned int PositiveFrequency(const std::string &name,
                                   unsigned int defaultValue);
    bool StrictBoolean(const std::string &name, bool defaultValue);
    double StartTime(double defaultValue = 0.0);
    void Consume(const std::string &name);
    void ValidateAllConsumed() const;

private:
    LibUtilities::SessionReaderSharedPtr m_session;
    const ParamMap &m_parameters;
    std::string m_filterName;
    std::set<std::string> m_consumed;

    double Evaluate(const std::string &name, const std::string &value) const;
};

class SOLVER_CORE_EXPORT StepSchedule
{
public:
    StepSchedule(unsigned int frequency = 1, bool includeInitial = true,
                 double startTime = 0.0);
    bool Due(const FilterContext &context) const;

private:
    unsigned int m_frequency;
    bool m_includeInitial;
    double m_startTime;
};

SOLVER_CORE_EXPORT std::string SetupFilterOutput(
    const LibUtilities::SessionReaderSharedPtr &session,
    const std::string &extension, const FilterParameters::ParamMap &parameters);
SOLVER_CORE_EXPORT std::string SetupFilterOutput(
    const LibUtilities::SessionReaderSharedPtr &session,
    const std::string &extension, const std::string &name);

class SOLVER_CORE_EXPORT OutputSeries
{
public:
    explicit OutputSeries(std::string basePath, unsigned int index = 0);
    std::string Next();
    std::string At(unsigned int index) const;
    unsigned int GetIndex() const;
    void SetIndex(unsigned int index);

private:
    std::string m_basePath;
    unsigned int m_index;
};

class SOLVER_CORE_EXPORT RootTableWriter
{
public:
    explicit RootTableWriter(LibUtilities::CommSharedPtr communicator);
    ~RootTableWriter();
    void Open(const std::string &path);
    void Close();
    bool IsWriter() const;
    std::ostream &Stream();
    void EndRow();
    void Flush();
    void Metadata(const std::string &text);
    void Columns(const std::vector<std::string> &columns);

private:
    bool m_writer = false;
    std::ofstream m_stream;
};

} // namespace Nektar::SolverCore
