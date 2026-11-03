///////////////////////////////////////////////////////////////////////////////
//
// File: FilterHistoryPoints.h
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

#pragma once

#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <SolverCore/Filters/Filter.h>
#include <SolverCore/Filters/FilterUtils.h>
#include <SolverCore/HistoryPoints/HistoryPointsOp.hpp>

namespace Nektar::SolverCore
{

class FilterHistoryPoints : public Filter
{
public:
    friend class MemoryManager<FilterHistoryPoints>;

    static FilterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params)
    {
        return MemoryManager<FilterHistoryPoints>::AllocateSharedPtr(session,
                                                                     params);
    }

    static std::string cmdSetStartFilterFileNum;
    static std::string className;

protected:
    SOLVER_CORE_EXPORT FilterHistoryPoints(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params);
    SOLVER_CORE_EXPORT ~FilterHistoryPoints() override = default;

    SOLVER_CORE_EXPORT void v_Initialise(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Apply(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Finalise(FilterContext &context) override;

private:
    std::vector<std::array<double, 3>> m_points;
    std::vector<std::string> m_variables;
    std::vector<size_t> m_componentCounts;
    std::vector<double> m_values;
    std::vector<std::shared_ptr<HistoryPointsOp<double>>> m_historyOps;

    std::string m_outputFile;
    unsigned int m_outputFrequency = 1;
    bool m_outputOneFile           = true;
    StepSchedule m_schedule;
    std::unique_ptr<OutputSeries> m_series;
    std::unique_ptr<RootTableWriter> m_writer;

    void OpenOutputFile();
    void WriteHeader();
    void WriteData(const double time);
};

} // namespace Nektar::SolverCore
