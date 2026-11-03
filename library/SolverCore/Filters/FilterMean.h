///////////////////////////////////////////////////////////////////////////////
//
// File: FilterMean.h
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
// Description: Outputs the domain average of every variable during
// time-stepping.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <SolverCore/Filters/Filter.h>
#include <SolverCore/Filters/FilterUtils.h>
#include <SolverCore/Mean/MeanOp.hpp>

namespace Nektar::SolverCore
{

class FilterMean : public Filter
{
public:
    friend class MemoryManager<FilterMean>;

    static FilterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params)
    {
        return MemoryManager<FilterMean>::AllocateSharedPtr(session, params);
    }

    static std::string className;

protected:
    SOLVER_CORE_EXPORT FilterMean(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params);
    SOLVER_CORE_EXPORT ~FilterMean() override = default;

    SOLVER_CORE_EXPORT void v_Initialise(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Apply(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Finalise(FilterContext &context) override;

private:
    std::vector<std::shared_ptr<MeanOp<double>>> m_meanOps;
    std::vector<size_t> m_componentCounts;
    std::vector<std::string> m_variables;
    std::vector<double> m_means;
    double m_volume = 0.0;
    std::string m_outputFile;
    unsigned int m_outputFrequency = 1;
    StepSchedule m_schedule;
    std::unique_ptr<RootTableWriter> m_writer;

    void ComputeMeans(FilterContext &context);
    void WriteHeader(unsigned int spaceDimension);
    void WriteMeans(const double time);
};

} // namespace Nektar::SolverCore
