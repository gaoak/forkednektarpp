///////////////////////////////////////////////////////////////////////////////
//
// File: FilterAverageFields.h
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
// Description: Outputs time-averaged solution fields during time-stepping.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <LibUtilities/BasicUtils/Math/MathHelper.hpp>
#include <SolverCore/Filters/Filter.h>
#include <SolverCore/Filters/FilterUtils.h>

namespace Nektar::SolverCore
{

class FilterAverageFields : public Filter
{
public:
    friend class MemoryManager<FilterAverageFields>;

    static FilterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params)
    {
        return MemoryManager<FilterAverageFields>::AllocateSharedPtr(session,
                                                                     params);
    }

    static std::string cmdSetStartFilterFileNum;
    static std::string className;

protected:
    SOLVER_CORE_EXPORT FilterAverageFields(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params);
    SOLVER_CORE_EXPORT ~FilterAverageFields() override = default;

    SOLVER_CORE_EXPORT void v_Initialise(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Apply(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Finalise(FilterContext &context) override;

private:
    std::string m_executionSpace;
    Math::MathHelper m_math;
    CoefficientFieldViews m_sourceFields;
    std::vector<LibUtilities::Field<double, FieldState::Coeff>> m_accumulators;
    LibUtilities::FieldMetaDataMap m_fieldMetaData;

    unsigned int m_numSamples      = 0;
    unsigned int m_outputFrequency = 0;
    unsigned int m_sampleFrequency = 1;

    std::string m_outputFile;
    std::string m_restartFile;

    bool m_phaseSample         = false;
    double m_phaseSamplePeriod = 0.0;
    double m_phaseSamplePhase  = 0.0;
    double m_phaseTolerance    = 0.0;
    StepSchedule m_sampleSchedule;
    StepSchedule m_outputSchedule;
    std::unique_ptr<OutputSeries> m_series;

    bool IsPhaseSample(const double time) const;
    void AccumulateSample(FilterContext &context);
    void WriteAverage(FilterContext &context, bool numbered = false);
    void LoadRestartFile();
};

} // namespace Nektar::SolverCore
