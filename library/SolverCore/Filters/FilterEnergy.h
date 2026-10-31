///////////////////////////////////////////////////////////////////////////////
//
// File: FilterEnergy.h
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
// Description: Outputs kinetic energy and enstrophy during time-stepping.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <fstream>
#include <memory>
#include <string>

#include <SolverCore/Energy/EnergyOp.hpp>
#include <SolverCore/Filters/Filter.h>
#include <SolverCore/Filters/FilterUtils.h>

namespace Nektar::SolverCore
{

class FilterEnergy : public Filter
{
public:
    friend class MemoryManager<FilterEnergy>;

    static FilterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params)
    {
        return MemoryManager<FilterEnergy>::AllocateSharedPtr(session, params);
    }

    static std::string className;

protected:
    SOLVER_CORE_EXPORT FilterEnergy(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params);
    SOLVER_CORE_EXPORT ~FilterEnergy() override = default;

    SOLVER_CORE_EXPORT void v_Initialise(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Apply(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Finalise(FilterContext &context) override;

private:
    std::shared_ptr<EnergyOp<double>> m_energyOp;
    std::string m_executionSpace;
    std::string m_outputFile;
    unsigned int m_outputFrequency = 1;
    StepSchedule m_schedule;
    std::unique_ptr<RootTableWriter> m_writer;

    void WriteHeader();
    void WriteEnergy(const double time, const double kineticEnergy,
                     const double enstrophy);
};

} // namespace Nektar::SolverCore
