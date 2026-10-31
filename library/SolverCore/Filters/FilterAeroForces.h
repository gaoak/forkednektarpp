///////////////////////////////////////////////////////////////////////////////
//
// File: FilterAeroForces.h
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
// Description: Outputs aerodynamic forces during time-stepping.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <SolverCore/AeroForces/AeroForcesOp.hpp>
#include <SolverCore/Filters/Filter.h>
#include <SolverCore/Filters/FilterUtils.h>

namespace Nektar::SolverCore
{

class FilterAeroForces : public Filter
{
public:
    friend class MemoryManager<FilterAeroForces>;

    static FilterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params)
    {
        return MemoryManager<FilterAeroForces>::AllocateSharedPtr(session,
                                                                  params);
    }

    static std::string className;

protected:
    SOLVER_CORE_EXPORT FilterAeroForces(
        const LibUtilities::SessionReaderSharedPtr &session,
        const ParamMap &params);
    SOLVER_CORE_EXPORT ~FilterAeroForces() override = default;

    SOLVER_CORE_EXPORT void v_Initialise(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Apply(FilterContext &context) override;
    SOLVER_CORE_EXPORT void v_Finalise(FilterContext &context) override;

private:
    MultiRegions::ExpListSharedPtr m_velocityExp;
    MultiRegions::ExpListSharedPtr m_pressureExp;
    std::shared_ptr<AeroForcesOp<double>> m_aeroForcesOp;
    std::string m_executionSpace;

    std::vector<unsigned int> m_boundaryRegions;
    std::vector<bool> m_boundaryRegionIsInList;

    std::array<std::array<double, 3>, 3> m_directions{};
    std::array<double, 3> m_pivotPoint{};
    std::vector<double> m_pressureForce;
    std::vector<double> m_viscousForce;
    std::vector<double> m_pressureMoment;
    std::vector<double> m_viscousMoment;

    unsigned int m_spaceDim        = 0;
    unsigned int m_momentDim       = 0;
    unsigned int m_outputFrequency = 1;
    double m_rho                   = 1.0;
    double m_mu                    = 1.0;
    std::string m_boundaryString;
    std::string m_outputFile;
    StepSchedule m_schedule;
    std::unique_ptr<RootTableWriter> m_writer;

    void CalculateForces(FilterContext &context);
    void WriteHeader(const std::string &boundaryString);
    void WriteForces(const double time);
};

} // namespace Nektar::SolverCore
