///////////////////////////////////////////////////////////////////////////////
//
// File: FilterCheckpoint.h
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

#pragma once

#include <SolverCore/Filters/Filter.h>

namespace Nektar::SolverCore
{

class FilterCheckpoint : public Filter
{
public:
    static FilterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const EquationSystemSharedPtr &equation, const ParamMap &params)
    {
        return MemoryManager<FilterCheckpoint>::AllocateSharedPtr(
            session, equation, params);
    }

    static std::string cmdSetStartFilterFileNum;
    static std::string className;

    SOLVER_CORE_EXPORT FilterCheckpoint(
        const LibUtilities::SessionReaderSharedPtr &session,
        const EquationSystemSharedPtr &equation, const ParamMap &params);
    SOLVER_CORE_EXPORT ~FilterCheckpoint() override = default;

protected:
    SOLVER_CORE_EXPORT void v_Initialise(const double time) override;
    SOLVER_CORE_EXPORT void v_Apply(const double time) override;
    SOLVER_CORE_EXPORT void v_Finalise(const double time) override;
    SOLVER_CORE_EXPORT bool v_IsTimeDependent() override;

private:
    unsigned int m_index           = 0;
    unsigned int m_outputIndex     = 0;
    unsigned int m_outputFrequency = 0;
    std::string m_outputFile;
    // Time after which checkpoint files should be written.
    double m_outputStartTime = 0.0;
};

} // namespace Nektar::SolverCore
