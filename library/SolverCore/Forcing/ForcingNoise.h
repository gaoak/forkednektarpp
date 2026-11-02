///////////////////////////////////////////////////////////////////////////////
//
// File: ForcingNoise.h
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
// Description: White noise forcing
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/Math/RandomGenerator.hpp>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <SolverCore/Forcing/Forcing.h>
#include <SolverCore/SolverCoreDeclspec.h>

namespace Nektar::SolverCore
{
/**
 * @brief White noise forcing: normal noise of zero mean and standard
 * deviation WHITENOISE, redrawn every UPDATEFREQ applications and applied for
 * the first NSTEPS applications.
 */
class ForcingNoise : public Forcing
{
public:
    friend class MemoryManager<ForcingNoise>;

    /// Creates an instance of this class.
    SOLVER_CORE_EXPORT static ForcingSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const MultiRegions::ExpListSharedPtr &expList,
        const std::vector<std::string> &variables,
        const LibUtilities::SessionNode &force)
    {
        auto forcing = MemoryManager<ForcingNoise>::AllocateSharedPtr(
            session, expList, variables);
        forcing->InitObject(force);
        return forcing;
    }

    /// Name of the class.
    static std::string className;

protected:
    SOLVER_CORE_EXPORT void v_InitObject(
        const LibUtilities::SessionNode &force) override;

    SOLVER_CORE_EXPORT void v_Apply(
        LibUtilities::Field<double, FieldState::Phys> &in,
        LibUtilities::Field<double, FieldState::Phys> &out, const double time,
        const double scale) override;

    SOLVER_CORE_EXPORT void v_SetAppend(const bool append) override;

private:
    double m_sigma            = 0.0;
    unsigned int m_updateFreq = 0;
    unsigned int m_numSteps   = 0;
    unsigned int m_index      = 0;
    bool m_append             = false;
    Math::RandomGenerator m_randomGenerator;
    LibUtilities::Field<double, FieldState::Phys> m_noise;

    ForcingNoise(const LibUtilities::SessionReaderSharedPtr &session,
                 const MultiRegions::ExpListSharedPtr &expList,
                 const std::vector<std::string> &variables);

    ~ForcingNoise() override = default;
};

} // namespace Nektar::SolverCore
