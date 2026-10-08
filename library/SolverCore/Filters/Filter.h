///////////////////////////////////////////////////////////////////////////////
//
// File: Filter.h
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
// Description: Base class for SolverCore filters.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverCore/EquationSystems/EquationSystem.h>
#include <SolverCore/SolverCoreDeclspec.h>

namespace Nektar::SolverCore
{

class Filter;

using FilterSharedPtr = std::shared_ptr<Filter>;

using FilterFactory =
    LibUtilities::NekFactory<std::string, Filter,
                             const LibUtilities::SessionReaderSharedPtr &,
                             const EquationSystemSharedPtr &,
                             const std::map<std::string, std::string> &>;

SOLVER_CORE_EXPORT FilterFactory &GetFilterFactory();

class Filter
{
public:
    using ParamMap = std::map<std::string, std::string>;

    SOLVER_CORE_EXPORT Filter(
        const LibUtilities::SessionReaderSharedPtr &session,
        const EquationSystemSharedPtr &equation);
    SOLVER_CORE_EXPORT virtual ~Filter() = default;

    void Initialise(const double time)
    {
        v_Initialise(time);
    }

    void Apply(const double time)
    {
        v_Apply(time);
    }

    void Finalise(const double time)
    {
        v_Finalise(time);
    }

    bool IsTimeDependent()
    {
        return v_IsTimeDependent();
    }

    void SetApplyOnInitialise(const bool flag)
    {
        m_applyOnInitialise = flag;
    }

protected:
    SOLVER_CORE_EXPORT std::string SetupOutput(const std::string &ext,
                                               const ParamMap &params);
    SOLVER_CORE_EXPORT std::string SetupOutput(const std::string &ext,
                                               const std::string &inname);

    LibUtilities::SessionReaderSharedPtr m_session;
    std::weak_ptr<EquationSystem> m_equ;
    bool m_applyOnInitialise = true;

    virtual void v_Initialise(const double time) = 0;
    virtual void v_Apply(const double time)      = 0;
    virtual void v_Finalise(const double time)   = 0;
    virtual bool v_IsTimeDependent()             = 0;
};

} // namespace Nektar::SolverCore
