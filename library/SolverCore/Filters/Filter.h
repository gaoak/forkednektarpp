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
#include <SolverCore/Filters/FilterContext.h>
#include <SolverCore/SolverCoreDeclspec.h>

namespace Nektar::SolverCore
{

class Filter;

using FilterSharedPtr = std::shared_ptr<Filter>;

using FilterFactory =
    LibUtilities::NekFactory<std::string, Filter,
                             const LibUtilities::SessionReaderSharedPtr &,
                             const std::map<std::string, std::string> &>;

SOLVER_CORE_EXPORT FilterFactory &GetFilterFactory();

class Filter
{
public:
    using ParamMap = std::map<std::string, std::string>;

    SOLVER_CORE_EXPORT virtual ~Filter() = default;

    void Initialise(FilterContext &context)
    {
        v_Initialise(context);
    }

    void Apply(FilterContext &context)
    {
        v_Apply(context);
    }

    void Finalise(FilterContext &context)
    {
        v_Finalise(context);
    }

protected:
    LibUtilities::SessionReaderSharedPtr m_session;

    SOLVER_CORE_EXPORT explicit Filter(
        const LibUtilities::SessionReaderSharedPtr &session);

    SOLVER_CORE_EXPORT std::string SetupOutput(const std::string &ext,
                                               const ParamMap &params);
    SOLVER_CORE_EXPORT std::string SetupOutput(const std::string &ext,
                                               const std::string &inname);

    virtual void v_Initialise(FilterContext &context) = 0;
    virtual void v_Apply(FilterContext &context)      = 0;
    virtual void v_Finalise(FilterContext &context)   = 0;
};

} // namespace Nektar::SolverCore
