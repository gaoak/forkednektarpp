///////////////////////////////////////////////////////////////////////////////
//
// File: Forcing.h
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
// Description: Abstract base class for forcing terms.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>
#include <Operators/Field/Field.hpp>
#include <Operators/Math/Math.hpp>
#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/SolverCoreDeclspec.h>

namespace Nektar::SolverCore
{
class Forcing;

using ForcingSharedPtr = std::shared_ptr<Forcing>;

/// Declaration of the forcing factory.
typedef LibUtilities::NekFactory<
    std::string, Forcing, const LibUtilities::SessionReaderSharedPtr &,
    const MultiRegions::ExpListSharedPtr &, const std::vector<std::string> &,
    const TiXmlElement *>
    ForcingFactory;

/// Declaration of the forcing factory singleton.
SOLVER_CORE_EXPORT ForcingFactory &GetForcingFactory();

/**
 * @class Forcing
 * @brief Defines a forcing term to be explicitly applied to a device support
 * Field.
 */
class Forcing
{
public:
    SOLVER_CORE_EXPORT virtual ~Forcing() = default;

    SOLVER_CORE_EXPORT void InitObject(const TiXmlElement *force)
    {
        v_InitObject(force);
    }

    SOLVER_CORE_EXPORT void Apply(
        Operators::Field<double, FieldState::Phys> &in,
        Operators::Field<double, FieldState::Phys> &out, const double time,
        const double scale = 1.0)
    {
        v_Apply(in, out, time, scale);
    }

    SOLVER_CORE_EXPORT void SetAppend(const bool append)
    {
        v_SetAppend(append);
    }

    SOLVER_CORE_EXPORT static std::vector<ForcingSharedPtr> Load(
        const LibUtilities::SessionReaderSharedPtr &session,
        const MultiRegions::ExpListSharedPtr &expList,
        const std::vector<std::string> &variables);

protected:
    SOLVER_CORE_EXPORT Forcing(
        const LibUtilities::SessionReaderSharedPtr &session,
        const MultiRegions::ExpListSharedPtr &expList,
        std::vector<std::string> variables);

    SOLVER_CORE_EXPORT virtual void v_InitObject(const TiXmlElement *force) = 0;

    SOLVER_CORE_EXPORT virtual void v_Apply(
        Operators::Field<double, FieldState::Phys> &in,
        Operators::Field<double, FieldState::Phys> &out, const double time,
        const double scale) = 0;

    SOLVER_CORE_EXPORT virtual void v_SetAppend(const bool append);

    SOLVER_CORE_EXPORT SessionFunctionSharedPtr
    GetFunction(const std::string &name, bool cache = true);

    LibUtilities::SessionReaderSharedPtr m_session;
    MultiRegions::ExpListSharedPtr m_expList;
    std::vector<std::string> m_variables;
    std::map<std::string, SessionFunctionSharedPtr> m_sessionFunctions;
    Operators::Math m_math;
};
} // namespace Nektar::SolverCore
