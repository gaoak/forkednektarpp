///////////////////////////////////////////////////////////////////////////////
//
// File: ForcingBody.h
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
// Description: Body or Field forcing
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <Operators/ElmtOps/Expression/ExpressionOp.hpp>
#include <SolverCore/Forcing/Forcing.h>
#include <SolverCore/SolverCoreDeclspec.h>

namespace Nektar::SolverCore
{
/**
 * @brief Body/field forcing evaluated from a session function.
 *
 * This device support version intentionally starts with the minimal explicit
 * physical-space path: evaluate the configured function with ExpressionOp and
 * add it to the output Field.
 */
class ForcingBody : public Forcing
{
public:
    friend class MemoryManager<ForcingBody>;

    /// Creates an instance of this class.
    SOLVER_CORE_EXPORT static ForcingSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &session,
        const MultiRegions::ExpListSharedPtr &expList,
        const std::vector<std::string> &variables, const TiXmlElement *force)
    {
        auto forcing = MemoryManager<ForcingBody>::AllocateSharedPtr(
            session, expList, variables);
        forcing->InitObject(force);
        return forcing;
    }

    /// Name of the class.
    static std::string className;

protected:
    SOLVER_CORE_EXPORT void v_InitObject(const TiXmlElement *force) override;

    SOLVER_CORE_EXPORT void v_Apply(
        MultiRegions::Field<double, FieldState::Phys> &in,
        MultiRegions::Field<double, FieldState::Phys> &out, const double time,
        const double scale) override;

    SOLVER_CORE_EXPORT void v_SetAppend(const bool append) override;

private:
    std::string m_funcName;
    std::shared_ptr<Operators::ExpressionOp<double>> m_forceOp;

    ForcingBody(const LibUtilities::SessionReaderSharedPtr &session,
                const MultiRegions::ExpListSharedPtr &expList,
                const std::vector<std::string> &variables);

    ~ForcingBody() override = default;
};

} // namespace Nektar::SolverCore
