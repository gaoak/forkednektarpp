///////////////////////////////////////////////////////////////////////////////
//
// File: ForcingBody.cpp
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
// Description: Body forcing
//
///////////////////////////////////////////////////////////////////////////////

#include <SolverCore/Forcing/ForcingBody.h>

namespace Nektar::SolverCore
{

std::string ForcingBody::className =
    GetForcingFactory().RegisterCreatorFunction("Body", ForcingBody::create,
                                                "Body Forcing");

ForcingBody::ForcingBody(const LibUtilities::SessionReaderSharedPtr &session,
                         const MultiRegions::ExpListSharedPtr &expList,
                         const std::vector<std::string> &variables)
    : Forcing(session, expList, variables)
{
}

void ForcingBody::v_InitObject(const TiXmlElement *force)
{
    ASSERTL0(force, "ForcingBody requires a valid FORCE XML element.");

    // Read function name from BODYFORCE. Both tags map to the
    // same minimal implementation, preserving the legacy session-file syntax.
    const TiXmlElement *funcNameElmt = force->FirstChildElement("BODYFORCE");

    m_funcName = funcNameElmt->GetText() ? funcNameElmt->GetText() : "";
    ASSERTL0(!m_funcName.empty(), "BODYFORCE must specify a function name.");
    ASSERTL0(m_session->DefinesFunction(m_funcName),
             "Function '" + m_funcName + "' not defined.");

    std::vector<LibUtilities::EquationSharedPtr> forcingEquations;
    forcingEquations.reserve(m_variables.size());
    for (const auto &var : m_variables)
    {
        if (m_session->DefinesFunction(m_funcName, var))
        {
            const auto functionType =
                m_session->GetFunctionType(m_funcName, var);
            ASSERTL0(functionType == LibUtilities::eFunctionTypeExpression,
                     "SolverCore ForcingBody currently supports expression "
                     "functions only.");
            forcingEquations.push_back(m_session->GetFunction(m_funcName, var));
        }
        else
        {
            // Preserve legacy behaviour where forcing can be defined for only a
            // subset of variables: missing components receive zero forcing.
            forcingEquations.push_back(
                MemoryManager<LibUtilities::Equation>::AllocateSharedPtr(
                    m_session->GetInterpreter(), "0"));
        }
    }

    m_forceOp = Operators::ExpressionOp<double>::Create(m_expList, m_variables);
    m_forceOp->SetExpressions(forcingEquations);
    m_forceOp->SetAppend(true);
}

void ForcingBody::v_Apply(MultiRegions::Field<double, FieldState::Phys> &in,
                          MultiRegions::Field<double, FieldState::Phys> &out,
                          const double time, const double scale)
{
    ASSERTL0(m_forceOp, "ForcingBody has not been initialised.");

    m_forceOp->SetTime(time);
    m_forceOp->SetScale(scale);

    // Add or assign the forcing according to the ExpressionOp append mode.
    m_forceOp->Apply(in, out);
}

void ForcingBody::v_SetAppend(const bool append)
{
    ASSERTL0(m_forceOp, "ForcingBody has not been initialised.");
    m_forceOp->SetAppend(append);
}

} // namespace Nektar::SolverCore
