///////////////////////////////////////////////////////////////////////////////
//
// File: Forcing.cpp
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

#include <SolverCore/Forcing/Forcing.h>

namespace Nektar::SolverCore
{

ForcingFactory &GetForcingFactory()
{
    static ForcingFactory instance;
    return instance;
}

Forcing::Forcing(const LibUtilities::SessionReaderSharedPtr &session,
                 const MultiRegions::ExpListSharedPtr &expList,
                 std::vector<std::string> variables)
    : m_session(session), m_expList(expList), m_variables(std::move(variables))
{
}

std::vector<ForcingSharedPtr> Forcing::Load(
    const LibUtilities::SessionReaderSharedPtr &session,
    const MultiRegions::ExpListSharedPtr &expList,
    const std::vector<std::string> &variables)
{
    std::vector<ForcingSharedPtr> forcings;

    if (!session->DefinesElement("Nektar/Forcing"))
    {
        return forcings;
    }

    TiXmlElement *forcingElement = session->GetElement("Nektar/Forcing");
    ASSERTL0(forcingElement, "Invalid Forcing element in session file.");

    TiXmlElement *force = forcingElement->FirstChildElement("FORCE");
    while (force)
    {
        const char *typeAttr = force->Attribute("TYPE");
        ASSERTL0(typeAttr, "Forcing FORCE block requires a TYPE attribute.");

        TiXmlElement *forceParam = force;
        LibUtilities::SessionReader::GetXMLElementTimeLevel(
            forceParam, session->GetTimeLevel(), false);

        forcings.push_back(GetForcingFactory().CreateInstance(
            typeAttr, session, expList, variables, forceParam));

        force = force->NextSiblingElement("FORCE");
    }

    return forcings;
}

void Forcing::v_SetAppend([[maybe_unused]] const bool append)
{
}

SessionFunctionSharedPtr Forcing::GetFunction(const std::string &name,
                                              bool cache)
{
    if (!cache)
    {
        return std::make_shared<SessionFunction>(m_session, m_expList, name);
    }

    auto function = m_sessionFunctions.find(name);
    if (function == m_sessionFunctions.end())
    {
        function = m_sessionFunctions
                       .emplace(name, std::make_shared<SessionFunction>(
                                          m_session, m_expList, name))
                       .first;
    }

    return function->second;
}

} // namespace Nektar::SolverCore
