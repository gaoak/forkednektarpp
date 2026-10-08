///////////////////////////////////////////////////////////////////////////////
//
// File: Driver.cpp
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
// Description: Base class for SolverCore drivers.
//
///////////////////////////////////////////////////////////////////////////////

#include <SolverCore/Driver/Driver.h>

namespace Nektar::SolverCore
{

std::string Driver::driverDefault =
    LibUtilities::SessionReader::RegisterDefaultSolverInfo("Driver",
                                                           "Standard");

DriverFactory &GetDriverFactory()
{
    static DriverFactory instance;
    return instance;
}

Driver::Driver(const LibUtilities::SessionReaderSharedPtr &session,
               const SpatialDomains::MeshGraphSharedPtr &graph)
    : m_comm(session->GetComm()), m_session(session), m_graph(graph)
{
}

void Driver::v_InitObject(std::ostream &out)
{
    try
    {
        // Assumes EquationSystem is named after either EqType, or, if
        // specified, the SolverType. i.e. SolverType overwrites EqType
        ASSERTL0(m_session->DefinesSolverInfo("EqType"),
                 "EqType SolverInfo tag must be defined.");

        std::string equationName = m_session->GetSolverInfo("EqType");
        if (m_session->DefinesSolverInfo("SolverType"))
        {
            equationName = m_session->GetSolverInfo("SolverType");
        }

        ASSERTL0(
            GetEquationSystemFactory().ModuleExists(equationName),
            "EquationSystem '" + equationName +
                "' is not defined.\n"
                "Ensure equation name is correct and module is compiled.\n");

        // Create EquationSystem
        // By default, we assume a single equation system
        m_equ    = std::vector<EquationSystemSharedPtr>(1);
        m_equ[0] = GetEquationSystemFactory().CreateInstance(
            equationName, m_session, m_graph);
    }
    catch (int e)
    {
        ASSERTL0(e == -1, "No such class defined.");
        out << "An error occurred during driver initialisation." << std::endl;
    }
}

} // namespace Nektar::SolverCore
