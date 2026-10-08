///////////////////////////////////////////////////////////////////////////////
//
// File: DriverStandard.cpp
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
// Description: Standard driver for device support solvers.
//
///////////////////////////////////////////////////////////////////////////////

#include <SolverCore/Driver/DriverStandard.h>

#include <LibUtilities/Backends/Backends.hpp>
#include <boost/algorithm/string/predicate.hpp>

namespace Nektar::SolverCore
{

std::string DriverStandard::className =
    GetDriverFactory().RegisterCreatorFunction("Standard",
                                               DriverStandard::create);
std::string DriverStandard::driverLookupId =
    LibUtilities::SessionReader::RegisterEnumValue("Driver", "Standard", 0);

DriverStandard::DriverStandard(
    const LibUtilities::SessionReaderSharedPtr &session,
    const SpatialDomains::MeshGraphSharedPtr &graph)
    : Driver(session, graph)
{
}

void DriverStandard::v_InitObject(std::ostream &out)
{
    Driver::v_InitObject(out);
}

void DriverStandard::v_Execute(std::ostream &out)
{
    Nektar::LibUtilities::Timer timer;
    m_equ[0]->PrintSummary(out);

    timer.Start();
    m_equ[0]->DoInitialise();
    m_equ[0]->DoSolve();
    // A device runs asynchronously, so wait for the work submitted above to
    // finish before the timer is read; otherwise the time reported is the
    // host's, up to its last submission.
    nekDeviceSynchronize();
    timer.Stop();

    m_equ[0]->Output();

    if (m_comm->GetRank() == 0)
    {
        NekDouble CpuTime;
        CpuTime = timer.Elapsed().count();

        // The execution space follows the operators': the command-line
        // argument when it is given, and otherwise whatever the build
        // provides.
        const std::string execSpace =
            m_session->DefinesCmdLineArgument("opExecSpace")
                ? m_session->GetCmdLineArgument<std::string>("opExecSpace")
                : (nekGetNumDevice() ? "Device" : "Host");

        out << "-------------------------------------------" << std::endl;
        out << "Total Computation Time = " << CpuTime << std::endl;
        out << "-------------------------------------------" << std::endl;
    }

    m_equ[0]->PrintNorms(out);
}
} // namespace Nektar::SolverCore
