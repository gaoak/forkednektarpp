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
    m_equ[0]->PrintSummary(out);
    m_equ[0]->DoInitialise();
    m_equ[0]->DoSolve();
    m_equ[0]->Output();

    if (m_comm->GetRank() == 0)
    {
        out << "-------------------------------------------" << std::endl;
        out << "Total Computation Time = "
            << "NO TIMER IMPLEMENTED" << std::endl;
        out << "-------------------------------------------" << std::endl;
    }

    m_equ[0]->PrintNorms(out);
}
} // namespace Nektar::SolverCore
