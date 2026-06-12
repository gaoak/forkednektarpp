///////////////////////////////////////////////////////////////////////////////
//
// File: Driver.h
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

#pragma once

#include <iostream>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/Communication/Comm.h>
#include <SolverCore/EquationSystems/EquationSystem.h>
#include <SolverCore/SolverCore.hpp>
#include <SpatialDomains/MeshGraph.h>

namespace Nektar::SolverCore
{

class Driver;

using DriverSharedPtr = std::shared_ptr<Driver>;
using DriverFactory =
    LibUtilities::NekFactory<std::string, Driver,
                             const LibUtilities::SessionReaderSharedPtr &,
                             const SpatialDomains::MeshGraphSharedPtr &>;

SOLVER_CORE_EXPORT DriverFactory &GetDriverFactory();

class Driver
{
public:
    SOLVER_CORE_EXPORT virtual ~Driver() = default;

    inline void InitObject(std::ostream &out = std::cout)
    {
        v_InitObject(out);
    }

    inline void Execute(std::ostream &out = std::cout)
    {
        v_Execute(out);
    }

    inline std::vector<EquationSystemSharedPtr> GetEqu()
    {
        return m_equ;
    }

protected:
    SOLVER_CORE_EXPORT Driver(
        const LibUtilities::SessionReaderSharedPtr &session,
        const SpatialDomains::MeshGraphSharedPtr &graph);

    SOLVER_CORE_EXPORT virtual void v_InitObject(std::ostream &out = std::cout);
    SOLVER_CORE_EXPORT virtual void v_Execute(
        std::ostream &out = std::cout) = 0;

    LibUtilities::CommSharedPtr m_comm;
    LibUtilities::SessionReaderSharedPtr m_session;
    SpatialDomains::MeshGraphSharedPtr m_graph;
    std::vector<EquationSystemSharedPtr> m_equ;

    static std::string driverDefault;
};

} // namespace Nektar::SolverCore
