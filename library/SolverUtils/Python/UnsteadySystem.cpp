//////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadySystem.cpp
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
// Description: Python wrapper for UnsteadySystem.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Python/BasicUtils/NekFactory.hpp>
#include <LibUtilities/Python/NekPyConfig.hpp>

#include <SolverUtils/UnsteadySystem.h>

#include <SolverUtils/Python/EquationSystem.h>

using namespace Nektar;
using namespace Nektar::SolverUtils;

void export_UnsteadySystem()
{
    using USWrap = EquationSystemWrap<UnsteadySystem>;

    py::class_<USWrap, std::shared_ptr<USWrap>, py::bases<EquationSystem>,
               boost::noncopyable>(
        "UnsteadySystem", py::init<LibUtilities::SessionReaderSharedPtr,
                                   SpatialDomains::MeshGraphSharedPtr>())

        // Add overrides for this class
        .def("InitObject", &EquationSystem::InitObject,
             &USWrap::Default_v_InitObject)
        .def("DoInitialise", &EquationSystem::DoInitialise,
             &USWrap::Default_v_DoInitialise)
        .def("DoSolve", &EquationSystem::DoSolve, &USWrap::Default_v_DoSolve)
        .def("SetInitialConditions", &EquationSystem::SetInitialConditions,
             &USWrap::Default_v_SetInitialConditions)
        .def("EvaluateExactSolution", &USWrap::v_EvaluateExactSolution,
             &USWrap::Default_v_EvaluateExactSolution)
        .def("LinfError", &USWrap::v_LinfError, &USWrap::Default_v_LinfError)
        .def("L2Error", &USWrap::v_L2Error, &USWrap::Default_v_L2Error)

        // Add properties for time integration
        .add_property(
            "ode",
            py::make_function(
                &UnsteadySystem::GetTimeIntegrationSchemeOperators,
                py::return_value_policy<py::reference_existing_object>()))
        .add_property(
            "int_scheme",
            py::make_function(
                &UnsteadySystem::GetTimeIntegrationScheme,
                py::return_value_policy<py::reference_existing_object>()));

    WrapConverter<UnsteadySystem>();
}
