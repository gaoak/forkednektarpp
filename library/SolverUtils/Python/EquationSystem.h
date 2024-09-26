//////////////////////////////////////////////////////////////////////////////
//
// File: EquationSystem.h
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
// Description: Python wrapper for EquationSystem.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIBRARY_SOLVERUTILS_PYTHON_EQUATIONSYSTEM_H
#define NEKTAR_LIBRARY_SOLVERUTILS_PYTHON_EQUATIONSYSTEM_H

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/Python/NekPyConfig.hpp>
#include <SpatialDomains/MeshGraph.h>

using namespace Nektar;

/**
 * @brief EquationSystem wrapper to handle virtual function calls in @c
 * EquationSystem and its subclasses.
 */
template <class T> struct EquationSystemWrap : public T, public py::wrapper<T>
{
    /**
     * @brief Constructor, which is identical to Filter.
     *
     * @param field  Input field.
     */
    EquationSystemWrap(LibUtilities::SessionReaderSharedPtr session,
                       SpatialDomains::MeshGraphSharedPtr graph)
        : T(session, graph), py::wrapper<T>()
    {
    }

    void v_InitObject(bool declareField) override
    {
        if (py::override f = this->get_override("InitObject")(declareField))
        {
            f();
        }
        else
        {
            T::v_InitObject();
        }
    }

    void Default_v_InitObject(bool declareField)
    {
        return this->T::v_InitObject(declareField);
    }

    void v_DoInitialise(bool dumpInitialConditions) override
    {
        if (py::override f = this->get_override("DoInitialise"))
        {
            f(dumpInitialConditions);
        }
        else
        {
            T::v_DoInitialise(dumpInitialConditions);
        }
    }

    void Default_v_DoInitialise(bool dumpInitialConditions)
    {
        this->T::v_DoInitialise(dumpInitialConditions);
    }

    void v_DoSolve() override
    {
        if (py::override f = this->get_override("DoSolve"))
        {
            f();
        }
        else
        {
            T::v_DoSolve();
        }
    }

    void Default_v_DoSolve()
    {
        this->T::v_DoSolve();
    }

    void v_SetInitialConditions(NekDouble initialtime,
                                bool dumpInitialConditions,
                                const int domain) override
    {
        if (py::override f = this->get_override("SetInitialConditions"))
        {
            f(initialtime, dumpInitialConditions, domain);
        }
        else
        {
            T::v_SetInitialConditions(initialtime, dumpInitialConditions,
                                      domain);
        }
    }

    void Default_v_SetInitialConditions(NekDouble initialtime,
                                        bool dumpInitialConditions,
                                        const int domain)
    {
        this->T::v_SetInitialConditions(initialtime, dumpInitialConditions,
                                        domain);
    }

    Array<OneD, NekDouble> v_EvaluateExactSolution(unsigned int field,
                                                   const NekDouble time)
    {
        if (py::override f = this->get_override("EvaluateExactSolution"))
        {
            py::object tmpPy = f(field, time);
            Array<OneD, NekDouble> tmp =
                py::extract<Array<OneD, NekDouble>>(tmpPy);
            return tmp;
        }
        else
        {
            Array<OneD, NekDouble> outfield(
                this->m_fields[field]->GetNpoints());
            T::v_EvaluateExactSolution(field, outfield, time);
            return outfield;
        }
    }

    Array<OneD, NekDouble> Default_v_EvaluateExactSolution(unsigned int field,
                                                           const NekDouble time)
    {
        Array<OneD, NekDouble> outfield(this->m_fields[field]->GetNpoints());
        this->T::v_EvaluateExactSolution(field, outfield, time);
        return outfield;
    }

    NekDouble v_LinfError(unsigned int field)
    {
        if (py::override f = this->get_override("LinfError"))
        {
            return f(field);
        }

        return T::v_LinfError(field);
    }

    NekDouble Default_v_LinfError(unsigned int field)
    {
        return this->T::v_LinfError(field);
    }

    NekDouble v_L2Error(unsigned int field)
    {
        if (py::override f = this->get_override("L2Error"))
        {
            return f(field);
        }

        return T::v_L2Error(field);
    }

    NekDouble Default_v_L2Error(unsigned int field)
    {
        return this->T::v_L2Error(field);
    }
};

#endif
