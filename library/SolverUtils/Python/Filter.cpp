///////////////////////////////////////////////////////////////////////////////
//
// File: Filter.cpp
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
// Description: Python wrapper for Filter.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Python/BasicUtils/NekFactory.hpp>
#include <LibUtilities/Python/NekPyConfig.hpp>

#include <SolverUtils/Filters/Filter.h>

using namespace Nektar;
using namespace Nektar::SolverUtils;

/**
 * @brief Filter wrapper to handle virtual function calls in @c Filter and its
 * subclasses.
 */
struct FilterWrap : public Filter, public py::wrapper<Filter>
{
    /**
     * @brief Constructor, which is identical to Filter.
     *
     * @param field  Input field.
     */
    FilterWrap(LibUtilities::SessionReaderSharedPtr session,
               std::shared_ptr<EquationSystem> eqn)
        : Filter(session, eqn), py::wrapper<Filter>()
    {
    }

    py::list ArrayOneDToPyList(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields)
    {
        py::list expLists;

        for (int i = 0; i < pFields.size(); ++i)
        {
            expLists.append(py::object(pFields[i]));
        }

        return expLists;
    }

    void v_Initialise(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        const NekDouble &time) override
    {
        this->get_override("Initialise")(ArrayOneDToPyList(pFields), time);
    }

    void v_Update(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        const NekDouble &time) override
    {
        this->get_override("Update")(ArrayOneDToPyList(pFields), time);
    }

    void v_Finalise(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        const NekDouble &time) override
    {
        this->get_override("Finalise")(ArrayOneDToPyList(pFields), time);
    }

    bool v_IsTimeDependent() override
    {
        return this->get_override("IsTimeDependent")();
    }
};

/**
 * @brief Lightweight wrapper for Filter factory creation function.
 */
FilterSharedPtr Filter_Create(py::tuple args, py::dict kwargs)
{
    using NekError = ErrorUtil::NekError;

    if (py::len(args) != 3)
    {
        throw NekError("Filter.Create() requires three arguments: "
                       "filter name, a SessionReader object and an "
                       "EquationSystem object.");
    }

    std::string filterName = py::extract<std::string>(args[0]);

    if (!py::extract<LibUtilities::SessionReaderSharedPtr>(args[1]).check())
    {
        throw NekError("Second argument to Create() should be a SessionReader "
                       "object.");
    }

    if (!py::extract<EquationSystemSharedPtr>(args[2]).check())
    {
        throw NekError("Second argument to Create() should be a EquationSystem "
                       "object.");
    }

    LibUtilities::SessionReaderSharedPtr session =
        py::extract<LibUtilities::SessionReaderSharedPtr>(args[1]);
    EquationSystemSharedPtr eqn = py::extract<EquationSystemSharedPtr>(args[2]);

    // Process keyword arguments.
    py::list items = kwargs.items();
    Filter::ParamMap params;

    for (int i = 0; i < py::len(items); ++i)
    {
        std::string arg = py::extract<std::string>(items[i][0]);
        std::string val =
            py::extract<std::string>(items[i][1].attr("__str__")());
        params[arg] = val;
    }

    std::shared_ptr<Filter> filter =
        GetFilterFactory().CreateInstance(filterName, session, eqn, params);

    return filter;
}

void Filter_Initialise(std::shared_ptr<Filter> filter,
                       Array<OneD, MultiRegions::ExpListSharedPtr> expLists,
                       NekDouble time)
{
    filter->Initialise(expLists, time);
}

void Filter_Update(std::shared_ptr<Filter> filter,
                   Array<OneD, MultiRegions::ExpListSharedPtr> expLists,
                   NekDouble time)
{
    filter->Update(expLists, time);
}

void Filter_Finalise(std::shared_ptr<Filter> filter,
                     Array<OneD, MultiRegions::ExpListSharedPtr> expLists,
                     NekDouble time)
{
    filter->Finalise(expLists, time);
}

bool Filter_IsTimeDependent(std::shared_ptr<Filter> filter)
{
    return filter->IsTimeDependent();
}

void export_Filter()
{
    static NekFactory_Register<FilterFactory> fac(GetFilterFactory());

    // Define FilterWrap to be implicitly convertible to a Filter, since it
    // seems that doesn't sometimes get picked up.
    py::implicitly_convertible<std::shared_ptr<FilterWrap>,
                               std::shared_ptr<Filter>>();

    // Wrapper for the Filter class. Note that since Filter contains a pure
    // virtual function, we need the FilterWrap helper class to handle this for
    // us. In the lightweight wrappers above, we therefore need to ensure we're
    // passing std::shared_ptr<Filter> as the first argument, otherwise they
    // won't accept objects constructed from Python.
    py::class_<FilterWrap, std::shared_ptr<FilterWrap>, boost::noncopyable>(
        "Filter", py::init<LibUtilities::SessionReaderSharedPtr,
                           EquationSystemSharedPtr>())

        .def("Initialise", &Filter_Initialise)
        .def("Update", &Filter_Update)
        .def("Finalise", &Filter_Finalise)
        .def("IsTimeDependent", &Filter_IsTimeDependent)

        // Factory functions.
        .def("Create", py::raw_function(Filter_Create))
        .staticmethod("Create")
        .def("Register", [](std::string const &filterName,
                            py::object &obj) { fac(filterName, obj); })
        .staticmethod("Register");

    WrapConverter<Filter>();
}
