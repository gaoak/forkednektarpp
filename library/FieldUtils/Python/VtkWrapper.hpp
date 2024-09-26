///////////////////////////////////////////////////////////////////////////////
//
// File: VtkWrapper.hpp
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
// Description: Helper routines from extracting VTK object pointers.
//
///////////////////////////////////////////////////////////////////////////////

#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVersion.h>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

/*
 * Adapted from the code at:
 * https://www.paraview.org/Wiki/Example_from_and_to_python_converters
 */

template <class T> struct vtkObjectPointer_to_python
{
    static PyObject *convert(const T &p)
    {
        if (p == NULL)
        {
            return py::incref(Py_None);
        }
        std::ostringstream oss;
        oss << (vtkObjectBase *)p; // here don't get address
        std::string address_str = oss.str();

        try
        {
            py::object obj = py::import("vtkmodules.vtkCommonCore")
                                 .attr("vtkObjectBase")(address_str);
            return py::incref(obj.ptr());
        }
        catch (py::error_already_set &)
        {
            // Clear error to avoid potential failure to catch error in
            // next try-catch block.
            PyErr_Clear();
        }

        try
        {
            py::object obj =
                py::import("vtk").attr("vtkObjectBase")(address_str);
            return py::incref(obj.ptr());
        }
        catch (py::error_already_set &)
        {
            using NekError = Nektar::ErrorUtil::NekError;
            throw NekError("Unable to import VTK.");
        }

        return py::incref(py::object().ptr());
    }
};

//
// This python to C++ converter uses the fact that VTK Python objects have an
// attribute called __this__, which is a string containing the memory address
// of the VTK C++ object and its class name.
// E.g. for a vtkPoints object __this__ might be "_0000000105a64420_p_vtkPoints"
//
void *extract_vtk_wrapped_pointer(PyObject *obj)
{
    std::string thisStr = "__this__";

    // first we need to get the __this__ attribute from the Python Object
    if (!PyObject_HasAttrString(obj, thisStr.c_str()))
    {
        return nullptr;
    }

    PyObject *thisAttr = PyObject_GetAttrString(obj, thisStr.c_str());
    if (thisAttr == nullptr)
    {
        return nullptr;
    }

#if PY_MAJOR_VERSION == 2
    std::string str = PyString_AsString(thisAttr);
#else
    std::string str = PyUnicode_AsUTF8(thisAttr);
#endif

    if (str.size() < 21)
    {
        return nullptr;
    }

    char hex_address[32], *pEnd;
    auto iter = str.find("_p_vtk");

    if (iter == std::string::npos)
    {
        return nullptr;
    }

    auto className = str.substr(iter).find("vtk");
    if (className == std::string::npos)
    {
        return nullptr;
    }

    long address             = stol(str.substr(1, 17));
    vtkObjectBase *vtkObject = (vtkObjectBase *)((void *)address);

    if (vtkObject->IsA(str.substr(className).c_str()))
    {
        return vtkObject;
    }

    return nullptr;
}

#define VTK_PYTHON_CONVERSION(type)                                            \
    /* register the to-python converter */                                     \
    py::to_python_converter<type *, vtkObjectPointer_to_python<type *>>();     \
    /* register the from-python converter */                                   \
    py::converter::registry::insert(&extract_vtk_wrapped_pointer,              \
                                    py::type_id<type>());
