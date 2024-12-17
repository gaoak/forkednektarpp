///////////////////////////////////////////////////////////////////////////////
//
// File: NekPyConfig.hpp
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
// Description: NekPy configuration to include boost headers and define
// commonly-used macros.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIBRARY_LIBUTILITIES_PYTHON_NEKPYCONFIG_HPP
#define NEKTAR_LIBRARY_LIBUTILITIES_PYTHON_NEKPYCONFIG_HPP

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

// Define some common STL opaque types
PYBIND11_MAKE_OPAQUE(std::vector<unsigned int>)

#define SIZENAME(s) SIZE_##s
#define NEKPY_WRAP_ENUM(MOD, ENUMNAME, MAPNAME)                                \
    {                                                                          \
        py::enum_<ENUMNAME> tmp(MOD, #ENUMNAME);                               \
        for (int a = 0; a < (int)SIZENAME(ENUMNAME); ++a)                      \
        {                                                                      \
            tmp.value(MAPNAME[a], (ENUMNAME)a);                                \
        }                                                                      \
        tmp.export_values();                                                   \
    }
#define NEKPY_WRAP_ENUM_STRING(MOD, ENUMNAME, MAPNAME)                         \
    {                                                                          \
        py::enum_<ENUMNAME> tmp(MOD, #ENUMNAME);                               \
        for (int a = 0; a < (int)SIZENAME(ENUMNAME); ++a)                      \
        {                                                                      \
            tmp.value(MAPNAME[a].c_str(), (ENUMNAME)a);                        \
        }                                                                      \
        tmp.export_values();                                                   \
    }
#define NEKPY_WRAP_ENUM_STRING_DOCS(MOD, ENUMNAME, MAPNAME, DOCSTRING)         \
    {                                                                          \
        py::enum_<ENUMNAME> tmp(MOD, #ENUMNAME);                               \
        for (int a = 0; a < (int)SIZENAME(ENUMNAME); ++a)                      \
        {                                                                      \
            tmp.value(MAPNAME[a].c_str(), (ENUMNAME)a);                        \
        }                                                                      \
        tmp.export_values();                                                   \
        PyTypeObject *pto = reinterpret_cast<PyTypeObject *>(tmp.ptr());       \
        PyDict_SetItemString(pto->tp_dict, "__doc__",                          \
                             PyUnicode_FromString(DOCSTRING));                 \
    }

/**
 * @brief Helper structure to construct C++ command line `argc` and `argv`
 * variables from a Python list.
 */
struct CppCommandLine
{
    /**
     * @brief Constructor.
     *
     * @param py_argv   List of command line arguments from Python.
     */
    CppCommandLine(py::list &py_argv) : m_argc(py::len(py_argv))
    {
        int i          = 0;
        size_t bufSize = 0;
        char *p;

        m_argv = new char *[m_argc + 1];

        // Create argc, argv to give to the session reader. Note that this needs
        // to be a contiguous block in memory, otherwise MPI (specifically
        // OpenMPI) will likely segfault.
        for (i = 0; i < m_argc; ++i)
        {
            std::string tmp = py::cast<std::string>(py_argv[i]);
            bufSize += tmp.size() + 1;
        }

        m_buf.resize(bufSize);
        for (i = 0, p = &m_buf[0]; i < m_argc; ++i)
        {
            std::string tmp = py::cast<std::string>(py_argv[i]);
            std::copy(tmp.begin(), tmp.end(), p);
            p[tmp.size()] = '\0';
            m_argv[i]     = p;
            p += tmp.size() + 1;
        }

        m_argv[m_argc] = nullptr;
    }

    /**
     * @brief Destructor.
     */
    ~CppCommandLine()
    {
        if (m_argv == nullptr)
        {
            return;
        }

        delete m_argv;
    }

    /**
     * @brief Returns the constructed `argv`.
     */
    char **GetArgv()
    {
        return m_argv;
    }

    /**
     * @brief Returns the constructed `argc`.
     */
    int GetArgc()
    {
        return m_argc;
    }

private:
    /// Pointers for strings `argv`.
    char **m_argv = nullptr;
    /// Number of arguments `argc`.
    int m_argc = 0;
    /// Buffer for storage of the argument strings.
    std::vector<char> m_buf;
};

#endif
