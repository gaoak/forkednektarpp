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

#include <boost/version.hpp>
#include <memory>

// Boost 1.62 and earlier don't have native support for std::shared_ptr. This
// includes various patches that are pulled from the git changeset 97e4b34a15
// from the main boost.python github repository, which is where fixes were
// added to include std::shared_ptr support.
#if BOOST_VERSION < 106300
#include "ShPtrFixes.hpp"
#endif

#include "FunctorSignature.hpp"

#ifdef BOOST_HAS_NUMPY

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

#else

#include <boost/numpy.hpp>
#include <boost/python.hpp>

namespace py = boost::python;
namespace np = boost::numpy;

#endif

#define SIZENAME(s) SIZE_##s
#define NEKPY_WRAP_ENUM(ENUMNAME, MAPNAME)                                     \
    {                                                                          \
        py::enum_<ENUMNAME> tmp(#ENUMNAME);                                    \
        for (int a = 0; a < (int)SIZENAME(ENUMNAME); ++a)                      \
        {                                                                      \
            tmp.value(MAPNAME[a], (ENUMNAME)a);                                \
        }                                                                      \
        tmp.export_values();                                                   \
    }
#define NEKPY_WRAP_ENUM_STRING(ENUMNAME, MAPNAME)                              \
    {                                                                          \
        py::enum_<ENUMNAME> tmp(#ENUMNAME);                                    \
        for (int a = 0; a < (int)SIZENAME(ENUMNAME); ++a)                      \
        {                                                                      \
            tmp.value(MAPNAME[a].c_str(), (ENUMNAME)a);                        \
        }                                                                      \
        tmp.export_values();                                                   \
    }
#if PY_MAJOR_VERSION == 2
#define NEKPY_WRAP_ENUM_STRING_DOCS(ENUMNAME, MAPNAME, DOCSTRING)              \
    {                                                                          \
        py::enum_<ENUMNAME> tmp(#ENUMNAME);                                    \
        for (int a = 0; a < (int)SIZENAME(ENUMNAME); ++a)                      \
        {                                                                      \
            tmp.value(MAPNAME[a].c_str(), (ENUMNAME)a);                        \
        }                                                                      \
        tmp.export_values();                                                   \
        PyTypeObject *pto = reinterpret_cast<PyTypeObject *>(tmp.ptr());       \
        PyDict_SetItemString(pto->tp_dict, "__doc__",                          \
                             PyString_FromString(DOCSTRING));                  \
    }
#else
#define NEKPY_WRAP_ENUM_STRING_DOCS(ENUMNAME, MAPNAME, DOCSTRING)              \
    {                                                                          \
        py::enum_<ENUMNAME> tmp(#ENUMNAME);                                    \
        for (int a = 0; a < (int)SIZENAME(ENUMNAME); ++a)                      \
        {                                                                      \
            tmp.value(MAPNAME[a].c_str(), (ENUMNAME)a);                        \
        }                                                                      \
        tmp.export_values();                                                   \
        PyTypeObject *pto = reinterpret_cast<PyTypeObject *>(tmp.ptr());       \
        PyDict_SetItemString(pto->tp_dict, "__doc__",                          \
                             PyUnicode_FromString(DOCSTRING));                 \
    }
#endif

#if BOOST_VERSION >= 106300

namespace boost::python::converter
{

/** @brief shared_ptr_from_python specialization for std:shared_ptr
 *
 * @details
 * The standard boost::python::shared_ptr_from_python have some special trick
 * trying to preserve the python id(obj) for some simple use-patterns.
 *
 * This is done using a shared_ptr<void> with customer deleter with reference to
 * the python object, allowing the same python object to be revealed to the
 * python user.
 *
 * Unfortunately, that implementation breaks the weak-pointer mechanism of
 * shared-ptr, invalidating any reasonable use inside the the c++ library.
 *
 * Thus, we choose to undo this approach, using standard robust c++ mechanisms,
 * and sacrifice the 'id(obj)' requirement.
 *
 * Notice that the implementation is a copy of the original, with the exception
 * of the construct method.
 */
template <class T> struct shared_ptr_from_python<T, std::shared_ptr>
{

    shared_ptr_from_python()
    {
        converter::registry::insert(
            &convertible, &construct, type_id<std::shared_ptr<T>>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                          ,
            &converter::expected_from_python_type_direct<T>::get_pytype
#endif
        );
    }

private:
    static void *convertible(PyObject *p)
    {
        if (p == Py_None)
        {
            return p;
        }
        return converter::get_lvalue_from_python(p, registered<T>::converters);
    }

    static void construct(PyObject *source,
                          rvalue_from_python_stage1_data *data)
    {
        void *const storage =
            ((converter::rvalue_from_python_storage<std::shared_ptr<T>> *)data)
                ->storage.bytes;
        if (data->convertible == source)
        {
            new (storage) std::shared_ptr<T>();
        }
        else
        {
            reference_arg_from_python<std::shared_ptr<T> &> sp(source);
            if (sp.convertible())
            {
                new (storage) std::shared_ptr<T>(sp());
            }
            else
            {
                std::shared_ptr<void> hold_convertible_ref_count(
                    (void *)nullptr,
                    shared_ptr_deleter(handle<>(borrowed(source))));
                new (storage)
                    std::shared_ptr<T>(hold_convertible_ref_count,
                                       static_cast<T *>(data->convertible));
            }
        }
        data->convertible = storage;
    }
};

} // namespace boost::python::converter

#endif

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
            std::string tmp = py::extract<std::string>(py_argv[i]);
            bufSize += tmp.size() + 1;
        }

        m_buf.resize(bufSize);
        for (i = 0, p = &m_buf[0]; i < m_argc; ++i)
        {
            std::string tmp = py::extract<std::string>(py_argv[i]);
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

/**
 * @brief A helper class that for factory-based classes allows
 * std::shared_ptr<T> as something that boost::python recognises, otherwise
 * modules constructed from the factory will not work from Python.
 */
template <typename T> struct WrapConverter
{
    WrapConverter()
    {
        py::objects::class_value_wrapper<
            std::shared_ptr<T>,
            py::objects::make_ptr_instance<
                T, py::objects::pointer_holder<std::shared_ptr<T>, T>>>();
    }
};

#endif
