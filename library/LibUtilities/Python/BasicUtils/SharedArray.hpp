///////////////////////////////////////////////////////////////////////////////
//
// File: SharedArray.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Scientific Computing and Imaging Institute,
// University of Utah (USA) and Department of Aeronautics, Imperial
// College London (UK).
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
// Description: Python wrapper for ShareArray.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIBUTILITIES_PYTHON_BASICUTILS_SHAREDARRAY_HPP
#define NEKTAR_LIBUTILITIES_PYTHON_BASICUTILS_SHAREDARRAY_HPP

#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include <LibUtilities/Python/NekPyConfig.hpp>

#include <type_traits>

using namespace Nektar;
using namespace Nektar::LibUtilities;

/// Template type to determine whether @tparam T is a std::shared_ptr.
template <typename T> struct is_shared_ptr : std::false_type
{
};

template <typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type
{
};

template <typename T>
struct is_shared_ptr<const std::shared_ptr<T>> : std::true_type
{
};

/// Template utility to determine whether @tparam T is a Nektar::Array<OneD, .>.
template <typename T> struct is_nekarray_oned : std::false_type
{
};

template <typename T> struct is_nekarray_oned<Array<OneD, T>> : std::true_type
{
};

template <typename T>
struct is_nekarray_oned<const Array<OneD, T>> : std::true_type
{
};

#if PY_MAJOR_VERSION == 2
template <typename T> void CapsuleDestructor(void *ptr)
{
    Array<OneD, T> *tmp = (Array<OneD, T> *)ptr;
    delete tmp;
}
#else
template <typename T> void CapsuleDestructor(PyObject *ptr)
{
    Array<OneD, T> *tmp = (Array<OneD, T> *)PyCapsule_GetPointer(ptr, nullptr);
    delete tmp;
}
#endif

/**
 * @brief Convert for Array<OneD, T> to Python list of objects for numeric types
 * T.
 */
template <typename T> struct OneDArrayToPython
{
    static PyObject *convert(Array<OneD, T> const &arr)
    {
        return convert_impl(arr);
    }

    template <typename U                                       = T,
              std::enable_if_t<std::is_arithmetic<U>::value> * = nullptr>
    static PyObject *convert_impl(Array<OneD, U> const &arr)
    {
        // Create a Python capsule to hold a pointer that contains a lightweight
        // copy of arr. Uhat way we guarantee Python will still have access to
        // the memory allocated inside arr even if arr is deallocated in C++.
#if PY_MAJOR_VERSION == 2
        py::object capsule(py::handle<>(PyCObject_FromVoidPtr(
            new Array<OneD, U>(arr), CapsuleDestructor<U>)));
#else
        py::object capsule(py::handle<>(
            PyCapsule_New(new Array<OneD, U>(arr), nullptr,
                          (PyCapsule_Destructor)&CapsuleDestructor<U>)));
#endif
        PyObject *tmp =
            py::incref(np::from_data(arr.data(), np::dtype::get_builtin<U>(),
                                     py::make_tuple(arr.size()),
                                     py::make_tuple(sizeof(U)), capsule)
                           .ptr());

        return tmp;
    }

    /**
     * @brief Converter function. This copies entries into the Python list and
     * relies on internal boost converter being available for the shared_ptr
     * type.
     */
    template <typename U                                     = T,
              std::enable_if_t<is_shared_ptr<U>::value ||
                               is_nekarray_oned<U>::value> * = nullptr>
    static PyObject *convert_impl(Array<OneD, U> const &arr)
    {
        py::list tmp;
        for (std::size_t i = 0; i < arr.size(); ++i)
        {
            tmp.append(arr[i]);
        }

        // Increment counter to avoid de-allocation of `tmp`.
        return py::incref(tmp.ptr());
    }
};

/**
 * @brief Converter for Python to Nektar::Array<OneD, T>.
 */
template <typename T> struct PythonToOneDArray
{
    /// Default constructor.
    PythonToOneDArray()
    {
        py::converter::registry::push_back(&convertible, &construct,
                                           py::type_id<Array<OneD, T>>());
    }

    /**
     * @brief Determine whether the given @p objPtr is convertible to
     * convertible to an Array type or not.
     *
     * This is a top-level function: different data types are handled separately
     * in the ::try_convertible functions.
     */
    static void *convertible(PyObject *objPtr)
    {
        try
        {
            py::object obj((py::handle<>(py::borrowed(objPtr))));
            return try_convertible(obj) ? objPtr : nullptr;
        }
        catch (boost::python::error_already_set &)
        {
            py::handle_exception();
            PyErr_Clear();
            return nullptr;
        }
    }

    template <typename U                                       = T,
              std::enable_if_t<std::is_arithmetic<U>::value> * = nullptr>
    static bool try_convertible(py::object &obj)
    {
        np::ndarray array = py::extract<np::ndarray>(obj);

        // Check data types match
        np::dtype dtype =
            np::dtype::get_builtin<typename std::remove_const<U>::type>();
        if (dtype != array.get_dtype())
        {
            return false;
        }

        // Check shape is 1D
        if (array.get_nd() != 1)
        {
            return false;
        }

        return true;
    }

    template <typename U                                     = T,
              std::enable_if_t<is_shared_ptr<U>::value ||
                               is_nekarray_oned<U>::value> * = nullptr>
    static bool try_convertible(py::object &obj)
    {
        py::extract<py::list> list_conv(obj);

        if (!list_conv.check())
        {
            return false;
        }

        py::list l               = list_conv();
        const std::size_t nItems = py::len(l);

        // We'll need to construct a temporary vector to hold each item in the
        // list.
        for (std::size_t i = 0; i < nItems; ++i)
        {
            py::extract<T> item_conv(l[i]);

            if (!py::extract<T>(l[i]).check())
            {
                return false;
            }
        }

        return true;
    }

    static void decrement(void *objPtr)
    {
        if (!Py_IsInitialized())
        {
            // In deinitialisation phase, reference counters are not terribly
            // robust; decremementing counters here can lead to segfaults during
            // program exit in some cases.
            return;
        }

        // Otherwise decrement reference counter.
        py::decref((PyObject *)objPtr);
    }

    static void construct(PyObject *objPtr,
                          py::converter::rvalue_from_python_stage1_data *data)
    {
        construct_impl(objPtr, data);
    }

    template <typename U                                       = T,
              std::enable_if_t<std::is_arithmetic<U>::value> * = nullptr>
    static void construct_impl(
        PyObject *objPtr, py::converter::rvalue_from_python_stage1_data *data)
    {
        // This has to be a _borrowed_ reference, otherwise at the end of this
        // scope it seems memory gets deallocated
        py::object obj(py::handle<>(py::borrowed(objPtr)));
        np::ndarray array = py::extract<np::ndarray>(obj);

        // If this array came from C++, extract the C++ array from PyCObject or
        // PyCapsule and ensure that we set up the C++ array to have a reference
        // to that object, so that it can be decremented as appropriate.
        py::object base     = array.get_base();
        Array<OneD, T> *ptr = nullptr;

#if PY_MAJOR_VERSION == 2
        if (PyCObject_Check(base.ptr()))
        {
            ptr = reinterpret_cast<Array<OneD, T> *>(
                PyCObject_AsVoidPtr(base.ptr()));
        }
#else
        if (PyCapsule_CheckExact(base.ptr()))
        {
            ptr = reinterpret_cast<Array<OneD, T> *>(
                PyCapsule_GetPointer(base.ptr(), nullptr));
        }
#endif

        void *storage =
            ((py::converter::rvalue_from_python_storage<Array<OneD, T>> *)data)
                ->storage.bytes;
        data->convertible = storage;

        using nonconst_t = typename std::remove_const<T>::type;
        new (storage)
            Array<OneD, T>(array.shape(0), (nonconst_t *)array.get_data(),
                           (void *)objPtr, &decrement);
        py::incref(objPtr);
    }

    template <typename U                                     = T,
              std::enable_if_t<is_shared_ptr<U>::value ||
                               is_nekarray_oned<U>::value> * = nullptr>
    static void construct_impl(
        PyObject *objPtr, py::converter::rvalue_from_python_stage1_data *data)
    {
        using nonconst_t = typename std::remove_const<T>::type;

        py::object obj(py::handle<>(py::borrowed(objPtr)));
        py::list l = py::extract<py::list>(obj);

        const std::size_t nItems = py::len(l);

        // Allocate some storage for the Array.
        void *storage =
            ((py::converter::rvalue_from_python_storage<Array<OneD, nonconst_t>>
                  *)data)
                ->storage.bytes;
        Array<OneD, nonconst_t> *tmp =
            new (storage) Array<OneD, nonconst_t>(nItems);
        data->convertible = storage;

        // Fill the Array.
        for (std::size_t i = 0; i < nItems; ++i)
        {
            (*tmp)[i] = py::extract<nonconst_t>(l[i]);
        }
    }
};

/**
 * @brief Convenience function to export C++-to-Python and Python-to-C++
 * converters for the requested type @tparam T.
 */
template <typename T> void export_SharedArray()
{
    py::to_python_converter<Array<OneD, T>, OneDArrayToPython<T>>();
    PythonToOneDArray<T>();
}

#endif
