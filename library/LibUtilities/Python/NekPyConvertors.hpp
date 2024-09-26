///////////////////////////////////////////////////////////////////////////////
//
// File: NekPyConvertors.hpp
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
// Description: Helper functions to convert STL containers to native python
// types
//
///////////////////////////////////////////////////////////////////////////////
#ifndef NEKTAR_LIBRARY_LIBUTILITIES_PYTHON_NEKPYCONVERTORS_HPP
#define NEKTAR_LIBRARY_LIBUTILITIES_PYTHON_NEKPYCONVERTORS_HPP

#include <boost/python.hpp>
#include <map>
#include <vector>

namespace py = boost::python;

/**
 * @brief Converts a std::map to a Python dict
 *
 * @param input a std::map
 * @returns a Python dict with the same entries as the input map
 */
template <typename KeyT, typename ValT>
inline py::dict MapToPyDict(const std::map<KeyT, ValT> &input)
{
    py::dict ret;
    for (auto &entry : input)
    {
        ret[entry.first] = entry.second;
    }
    return ret;
}

/**
 * @brief Converts a std::vector to a Python list
 *
 * @param input a std::vector
 * @returns a Python list with the same elements as the input vector
 */
template <typename T>
inline py::list VectorToPyList(const std::vector<T> &input)
{
    py::list ret;
    for (auto &entry : input)
    {
        ret.append(entry);
    }
    return ret;
}

#endif // NEKTAR_LIBRARY_LIBUTILITIES_PYTHON_NEKPYCONVERTORS_HPP