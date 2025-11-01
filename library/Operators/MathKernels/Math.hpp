///////////////////////////////////////////////////////////////////////////////
//
// File: Math.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Common/OperatorsDeclspec.hpp"

#include <string>

namespace Nektar::Operators
{

class Math
{
public:
    Math(void) : m_defaultExecSpace("Serial")
    {
    }

    Math(const std::string &defaultExecSpace)
        : m_defaultExecSpace(defaultExecSpace)
    {
    }

    Math &operator=(const Math &rhs)
    {
        m_defaultExecSpace = rhs.m_defaultExecSpace;
        return *this;
    }

    template <typename T>
    OPERATORS_EXPORT void abs(T &x, T &y, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT void neg(T &x, T &y, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT void sqrt(T &x, T &y, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT void add(T &x, T &y, T &z,
                              const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT void sub(T &x, T &y, T &z,
                              const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT void mul(const typename T::value_type alpha, T &x, T &y,
                              const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT void mul(T &x, T &y, T &z,
                              const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT void div(const typename T::value_type alpha, T &x, T &y,
                              const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT void div(T &x, T &y, T &z,
                              const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT void daxpy(const typename T::value_type alpha, T &x, T &y,
                                T &z, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT typename T::value_type reduceSum(
        T &x, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT typename T::value_type reduceMax(
        T &x, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT typename T::value_type reduceMin(
        T &x, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT typename T::value_type ddot(
        T &x, T &y, const std::string &execSpace = "");

    template <typename M, typename T>
    OPERATORS_EXPORT typename T::value_type ddot(
        M &mask, T &x, T &y, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT typename T::value_type l1norm(
        T &x, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT typename T::value_type l2norm(
        T &, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT typename T::value_type lpnorm(
        const unsigned int p, T &x, const std::string &execSpace = "");

    template <typename T>
    OPERATORS_EXPORT typename T::value_type linfnorm(
        T &x, const std::string &execSpace = "");

private:
    std::string m_defaultExecSpace;
};

} // namespace Nektar::Operators
