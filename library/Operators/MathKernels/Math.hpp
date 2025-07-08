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
#include "Operators/Field/Field.hpp"

#include <string>

namespace Nektar
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

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT void neg(Field<TData, TFieldState> &x,
                              Field<TData, TFieldState> &y,
                              const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT void neg(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
                              const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT void add(Field<TData, TFieldState> &x,
                              Field<TData, TFieldState> &y,
                              Field<TData, TFieldState> &z,
                              const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT void add(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
                              MemoryRegion<TData> &z,
                              const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT void sub(Field<TData, TFieldState> &x,
                              Field<TData, TFieldState> &y,
                              Field<TData, TFieldState> &z,
                              const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT void sub(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
                              MemoryRegion<TData> &z,
                              const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT void mul(const TData alpha, Field<TData, TFieldState> &x,
                              Field<TData, TFieldState> &y,
                              const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT void mul(Field<TData, TFieldState> &x,
                              Field<TData, TFieldState> &y,
                              Field<TData, TFieldState> &z,
                              const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT void mul(const TData alpha, MemoryRegion<TData> &x,
                              MemoryRegion<TData> &y,
                              const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT void mul(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
                              MemoryRegion<TData> &z,
                              const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT void div(const TData alpha, Field<TData, TFieldState> &x,
                              Field<TData, TFieldState> &y,
                              const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT void div(Field<TData, TFieldState> &x,
                              Field<TData, TFieldState> &y,
                              Field<TData, TFieldState> &z,
                              const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT void div(const TData alpha, MemoryRegion<TData> &x,
                              MemoryRegion<TData> &y,
                              const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT void div(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
                              MemoryRegion<TData> &z,
                              const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT void daxpy(const TData alpha, Field<TData, TFieldState> &x,
                                Field<TData, TFieldState> &y,
                                Field<TData, TFieldState> &z,
                                const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT void daxpy(const TData alpha, MemoryRegion<TData> &x,
                                MemoryRegion<TData> &y, MemoryRegion<TData> &z,
                                const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT TData reduceSum(Field<TData, TFieldState> &x,
                                     const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT TData reduceSum(MemoryRegion<TData> &x,
                                     const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT TData reduceMax(Field<TData, TFieldState> &x,
                                     const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT TData reduceMax(MemoryRegion<TData> &x,
                                     const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT TData reduceMin(Field<TData, TFieldState> &x,
                                     const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT TData reduceMin(MemoryRegion<TData> &x,
                                     const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT TData ddot(Field<TData, TFieldState> &x,
                                Field<TData, TFieldState> &y,
                                const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT TData ddot(MemoryRegion<TData> &x, MemoryRegion<TData> &y,
                                const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT TData l1norm(Field<TData, TFieldState> &x,
                                  const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT TData l1norm(MemoryRegion<TData> &x,
                                  const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT TData l2norm(Field<TData, TFieldState> &x,
                                  const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT TData l2norm(MemoryRegion<TData> &,
                                  const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT TData lpnorm(const unsigned int p,
                                  Field<TData, TFieldState> &x,
                                  const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT TData lpnorm(const unsigned int p, MemoryRegion<TData> &x,
                                  const std::string &execSpace = "");

    template <typename TData, FieldState TFieldState>
    OPERATORS_EXPORT TData linfnorm(Field<TData, TFieldState> &x,
                                    const std::string &execSpace = "");

    template <typename TData>
    OPERATORS_EXPORT TData linfnorm(MemoryRegion<TData> &x,
                                    const std::string &execSpace = "");

private:
    std::string m_defaultExecSpace;
};

} // namespace Nektar
