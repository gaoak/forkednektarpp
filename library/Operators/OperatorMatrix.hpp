///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorMatrix.hpp
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

#include "Common/OperatorLinear.hpp"
#include "Field/Field.hpp"

namespace Nektar::Operators
{

// Matrix operator base class
template <FieldState TFieldState, typename TData>
class OperatorMatrix : public OperatorLinear<TFieldState, TFieldState, TData>
{
public:
    OperatorMatrix(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TFieldState, TFieldState, TData>(expansionList)
    {
    }

    ~OperatorMatrix() override = default;

    virtual size_t size()                     = 0;
    virtual void fill(std::vector<TData> src) = 0;
    virtual std::string toString()            = 0;
};

// Descriptor / traits class for Matrix to be used by Operator create function
template <FieldState TFieldState, typename TData> struct Matrix
{
    using class_name = OperatorMatrix<TFieldState, TData>;

    Matrix() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return Operator<TData>::template create<Matrix<TFieldState, TData>,
                                                ExecSpace, Impl>(expansionList);
    }
};

} // namespace Nektar::Operators
