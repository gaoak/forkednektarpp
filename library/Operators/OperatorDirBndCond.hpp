///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorDirBndCond.hpp
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

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

// Dirichlet boundary condition operator base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorDirBndCond : public Operator<TData>
{

public:
    virtual ~OperatorDirBndCond() = default;

    OperatorDirBndCond(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &out) = 0;

    virtual void operator()(Field<TData, FieldState::Coeff> &out)
    {
        apply(out);
    }
};

// Descriptor / traits class for DirBndCond to be used by Operator create
// function
template <typename TData = default_fp_type> struct DirBndCond
{
    using class_name = OperatorDirBndCond<TData>;
    OPERATORS_EXPORT static const std::string key;
    OPERATORS_EXPORT static const std::string default_impl;

    DirBndCond() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<DirBndCond<TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// Template for implementation of DirBndCond operator
template <typename TData, typename Op> class OperatorDirBndCondImpl;
} // namespace detail

} // namespace Nektar::Operators
