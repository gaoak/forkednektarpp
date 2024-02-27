///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorDiagPrecon.hpp
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

#include "OperatorPrecon.hpp"
#include "Operators/Field.hpp"

namespace Nektar::Operators
{

// DiagPrecon base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorDiagPrecon : public OperatorPrecon<TData>
{
public:
    ~OperatorDiagPrecon() override = default;

    OperatorDiagPrecon(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPrecon<TData>(expansionList)
    {
    }
};

// Descriptor / traits class for DiagPrecon to be used by Operator create
// function
template <typename TData = default_fp_type> struct DiagPrecon
{
    using class_name = OperatorDiagPrecon<TData>;
    OPERATORS_EXPORT static const std::string key;
    OPERATORS_EXPORT static const std::string default_impl;

    DiagPrecon() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<DiagPrecon<TData>>(
            expansionList, pKey);
    }
};

// Avoid [-Wundefined-var-template] warnings
#if defined(__GNUC__) || defined(__clang__)
template <> const std::string DiagPrecon<default_fp_type>::key;
template <> const std::string DiagPrecon<default_fp_type>::default_impl;
#endif

namespace detail
{
// Template for implementation of DiagPrecon operator
template <typename TData, typename Op> class OperatorDiagPreconImpl;
} // namespace detail

} // namespace Nektar::Operators
