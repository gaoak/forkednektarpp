///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorIdentity.hpp
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

#include "OperatorLinear.hpp"
#include "Operators/Field.hpp"

namespace Nektar::Operators
{

// Identity base class
// Defines the apply operator to enforce apply parameter types
template <typename TData, FieldState TFieldState>
class OperatorIdentity : public OperatorLinear<TData, TFieldState, TFieldState>
{
public:
    ~OperatorIdentity() override = default;

    OperatorIdentity(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, TFieldState, TFieldState>(expansionList)
    {
    }

    void apply([[maybe_unused]] Field<TData, TFieldState> &in,
               [[maybe_unused]] Field<TData, TFieldState> &out) override
    {
    }

    virtual void operator()(Field<TData, TFieldState> &in,
                            Field<TData, TFieldState> &out)
    {
        apply(in, out);
    }
};

// Descriptor / traits class for Identity
template <FieldState TFieldState, typename TData = default_fp_type>
struct Identity
{
    using class_name = OperatorIdentity<TData, TFieldState>;
    using FieldIn    = Field<TData, TFieldState>;
    using FieldOut   = Field<TData, TFieldState>;
    OPERATORS_EXPORT static const std::string key;
    OPERATORS_EXPORT static const std::string default_impl;

    Identity() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<Identity<TFieldState, TData>>(
            expansionList, pKey);
    }
};

// Avoid [-Wundefined-var-template] warnings
#if defined(__GNUC__) || defined(__clang__)
template <> const std::string Identity<FieldState::Coeff, default_fp_type>::key;
template <>
const std::string Identity<FieldState::Coeff, default_fp_type>::default_impl;
template <> const std::string Identity<FieldState::Phys, default_fp_type>::key;
template <>
const std::string Identity<FieldState::Phys, default_fp_type>::default_impl;
#endif

namespace detail
{
// Template for Identity implementations
template <typename TData, FieldState TFieldState, typename Op>
class OperatorIdentityImpl;
} // namespace detail

} // namespace Nektar::Operators
