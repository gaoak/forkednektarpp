///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorConjGrad.hpp
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

// ConjGrad base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorConjGrad : public Operator<TData>
{
public:
    ~OperatorConjGrad() override = default;

    OperatorConjGrad(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Coeff> &in,
                       Field<TData, FieldState::Coeff> &out) = 0;

    virtual void operator()(Field<TData, FieldState::Coeff> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        apply(in, out);
    }

    void setLHS(const std::shared_ptr<OperatorLinear<TData, FieldState::Coeff,
                                                     FieldState::Coeff>> &ptr)
    {
        m_LHS = ptr;
    }

    void setPrecon(
        const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &ptr)
    {
        m_precon = ptr;
    }

protected:
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>>
        m_LHS;
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>>
        m_precon;
};

// Descriptor / traits class for ConjGrad to be used by Operator create function
template <typename TData = default_fp_type> struct ConjGrad
{
    using class_name = OperatorConjGrad<TData>;
    OPERATORS_EXPORT static const std::string key;
    OPERATORS_EXPORT static const std::string default_impl;

    ConjGrad() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<ConjGrad<TData>>(expansionList,
                                                                 pKey);
    }
};

// Avoid [-Wundefined-var-template] warnings
#if defined(__GNUC__) || defined(__clang__)
template <> const std::string ConjGrad<default_fp_type>::key;
template <> const std::string ConjGrad<default_fp_type>::default_impl;
#endif

namespace detail
{
// Template for implementation of CG operator
template <typename TData, typename Op> class OperatorConjGradImpl;
} // namespace detail

} // namespace Nektar::Operators
