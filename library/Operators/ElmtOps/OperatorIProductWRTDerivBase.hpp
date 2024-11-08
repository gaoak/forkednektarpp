///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorIProductWRTDerivBase.hpp
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

#include "Operators/Common/Operator.hpp"

namespace Nektar::Operators
{

// IProductWRTDerivBase base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorIProductWRTDerivBase : public Operator<TData>
{
public:
    OperatorIProductWRTDerivBase(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    ~OperatorIProductWRTDerivBase() override = default;

    virtual void apply(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Coeff> &out,
                       bool APPEND = false) = 0;

    virtual void operator()(Field<TData, FieldState::Phys> &in,
                            Field<TData, FieldState::Coeff> &out,
                            [[maybe_unused]] bool APPEND = false)
    {
        apply(in, out);
    }
};

// Descriptor / traits class for IProductWRTDerivBase
template <typename TData = default_fp_type> struct IProductWRTDerivBase
{
    using class_name = OperatorIProductWRTDerivBase<TData>;

    using FieldIn  = Field<TData, FieldState::Phys>;
    using FieldOut = Field<TData, FieldState::Coeff>;

    IProductWRTDerivBase() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return Operator<TData>::template Create<IProductWRTDerivBase<TData>,
                                                ExecSpace, Impl>(expansionList);
    }
};

} // namespace Nektar::Operators
