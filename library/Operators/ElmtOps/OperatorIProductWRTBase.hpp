///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorIProductWRTBase.hpp
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

// IProductWRTBase base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class OperatorIProductWRTBase : public Operator<TData>
{
public:
    OperatorIProductWRTBase(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    ~OperatorIProductWRTBase() override = default;

    virtual void apply(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Coeff> &out) = 0;

    virtual void operator()(Field<TData, FieldState::Phys> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        apply(in, out);
    }

    void SetScale(TData scale)
    {
        m_scale = scale;
    }

protected:
    TData m_scale = 1.0;
};

// Descriptor / traits class for IProductWRTBase
template <typename TData> struct IProductWRTBase
{
    using class_name = OperatorIProductWRTBase<TData>;

    IProductWRTBase() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return Operator<TData>::template Create<IProductWRTBase<TData>,
                                                ExecSpace, Impl>(expansionList);
    }
};

} // namespace Nektar::Operators
