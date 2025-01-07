///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorPhysInterp1DScaled.hpp
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

#include "Operators/ElmtOps/OperatorElmt.hpp"

namespace Nektar::Operators
{

// PhysInterp1DScaled base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorPhysInterp1DScaled
    : public OperatorElmt<FieldState::Phys, FieldState::Phys, TData>
{
public:
    OperatorPhysInterp1DScaled(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorElmt<FieldState::Phys, FieldState::Phys, TData>(expansionList)
    {
    }

    ~OperatorPhysInterp1DScaled() override = default;

    virtual void operator()(Field<TData, FieldState::Phys> &in,
                            Field<TData, FieldState::Phys> &out)
    {
        ASSERTL1(m_scale != -1.0, "Scale factor has not been initialised");
        this->apply(in, out);
    }

    virtual void SetScaleFactor(double scale)
    {
        m_scale = scale;
    }

protected:
    TData m_scale = -1.0; // scaling factor
};

// Descriptor / traits class for PhysInterp1DScaled
template <typename TData> struct PhysInterp1DScaled
{
    using class_name = OperatorPhysInterp1DScaled<TData>;

    PhysInterp1DScaled() = delete;

    template <typename ExecSpace, typename Impl>
    static std::shared_ptr<class_name> Create(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return Operator<TData>::template Create<PhysInterp1DScaled<TData>,
                                                ExecSpace, Impl>(expansionList);
    }
};

} // namespace Nektar::Operators
