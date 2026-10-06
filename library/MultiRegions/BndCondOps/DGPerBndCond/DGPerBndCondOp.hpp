///////////////////////////////////////////////////////////////////////////////
//
// File: DGPerBndCondOp.hpp
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

#include <MultiRegions/Common/Operator.hpp>

namespace Nektar::MultiRegions
{

// Periodic boundary condition operator base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class DGPerBndCondOp : public Operator<TData>
{

public:
    static std::shared_ptr<DGPerBndCondOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operator<TData>::template Create<DGPerBndCondOp>(
            expansionList, components, execStr);
    }

    static inline const std::string name = "DGPerBndCond";

    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_Apply(in, out);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_Apply(in, out);
    }

protected:
    DGPerBndCondOp(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    ~DGPerBndCondOp() override = default;

    virtual void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                         LibUtilities::Field<TData, FieldState::Phys> &out) = 0;
};

} // namespace Nektar::MultiRegions
