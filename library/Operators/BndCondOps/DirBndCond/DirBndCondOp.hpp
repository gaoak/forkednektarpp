///////////////////////////////////////////////////////////////////////////////
//
// File: DirBndCondOp.hpp
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

// Dirichlet boundary condition operator base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class DirBndCondOp : public Operator<TData>
{

public:
    static std::shared_ptr<DirBndCondOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operator<TData>::template Create<DirBndCondOp>(
            expansionList, components, execStr);
    }

    static inline const std::string name = "DirBndCond";

    void Apply(Field<TData, FieldState::Coeff> &out)
    {
        v_Apply(out);
    }

    void operator()(Field<TData, FieldState::Coeff> &out)
    {
        v_Apply(out);
    }

    void UpdateBndCoeffs(const TData &time = 0.0)
    {
        if (m_bndCoeffsInitialised && !m_hasTimeDependentBndCoeffs)
        {
            return;
        }
        v_UpdateBndCoeffs(time);
        m_bndCoeffsInitialised = true;
    }

    // Number of boundary coefficients (summed over all components and
    // execution blocks) with a Dirichlet condition on this rank, i.e. the
    // number of dofs actually written to by Apply(). Mainly intended for
    // profiling/introspection.
    size_t GetNumBndDofs() const
    {
        return v_GetNumBndDofs();
    }

protected:
    bool m_bndCoeffsInitialised      = false;
    bool m_hasTimeDependentBndCoeffs = false;

    DirBndCondOp(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    ~DirBndCondOp() override = default;

    virtual void v_Apply(Field<TData, FieldState::Coeff> &out) = 0;

    virtual void v_UpdateBndCoeffs(const TData &time) = 0;

    virtual size_t v_GetNumBndDofs() const = 0;
};

} // namespace Nektar::Operators
