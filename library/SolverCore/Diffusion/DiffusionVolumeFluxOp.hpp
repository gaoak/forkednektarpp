///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionVolumeFluxOp.hpp
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
// Description: DiffusionVolume flux operator base classes.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "SolverCore/Flux/FluxOp.hpp"

namespace Nektar::SolverCore
{

template <typename TData> class DiffusionVolumeFluxOp : public FluxOp<TData>
{
public:
    // Build the physical diffusion volume flux from u and grad(u).
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &deriv,
               LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        this->v_Apply(in, deriv, out);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &deriv,
                    LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        this->v_Apply(in, deriv, out);
    }

protected:
    DiffusionVolumeFluxOp(const MultiRegions::ExpListSharedPtr &expansionList,
                          const std::vector<std::string> &components)
        : FluxOp<TData>(expansionList, components)
    {
    }

    ~DiffusionVolumeFluxOp() override = default;

    virtual void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                         LibUtilities::Field<TData, FieldState::Phys> &deriv,
                         LibUtilities::Field<TData, FieldState::Phys> &out) = 0;
};

template <typename TData>
class ScalarIPDiffusionVolumeFluxOp : public DiffusionVolumeFluxOp<TData>
{
public:
    void SetDiffCoeff(std::vector<TData> &diffCoeff)
    {
        this->v_SetDiffCoeff(diffCoeff);
    }

protected:
    ScalarIPDiffusionVolumeFluxOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : DiffusionVolumeFluxOp<TData>(expansionList, components)
    {
    }

    ~ScalarIPDiffusionVolumeFluxOp() override = default;

    virtual void v_SetDiffCoeff(std::vector<TData> &diffCoeff) = 0;
};

} // namespace Nektar::SolverCore
