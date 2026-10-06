///////////////////////////////////////////////////////////////////////////////
//
// File: PhysGalerkinProject1DScaledOp.hpp
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
// Description: Physical space (1D tensor based) Galerkin projection from a
// scaled (finer) quadrature grid back down to the native quadrature grid.
// This is the reverse operator to PhysInterp1DScaled and is used to
// complete the anti-aliasing filter of 3/2-rule spectral/hp dealiasing.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/PhysGalerkinProject1DScaled/PhysGalerkinProject1DScaledBlockOp.hpp>

namespace Nektar::MultiRegions
{

// PhysGalerkinProject1DScaled base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class PhysGalerkinProject1DScaledOp
    : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    static std::shared_ptr<PhysGalerkinProject1DScaledOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<PhysGalerkinProject1DScaledOp,
                            PhysGalerkinProject1DScaledBlockOp>(
                expansionList, components, execStr, implStr);
    }

    static inline const std::string name = "PhysGalerkinProject1DScaled";

    // scale is the same over-integration factor used to build the source
    // (finer) grid via PhysInterp1DScaled, e.g. 1.5 for the 3/2 rule.
    void SetScaleFactor(TData scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetScaleFactor(scale);
        }
    }

    void SetAppend(const bool &append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

protected:
    std::vector<std::shared_ptr<PhysGalerkinProject1DScaledBlockOp<TData>>>
        m_blockOp;

    PhysGalerkinProject1DScaledOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~PhysGalerkinProject1DScaledOp() override = default;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, outblock);
        }
    }
};

} // namespace Nektar::MultiRegions
