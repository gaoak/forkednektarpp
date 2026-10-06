///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasOp.hpp
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
// Description: Strong-form advection with 3/2-rule spectral/hp dealiasing.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/AdvectionDealias/AdvectionDealiasBlockOp.hpp>

namespace Nektar::MultiRegions
{

// AdvectionDealias base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class AdvectionDealiasOp
    : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    static std::shared_ptr<AdvectionDealiasOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<AdvectionDealiasOp, AdvectionDealiasBlockOp>(
                expansionList, components, execStr, implStr);
    }

    static inline const std::string name = "AdvectionDealias";

    void SetScale(const TData &scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetScale(scale);
        }
        v_SetScaleFFT(scale);
    }

    void SetAppend(const bool &append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

    void SetAdvVel(LibUtilities::Field<TData, FieldState::Phys> &advVel)
    {
        // A 3DH1 expansion carries a third velocity component along the
        // homogeneous direction, which the planes themselves do not count.
        ASSERTL1(advVel.GetNumComponents() ==
                     ((advVel.GetNumHomoModes() == 1)
                          ? static_cast<unsigned int>(
                                this->m_expansionList->GetCoordim(0))
                          : 3u),
                 "Advection velocity must have coordDim components, or three "
                 "on a 3DH1 expansion");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAdvVel(advVel.GetBlocks()[blk]);
        }
        m_isSetAdvVel = true;

        v_SetAdvVelFFT(advVel);
    }

protected:
    bool m_isSetAdvVel = false;
    std::vector<std::shared_ptr<AdvectionDealiasBlockOp<TData>>> m_blockOp;

    AdvectionDealiasOp(const MultiRegions::ExpListSharedPtr &expansionList,
                       const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~AdvectionDealiasOp() override = default;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        ASSERTL1(m_isSetAdvVel,
                 "Advection velocity has not been set."
                 "Set the value with SetAdvVel() before calling Apply().");

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, outblock);
        }

        // Apply FFT.
        v_ApplyFFT(in, out);
    }

    /// The advection along the homogeneous direction. The 3/2 rule is
    /// applied in the plane only, so this is w du/dz taken to the fine grid,
    /// multiplied there and projected back on top of the plane part the
    /// block operators have already written.
    virtual void v_ApplyFFT(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out) = 0;

    /// Hand the z pass the velocity and the scale, which the block
    /// operators take for their own part.
    virtual void v_SetAdvVelFFT(
        LibUtilities::Field<TData, FieldState::Phys> &advVel) = 0;
    virtual void v_SetScaleFFT(const TData &scale)            = 0;
};

} // namespace Nektar::MultiRegions
