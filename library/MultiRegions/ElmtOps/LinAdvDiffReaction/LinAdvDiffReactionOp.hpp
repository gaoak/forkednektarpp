///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionOp.hpp
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

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionBlockOp.hpp>

namespace Nektar::MultiRegions
{

// LinAdvDiffReaction base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class LinAdvDiffReactionOp
    : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
    friend class ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>;

public:
    static std::shared_ptr<LinAdvDiffReactionOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<LinAdvDiffReactionOp, LinAdvDiffReactionBlockOp>(
                expansionList, components, execStr, implStr);
    }

    static inline const std::string name = "LinAdvDiffReaction";

    void SetLambda(TData lambda)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetLambda(lambda);
        }
        m_isSetLambda = true;
    }

    void SetDiffCoeff(std::vector<TData> &diffCoeff)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetDiffCoeff(diffCoeff);
        }
        m_isSetDiffCoeff = true;
    }

    /// \param advVel  One velocity per coordinate direction of a block, on
    ///                every plane of the field. On a 3DH1 expansion the
    ///                plane block operators take the first two, each plane
    ///                its own, and the third is the through-plane velocity
    ///                the z pass advects with.
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
    bool m_isSetLambda       = false;
    bool m_isSetDiffCoeff    = false;
    bool m_isSetVarDiffCoeff = false;
    bool m_isSetAdvVel       = false;
    std::vector<std::shared_ptr<LinAdvDiffReactionBlockOp<TData>>> m_blockOp;

    LinAdvDiffReactionOp(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList,
                                                              components)
    {
    }

    ~LinAdvDiffReactionOp() override = default;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        ASSERTL1(m_isSetLambda,
                 "lambda has not been set."
                 "Set the value with SetLambda() before calling Apply().");

        ASSERTL1(m_isSetDiffCoeff || m_isSetVarDiffCoeff,
                 "diffusion coefficient has not been set."
                 "Set the value with SetDiffCoeff() OR SetVarDiffCoeff() "
                 "before calling Apply().");

        ASSERTL1(m_isSetDiffCoeff != m_isSetVarDiffCoeff,
                 "can't set both DiffCoeff and VarDiffCoeff."
                 "Set the value with SetDiffCoeff() OR SetVarDiffCoeff() "
                 "before calling Apply().");

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

    /// The z terms of a multi-plane 3DH1 field: the weak z-Laplacian, and
    /// the advection along the homogeneous direction when SetAdvVel() was
    /// given a through-plane velocity.
    virtual void v_ApplyFFT(
        LibUtilities::Field<TData, FieldState::Coeff> &in,
        LibUtilities::Field<TData, FieldState::Coeff> &out) = 0;

    /// Hand the z pass the through-plane velocity, if there is one. The
    /// plane block operators have already taken the in-plane components.
    virtual void v_SetAdvVelFFT(
        LibUtilities::Field<TData, FieldState::Phys> &advVel) = 0;
};

} // namespace Nektar::MultiRegions
