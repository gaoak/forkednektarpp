///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseOp.hpp
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

#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp>

namespace Nektar::MultiRegions
{

// IProductWRTDerivBase base class
// Defines the apply operator to enforce apply parameter types
template <FieldState TFieldOut, typename TData>
class IProductWRTDerivBaseOp : public ElmtOp<FieldState::Phys, TFieldOut, TData>
{
    template <typename TDataOp>
    using TIProductWRTDerivBaseOp = IProductWRTDerivBaseOp<TFieldOut, TDataOp>;
    template <typename TDataOp>
    using TIProductWRTDerivBaseBlockOp =
        IProductWRTDerivBaseBlockOp<TFieldOut, TDataOp>;
    friend class ElmtOp<FieldState::Phys, TFieldOut, TData>;

public:
    static std::shared_ptr<IProductWRTDerivBaseOp<TFieldOut, TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        return ElmtOp<FieldState::Phys, TFieldOut, TData>::template Create<
            TIProductWRTDerivBaseOp, TIProductWRTDerivBaseBlockOp>(
            expansionList, components, execStr, implStr);
    }

    static inline const std::string name =
        "IProductWRTDerivBase" + FieldStateToString<TFieldOut>();

    void SetAppend(bool append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

    void SetScale(const TData &scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetScale(scale);
        }
        v_SetScaleFFT(scale);
    }

protected:
    std::vector<std::shared_ptr<IProductWRTDerivBaseBlockOp<TFieldOut, TData>>>
        m_blockOp;

    IProductWRTDerivBaseOp(const MultiRegions::ExpListSharedPtr &expansionList,
                           const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, TFieldOut, TData>(expansionList, components)
    {
    }

    ~IProductWRTDerivBaseOp() override = default;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, TFieldOut> &out) override
    {
        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        // The z direction's weak derivative is the plane inner product
        // weighted by the wavenumber, which lands in coefficient space, so
        // only the Coeff variant carries it. The guard is L0: without it a
        // homogeneous field would come back silently wrong in a release
        // build.
        if constexpr (TFieldOut == FieldState::Phys)
        {
            ASSERTL0(in.GetNumHomoModes() == 1,
                     "IProductWRTDerivBaseOp does not support homogeneous "
                     "(3DH1/3DH2) configurations with a Phys output; use the "
                     "Coeff variant.");
        }

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

    virtual void v_SetScaleFFT(const TData &scale) = 0;

    virtual void v_ApplyFFT(LibUtilities::Field<TData, FieldState::Phys> &in,
                            LibUtilities::Field<TData, TFieldOut> &out) = 0;
};

} // namespace Nektar::MultiRegions
