///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivOp.hpp
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

#include "Operators/ElmtOps/ElmtOp.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivBlockOp.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivZOp.h"

#include <MultiRegions/ExpListHomogeneous1D.h>

namespace Nektar::Operators
{

/// \brief Element-block operator that computes the physical-space derivative
/// \f$\nabla u\f$ over a collection of elements.
///
/// The backend for the homogeneous z-derivative in 3DH1 configurations is
/// selected at Create() time via the \p execStr argument:
///   - "Serial" / "AVX" -- FFTW-based serial path (always built).
///   - "Device"         -- cuFFT pipeline, or cuFFTDx when
///                         NEKTAR_USE_CUFFTDX. A device build without a
///                         z-FFT of its own falls back to the host path.
/// An empty execStr resolves the space from the session.
template <typename TData>
class PhysDerivOp : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    static std::shared_ptr<PhysDerivOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto op =
            ElmtOp<FieldState::Phys, FieldState::Phys, TData>::template Create<
                PhysDerivOp, PhysDerivBlockOp>(expansionList, components,
                                               execStr, implStr);

        // FFT setup. A z-op is built only for a multi-plane 3DH1 expansion;
        // m_zOp stays null for 2D/3D, 3DH2 and single-plane 3DH1, and
        // v_Apply then does the xy derivatives alone.
        auto homo =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);

        if (homo)
        {
            const unsigned int nhomo =
                static_cast<unsigned int>(expansionList->GetTotPoints() /
                                          homo->GetPlane(0)->GetTotPoints());

            if (nhomo > 1)
            {
                op->m_zOp =
                    PhysDerivZOpBase<TData>::Create(expansionList, execStr);

                op->m_beta = 2.0 * M_PI / homo->GetHomoLen();
                op->m_zOp->Init(op->m_beta);
            }
        }

        return op;
    }

    static inline const std::string name = "PhysDeriv";

protected:
    std::vector<std::shared_ptr<PhysDerivBlockOp<TData>>> m_blockOp;
    std::shared_ptr<PhysDerivZOpBase<TData>> m_zOp;
    TData m_beta = 0.0;

    PhysDerivOp(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~PhysDerivOp() override = default;

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() ==
                     out.GetNumComponents() /
                         (in.GetNumHomoModes() == 1
                              ? static_cast<unsigned int>(
                                    this->m_expansionList->GetCoordim(0))
                              : 3u),
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

        // Apply FFT.
        if (m_zOp && in.GetNumHomoModes() > 1)
        {
            m_zOp->Launch(in, out);
        }
    }
};

} // namespace Nektar::Operators
