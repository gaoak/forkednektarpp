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
///   - "" / "Serial" / "Host" -- FFTW-based serial path (always built).
///   - "Device"               -- cuFFT pipeline (requires NEKTAR_ENABLE_CUDA).
///   - "DeviceDx"             -- fused cuFFTDx kernel (additionally
///                               NEKTAR_USE_CUFFTDX).
///
/// v_Apply() makes one unconditional call to m_zOp->Launch(), which owns
/// both the xy block loop and the z-FFT.
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
        auto session = expansionList->GetSession();

        // execStr0: the resolved exec space (from session if not explicit).
        const std::string execStr0 =
            execStr.empty() ? Operator<TData>::GetOpExecSpace(session)
                            : execStr;

        // DeviceDx z-kernels share the same xy block ops as Device.
        const std::string blockExecStr =
            (execStr0 == "DeviceDx") ? std::string("Device") : execStr0;

        const std::string implStr0 =
            implStr.empty()
                ? ElmtOp<FieldState::Phys, FieldState::Phys, TData>::GetOpImpl(
                      PhysDerivOp::name, blockExecStr, session)
                : implStr;

        auto op = std::shared_ptr<PhysDerivOp<TData>>(
            new PhysDerivOp<TData>(expansionList, components));

        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList);
        for (unsigned int blkIdx = 0;
             blkIdx < static_cast<unsigned int>(blockAttr.size()); ++blkIdx)
        {
            const auto expId =
                MultiRegions::GetCollection(expansionList, blkIdx)
                    .GetExpVector()[0]
                    ->GetElmtId();
            const auto exp = expansionList->GetExp(expId);
            op->m_blockOp.push_back(PhysDerivBlockOp<TData>::Create(
                blkIdx, exp, expansionList->GetDataWarehouseSharedPtr(),
                blockExecStr, implStr0));
        }

        // Always create a z-op so v_Apply can call Launch() unconditionally.
        op->m_zOp = PhysDerivZOpBase<TData>::Create(execStr0, expansionList);

        // For 3DH1 problems, configure blockNXY and call Init on the z-op.
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
                ASSERTL1(
                    blockAttr.size() == op->m_blockOp.size(),
                    "Block attribute count does not match block op count.");

                for (std::size_t i = 0; i < blockAttr.size(); ++i)
                {
                    const int NXY = static_cast<int>(
                        blockAttr[i].GetNumElementsWithPadding() *
                        blockAttr[i].GetNumData());
                    op->m_blockNXY.push_back(NXY);
                    op->m_totalNXY += NXY;
                }

                op->m_beta = 2.0 * M_PI / homo->GetHomoLen();
                op->m_zOp->Init(op->m_beta);
            }
        }

        return op;
    }

    static inline const std::string name = "PhysDeriv";

    ~PhysDerivOp() override = default;

protected:
    std::vector<std::shared_ptr<PhysDerivBlockOp<TData>>> m_blockOp;

    TData m_beta = 0.0;
    std::vector<int> m_blockNXY;
    int m_totalNXY = 0;

    std::shared_ptr<PhysDerivZOpBase<TData>> m_zOp;

    PhysDerivOp(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    void v_Apply(MultiRegions::Field<TData, FieldState::Phys> &in,
                 MultiRegions::Field<TData, FieldState::Phys> &out) override
    {
        const unsigned int nhomo = in.GetNumHomoModes();

        // m_blockNXY is non-empty only for 3DH1 (set in Create when
        // dynamic_cast to ExpListHomogeneous1D succeeds). For 2D/3D/3DH2
        // use coordim to determine the output-component ratio.
        ASSERTL1(in.GetNumComponents() ==
                     out.GetNumComponents() /
                         (!m_blockNXY.empty()
                              ? 3u
                              : static_cast<unsigned int>(
                                    this->m_expansionList->GetCoordim(0))),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        // For backward compatilbity for non-CUDA device backend.
        if (nhomo == 1)
        {
            // Loop over the blocks.
            for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
            {
                // Block dependent.
                auto &inblock  = in.GetBlocks()[blk];
                auto &outblock = out.GetBlocks()[blk];

                this->m_blockOp[blk]->Apply(inblock, outblock);
            }
        }
        else
        {
            m_zOp->Launch(m_blockOp, in, out, nhomo, m_blockNXY);
        }
    }
};

} // namespace Nektar::Operators
