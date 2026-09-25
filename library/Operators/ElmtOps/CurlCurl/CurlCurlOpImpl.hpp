///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlOpImpl.hpp
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

#include <array>

#include <MultiRegions/ExpListHomogeneous1D.h>

#include "Operators/ElmtOps/CurlCurl/CurlCurlOp.hpp"
#include "Operators/ElmtOps/DerivZOpImpl.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class CurlCurlOpImpl : public CurlCurlOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    CurlCurlOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : CurlCurlOp<TData>(expansionList, components)
    {
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);
        // The planes this rank holds, not the homogeneous basis' point count:
        // with npz > 1 the direction is split over the column communicator
        // while the basis still reports the global total.
        const unsigned int nhomo =
            homoExpList
                ? static_cast<unsigned int>(homoExpList->GetZIDs().size())
                : 1u;

        if (nhomo > 1)
        {
            // A z-op per curl rather than one used twice: a captured device
            // graph holds the Field pointers it was built with, and the two
            // curls are given different pairs.
            for (auto &zOp : m_zOp)
            {
                zOp = std::make_shared<
                    DerivZOpImpl<ExecSpace, TData, DerivZLayout::CurlZ,
                                 DerivZOrder::First, true>>(expansionList);
            }

            // A curl needs all three components whichever way round it is
            // taken, so the workspace carries three whatever the variable
            // count is.
            constexpr unsigned int nComp = 3u;

            m_wsp = LibUtilities::Field<TData, FieldState::Phys>(
                "CurlCurlWsp",
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    expansionList),
                nComp, nhomo);
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<CurlCurlOpImpl<ExecSpace, TData>>(expansionList,
                                                                  components);
    }

protected:
    /// The z part of one curl, (-dfy/dz, dfx/dz, 0), added on top of the
    /// plane part the block operators leave in the output. One per curl:
    /// each keeps the transform state of the Field pair it is given.
    std::array<
        std::shared_ptr<DerivZOpImpl<ExecSpace, TData, DerivZLayout::CurlZ,
                                     DerivZOrder::First, true>>,
        2>
        m_zOp;
    /// Holds the second curl while it is being formed. The block operators
    /// cannot write the curl of a Field back into that same Field, so the
    /// second pass needs somewhere of its own and the result is copied into
    /// the output at the end. Uninstantiated on every other expansion.
    LibUtilities::Field<TData, FieldState::Phys> m_wsp;

    void v_ApplyFFT(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        // The double curl of a multi-plane 3DH1 field does not fuse: the z
        // part of the second curl reads the first curl's result, so the pair
        // of plane part and z part runs twice. The operator's own v_Apply has
        // already left the first curl's plane part in the output, which the
        // z-op completes there; the output then holds omega, and the second
        // pair runs out of it.
        //
        // m_zOp stays null for 2D/3D, 3DH2 and single-plane 3DH1, where the
        // block operators take both curls themselves and there is nothing to
        // add here.
        if (m_zOp[0] && in.GetNumHomoModes() > 1)
        {
            // First curl: the z part on top of the plane part already in the
            // output, leaving omega there.
            m_zOp[0]->Launch(in, out);

            // Second curl, out of omega. Loop over the blocks.
            for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
            {
                // Block dependent.
                auto &inblock  = out.GetBlocks()[blk];
                auto &outblock = m_wsp.GetBlocks()[blk];

                this->m_blockOp[blk]->Apply(inblock, outblock);
            }

            // The z part of the second curl, on top of its plane part.
            m_zOp[1]->Launch(out, m_wsp);

            out.template Copy<MemSpace>(m_wsp);
        }
    }
};

} // namespace Nektar::Operators::detail
