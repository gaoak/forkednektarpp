///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondSlipWallCFEOpImpl.hpp
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

#include <MultiRegions/DisContField.h>

#include "BndCondSlipWallCFEKernels.hpp"
#include "BndCondSlipWallCFEOp.hpp"

namespace Nektar::detail
{

/// Tags this operator claims: the two name the same condition.
inline bool IsSlipWallTag(const std::string &tag)
{
    return tag == "Wall" || tag == "Symmetry";
}

template <typename ExecSpace, typename TData>
class BndCondSlipWallCFEOpImpl : public BndCondSlipWallCFEOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BndCondSlipWallCFEOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : BndCondSlipWallCFEOp<TData>(expansionList, components)
    {
        m_numComp  = components.size();
        m_coordDim = MultiRegions::GetCollection(expansionList, 0)
                         .GetExpVector()[0]
                         ->GetCoordim();

        ASSERTL0(m_numComp >= m_coordDim + 2,
                 "A compressible slip wall needs density, one momentum "
                 "component per direction and energy.");
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<BndCondSlipWallCFEOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_numComp  = 0;
    unsigned int m_coordDim = 0;

    /// The mirrored state *is* the exterior state, so the diffusion must not
    /// reflect it a second time.
    bool v_GhostIsReflected() const override
    {
        return true;
    }

    std::vector<bool> v_GetOwnedBlocks() const override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const auto numBlocks = this->m_bndCondPhysOp->GetNumBlocks();
        std::vector<bool> owned(numBlocks, false);

        for (unsigned blk = 0; blk < numBlocks; ++blk)
        {
            owned[blk] =
                IsSlipWallTag(this->m_bndCondPhysOp->GetBndUserDefined(blk, 0));
        }

        return owned;
    }

    /**
     * @brief Nothing to keep: this condition prescribes no value of its own.
     *
     * The wall is imposed by reflecting the interior state the caller seeds,
     * so there is nothing the session evaluated for it to copy.
     */
    void v_SetBndCondPhysOp() override
    {
    }

    void v_Apply() override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const auto owned = v_GetOwnedBlocks();

        for (unsigned blk = 0; blk < owned.size(); ++blk)
        {
            if (!owned[blk])
            {
                continue;
            }

            const auto stride  = this->m_bndCondPhysOp->GetBlockCompStride(blk);
            auto bndPtr        = this->m_bndCondPhysOp->UpdateBlockPtr(blk);
            const TData *norms = this->m_bndCondPhysOp->GetBlockNormals(blk);

            MirrorNormalMomentumBlock<ExecSpace>(stride, m_coordDim, norms,
                                                 bndPtr);
        }
    }
};

} // namespace Nektar::detail
