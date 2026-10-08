///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondExtrapOrder0CFEOpImpl.hpp
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

#include "BndCondExtrapOrder0CFEOp.hpp"

namespace Nektar::detail
{

/// Tag this operator claims.
inline bool IsExtrapOrder0Tag(const std::string &tag)
{
    return tag == "ExtrapOrder0";
}

template <typename ExecSpace, typename TData>
class BndCondExtrapOrder0CFEOpImpl : public BndCondExtrapOrder0CFEOp<TData>
{
public:
    BndCondExtrapOrder0CFEOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : BndCondExtrapOrder0CFEOp<TData>(expansionList, components)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<BndCondExtrapOrder0CFEOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    std::vector<bool> v_GetOwnedBlocks() const override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const auto numBlocks = this->m_bndCondPhysOp->GetNumBlocks();
        std::vector<bool> owned(numBlocks, false);

        for (unsigned blk = 0; blk < numBlocks; ++blk)
        {
            owned[blk] = IsExtrapOrder0Tag(
                this->m_bndCondPhysOp->GetBndUserDefined(blk, 0));
        }

        return owned;
    }

    /**
     * @brief Nothing to keep: this condition prescribes no value of its own.
     *
     * Everything it imposes comes from the interior state the caller seeds,
     * so there is nothing the session evaluated for it to copy.
     */
    void v_SetBndCondPhysOp() override
    {
    }

    /**
     * @brief Nothing, deliberately.
     *
     * The caller seeds every owned block with the interior state, which is
     * precisely the zeroth order extrapolation. Claiming the region is the
     * whole of the work; the empty body is the condition, not an omission.
     */
    void v_Apply() override
    {
    }

    /**
     * @brief Either answer gives the same exterior state here, so the default
     * stands.
     *
     * Reflective would leave the ghost at Q+; imposed reflects it to
     * 2g - Q+ with g = Q+, which is also Q+. The distinction that matters
     * everywhere else collapses when the imposed value *is* the interior
     * state, so there is nothing to declare.
     */
};

} // namespace Nektar::detail
