///////////////////////////////////////////////////////////////////////////////
//
// File: DiagPreconOpImpl.hpp
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

#include <MultiRegions/ContField.h>

#include "Operators/PreconOps/DiagPrecon/DiagPreconOp.hpp"

#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondOp.hpp"
#include "Operators/MathKernels/MathKernels.hpp"

#include "Operators/PreconOps/DiagPrecon/DiagPreconKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class DiagPreconOpImpl : public DiagPreconOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DiagPreconOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : DiagPreconOp<TData>(expansionList)
    {
        m_assmbScatrNoSignOp =
            std::make_unique<AssmbScatrNoSignOpImpl<ExecSpace, TData>>(
                this->m_expansionList);
        m_robBCOp =
            RobBndCondOp<TData>::Create(this->m_expansionList, ExecSpace::name);
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<DiagPreconOpImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    std::unique_ptr<AssmbScatrNoSignOpImpl<ExecSpace, TData>>
        m_assmbScatrNoSignOp;
    std::shared_ptr<RobBndCondOp<TData>> m_robBCOp;

    Field<TData, FieldState::Coeff> m_invDiag;
    MemoryRegion<TData> m_glodiag;
    MemoryRegion<TData> m_wk;

    size_t m_nGlobal;
    size_t m_nDir;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.size() == out.size(),
                 "Input and output arrays are of different size");

        for (size_t blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            auto &inblock   = in.GetBlocks()[blk];
            auto &outblock  = out.GetBlocks()[blk];
            auto &diagblock = m_invDiag.GetBlocks()[blk];

            auto inPtr   = inblock.template GetPtr<MemSpace, ReadOnly>();
            auto outPtr  = outblock.template GetPtr<MemSpace, WriteOnly>();
            auto diagPtr = diagblock.template GetPtr<MemSpace, ReadOnly>();

            auto blkSize =
                outblock.GetNumElementsWithPadding() * outblock.GetNumData();

            for (auto n = 0; n < outblock.GetNumComponents(); ++n)
            {
                mulKernel<ExecSpace>(blkSize, diagPtr, inPtr + n * blkSize,
                                     outPtr + n * blkSize);
            }
        }
    }

    void v_Configure(
        const std::shared_ptr<
            ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> &op) override
    {
        // Create block attributes.
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, this->m_expansionList);

        // Create local diagonal field.
        m_invDiag = Field<TData, FieldState::Coeff>("inverse diagonal", blocks,
                                                    1, 1, ExecSpace::alignment);

        // Create unit vector field to extract diagonal.
        Field<TData, FieldState::Coeff> unit_vec =
            Field<TData, FieldState::Coeff>("DiagPrecon unit vec", blocks, 1, 1,
                                            ExecSpace::alignment);

        // Create action field to receive column action from unit vector.
        Field<TData, FieldState::Coeff> action =
            Field<TData, FieldState::Coeff>("DiagPrecon action", blocks, 1, 1,
                                            ExecSpace::alignment);

        // intiialisating to 1 so padded elements can be inverted
        m_invDiag.template Initialize<NektarSpaces::HostSpace>(1);
        // Initialize field.
        unit_vec.template Initialize<MemSpace>(0);

        unsigned nCoeffMax = 0;
        for (unsigned blk = 0; blk < unit_vec.GetBlocks().size(); ++blk)
        {
            auto &unitvecblock = unit_vec.GetBlocks()[blk];
            const auto nCoeff  = unitvecblock.GetNumData();

            nCoeffMax = (nCoeff > nCoeffMax) ? nCoeff : nCoeffMax;
        }

        for (unsigned mode = 0; mode < nCoeffMax; ++mode)
        {
            for (unsigned blk = 0; blk < unit_vec.GetBlocks().size(); ++blk)
            {
                auto &unitblk     = unit_vec.GetBlocks()[blk];
                const auto nCoeff = unitblk.GetNumData();

                if (mode < nCoeff)
                {
                    const auto numdata   = unitblk.GetNumData();
                    const auto nelmtgrps = unitblk.GetNumElmtGroups();
                    const auto width     = unitblk.GetInterleaveWidth();
                    auto *blkptr =
                        unitblk.template GetPtr<MemSpace, WriteOnly>();

                    // Set ith term in unit vector to be 1.
                    SetModeBlkKernel<ExecSpace>(mode, nelmtgrps, width, numdata,
                                                1.0, blkptr);
                }
            }

            // Apply the operator to unit vector and store in the
            // action field -- ideallly could be a block operator rather than
            // field operator
            op->Apply(unit_vec, action);
            m_robBCOp->Apply(unit_vec, action);

            for (unsigned blk = 0; blk < unit_vec.GetBlocks().size(); ++blk)
            {
                auto &unitblk     = unit_vec.GetBlocks()[blk];
                auto &actionblk   = action.GetBlocks()[blk];
                auto &diagblk     = m_invDiag.GetBlocks()[blk];
                const auto nCoeff = unitblk.GetNumData();

                if (mode < nCoeff)
                {
                    const auto numdata   = unitblk.GetNumData();
                    const auto nelmtgrps = unitblk.GetNumElmtGroups();
                    const auto width     = unitblk.GetInterleaveWidth();
                    auto *unitblkptr =
                        unitblk.template GetPtr<MemSpace, WriteOnly>();
                    auto *fromblkptr =
                        actionblk.template GetPtr<MemSpace, ReadOnly>();
                    auto *toblkptr =
                        diagblk.template GetPtr<MemSpace, WriteOnly>();

                    // Copy the ith row term from the action field to get
                    // the ith diagonal.
                    CopyModeBlkKernel<ExecSpace>(mode, nelmtgrps, width,
                                                 numdata, fromblkptr, toblkptr);

                    // Reset the ith term in the unit vector to be 0.
                    SetModeBlkKernel<ExecSpace>(mode, nelmtgrps, width, numdata,
                                                0.0, unitblkptr);
                }
                diagblk.template SetInterleaveWidth<TData>(
                    actionblk.GetInterleaveWidth());
            }
        }

        // Assembly and scatr  values (without a sign change)
        m_assmbScatrNoSignOp->Apply(m_invDiag, m_invDiag);

        // TODO: Universal assembly on device.
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        if (contfield->GetSession()->GetComm()->GetRowComm()->GetSize() > 1)
        {
            ASSERTL0(false, "Needs setting up");
        }

        // invert diagonal
        for (unsigned blk = 0; blk < m_invDiag.GetBlocks().size(); ++blk)
        {
            // Block dependent.
            auto &block = m_invDiag.GetBlocks()[blk];
            auto diagptr =
                block.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

            // set any zero terms to 1.0 - arises in variable p case.
            // Could set this up as a math kernel operations?
            for (unsigned n = 0; n < block.size(); ++n)
            {
                diagptr[n] = (diagptr[n] == 0.0) ? 1.0 : diagptr[n];
            }

            divKernel<NektarSpaces::Serial>(block.size(), 1.0, diagptr,
                                            diagptr);
        }
    }
};

} // namespace Nektar::Operators::detail
