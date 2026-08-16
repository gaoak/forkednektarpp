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

#include "Operators/PreconOps/DiagPrecon/DiagPreconOp.hpp"

#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"
#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondOp.hpp"

#include "Operators/PreconOps/DiagPrecon/DiagPreconKernels.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class DiagPreconOpImpl : public DiagPreconOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DiagPreconOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : DiagPreconOp<TData>(expansionList, components)
    {
        m_assmbScatrNoSignOp =
            std::make_unique<AssmbScatrNoSignOpImpl<ExecSpace, TData>>(
                this->m_expansionList, components);
        m_robBCOp = RobBndCondOp<TData>::Create(this->m_expansionList,
                                                components, ExecSpace::name);
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<DiagPreconOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    std::unique_ptr<AssmbScatrNoSignOpImpl<ExecSpace, TData>>
        m_assmbScatrNoSignOp;
    std::shared_ptr<RobBndCondOp<TData>> m_robBCOp;

    LibUtilities::Field<TData, FieldState::Coeff> m_invDiag;

    size_t m_nGlobal;
    size_t m_nDir;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        ASSERTL1(in.size() == out.size(),
                 "Input and output arrays are of different size");

        for (size_t blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &inblock   = in.GetBlocks()[blk];
            auto &outblock  = out.GetBlocks()[blk];
            auto &diagblock = m_invDiag.GetBlocks()[blk];

            auto inPtr = inblock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto outPtr =
                outblock.template GetPtr<MemSpace, WriteOnly>(streamID);
            auto diagPtr =
                diagblock.template GetPtr<MemSpace, ReadOnly>(streamID);

            // Reshape, if necessary.
            auto in_width = inblock.GetInterleaveWidth();
            const unsigned int nComp =
                inblock.GetNumComponents() * inblock.GetNumHomoModes();
            if (in_width != diagblock.GetInterleaveWidth())
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    in_width, diagblock.GetInterleaveWidth(),
                    diagblock.GetNumElementsWithPadding() * nComp,
                    diagblock.GetNumData(), (TData *)diagPtr, streamID);
            }

            // Apply diagonal preconditioner.
            auto blkSize =
                outblock.GetNumElementsWithPadding() * outblock.GetNumData();
            Math::mulKernel<ExecSpace>(blkSize * nComp, diagPtr, inPtr, outPtr,
                                       streamID);

            // Set output block to input interleave.
            outblock.template SetInterleaveWidth<TData>(in_width);
            diagblock.template SetInterleaveWidth<TData>(in_width);
        }
    }

    void v_Configure(
        const std::shared_ptr<
            ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> &op) override
    {
        // Create block attributes.
        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                this->m_expansionList);

        // Create local diagonal field.
        m_invDiag = LibUtilities::Field<TData, FieldState::Coeff>(
            "inverse diagonal", blockAttr, this->m_components, 1);

        // Create unit vector field to extract diagonal.
        LibUtilities::Field<TData, FieldState::Coeff> unit_vec =
            LibUtilities::Field<TData, FieldState::Coeff>(
                "DiagPrecon unit vec", blockAttr, this->m_components, 1);

        // Create action field to receive column action from unit vector.
        LibUtilities::Field<TData, FieldState::Coeff> action =
            LibUtilities::Field<TData, FieldState::Coeff>(
                "DiagPrecon action", blockAttr, this->m_components, 1);

        // Intialisating to 1 so padded elements can be inverted.
        m_invDiag.template Initialize<MemSpace>(1);

        // Compute maximum block nCoeff.
        unsigned nCoeffMax = 0;
        for (unsigned blk = 0; blk < unit_vec.GetBlocks().size(); ++blk)
        {
            auto &unitvecblock = unit_vec.GetBlocks()[blk];
            const auto nCoeff  = unitvecblock.GetNumData();

            nCoeffMax = (nCoeff > nCoeffMax) ? nCoeff : nCoeffMax;
        }

        // Compute diagonal.
        for (unsigned mode = 0; mode < nCoeffMax; ++mode)
        {
            for (unsigned blk = 0; blk < unit_vec.GetBlocks().size(); ++blk)
            {
                const unsigned int streamID = blk + 1;

                auto &unitblk     = unit_vec.GetBlocks()[blk];
                const auto nCoeff = unitblk.GetNumData();

                if (mode < nCoeff)
                {
                    const bool isInterleaved =
                        (unitblk.GetInterleaveWidth() != 1);
                    const auto numdata = unitblk.GetNumData();
                    const auto nelmt   = unitblk.GetNumElementsWithPadding();
                    auto *blkptr =
                        unitblk.template GetPtr<MemSpace, WriteOnly>(streamID);

                    // Loop over components.
                    for (unsigned int n = 0; n < unitblk.GetNumComponents();
                         ++n)
                    {
                        // Set ith term in unit vector to be 1.
                        SetModeBlkKernel<ExecSpace>(mode, nelmt, numdata,
                                                    (TData)1.0, blkptr,
                                                    isInterleaved, streamID);
                        blkptr +=
                            unitblk.CompSize() * unitblk.GetNumHomoModes();
                    }
                }
            }

            // Apply the operator to unit vector and store in the action field.
            op->Apply(unit_vec, action);
            m_robBCOp->Apply(unit_vec, action);

            for (unsigned blk = 0; blk < unit_vec.GetBlocks().size(); ++blk)
            {
                const unsigned int streamID = blk + 1;

                auto &unitblk     = unit_vec.GetBlocks()[blk];
                auto &actionblk   = action.GetBlocks()[blk];
                auto &diagblk     = m_invDiag.GetBlocks()[blk];
                const auto nCoeff = unitblk.GetNumData();

                if (mode < nCoeff)
                {
                    const bool isInterleaved =
                        (unitblk.GetInterleaveWidth() != 1);
                    const auto numdata = unitblk.GetNumData();
                    const auto nelmt   = unitblk.GetNumElementsWithPadding();
                    auto *unitptr =
                        unitblk.template GetPtr<MemSpace, WriteOnly>(streamID);
                    auto *actionptr =
                        actionblk.template GetPtr<MemSpace, ReadOnly>(streamID);
                    auto *diagptr =
                        diagblk.template GetPtr<MemSpace, WriteOnly>(streamID);

                    // Loop over components.
                    for (unsigned int n = 0; n < diagblk.GetNumComponents();
                         ++n)
                    {
                        // Copy the ith row term from the action field to get
                        // the ith diagonal.
                        CopyModeBlkKernel<ExecSpace>(mode, nelmt, numdata,
                                                     actionptr, diagptr,
                                                     isInterleaved, streamID);

                        // Reset the ith term in the unit vector to be 0.
                        SetModeBlkKernel<ExecSpace>(mode, nelmt, numdata,
                                                    (TData)0.0, unitptr,
                                                    isInterleaved, streamID);

                        // Increment pointers.
                        actionptr +=
                            actionblk.CompSize() * actionblk.GetNumHomoModes();
                        unitptr +=
                            unitblk.CompSize() * unitblk.GetNumHomoModes();
                        diagptr +=
                            diagblk.CompSize() * diagblk.GetNumHomoModes();
                    }

                    // Set diagonal interleave format.
                    diagblk.template SetInterleaveWidth<TData>(
                        actionblk.GetInterleaveWidth());
                }
            }
        }

        // Assembly and scatter values (without a sign change).
        m_assmbScatrNoSignOp->Apply(m_invDiag);

        // Invert diagonal.
        for (unsigned blk = 0; blk < m_invDiag.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            // Block dependent.
            auto &block  = m_invDiag.GetBlocks()[blk];
            auto diagptr = block.template GetPtr<MemSpace, ReadWrite>(streamID);

            // Loop over components.
            for (unsigned int n = 0; n < block.GetNumComponents(); ++n)
            {
                InvDiagBlkKernel<ExecSpace>(block.CompSize(), diagptr,
                                            streamID);
                diagptr += block.CompSize() * block.GetNumHomoModes();
            }
        }
    }
};

} // namespace Nektar::Operators::detail
