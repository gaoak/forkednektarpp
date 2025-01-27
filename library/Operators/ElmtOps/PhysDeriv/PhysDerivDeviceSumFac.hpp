///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceSumFac.hpp
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

#include "Operators/Common/OperatorHelper.hpp"
#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorPhysDerivImpl : public OperatorPhysDeriv<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
        // Initialise the derivative factor.
        auto transpose = std::is_same_v<Implementation, Operators::SumFacQP>;
        auto locblocks = GetBlockAttributes<TData>(
            FieldState::Phys, expansionList, m_implInterleaveWidth);
        m_df = SetDerivativeFactor<MemSpace, TData>(
            expansionList, locblocks, ExecSpace::alignment, transpose);

        // Initialize the derivative matrix.
        m_derivativeMap = GetBasisData<MemSpace, NekDouble, TData>(
            expansionList, eDerivative);

        // Initialize the geometric factors.
        m_fac0 = GetBasisData<MemSpace, NekDouble, TData>(
            expansionList, eHalfMultOnePlusZero, ExecSpace::alignment);
        m_fac1 = GetBasisData<MemSpace, NekDouble, TData>(
            expansionList, eTwoOverOneMinusZero, ExecSpace::alignment);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Initialize index.
        size_t exp_idx = 0;

        m_nComps = in.GetNumComponents();
        ASSERTL1(m_nComps == out.GetNumComponents() /
                                 this->m_expansionList->GetCoordim(0),
                 "Number of input and output components differ");

        for (m_blk = 0; m_blk < in.GetBlocks().size(); ++m_blk)
        {
            m_expPtr = this->m_expansionList->GetExp(exp_idx);

            // Block dependent.
            auto &inblock  = in.GetBlocks()[m_blk];
            auto &outblock = out.GetBlocks()[m_blk];

            this->BlockOperator(inblock, outblock);

            // Increment index for next element type.
            exp_idx += inblock.GetNumElements();
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    void BlockOperator(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock)
    {
        // Determine shape and type of the element.
        const auto shapeType = m_expPtr->DetShapeType();

        switch (shapeType)
        {
            // Segment
            case LibUtilities::Seg:
            {
                SegBlock(inblock, outblock);
                break;
            }
            // Quads
            case LibUtilities::Quad:
            {
                QuadBlock(inblock, outblock);
                break;
            }
            // Triangles
            case LibUtilities::Tri:
            {
                TriBlock(inblock, outblock);
                break;
            }
            // Hexes
            case LibUtilities::Hex:
            {
                HexBlock(inblock, outblock);
                break;
            }
            // Tet
            case LibUtilities::Tet:
            {
                TetBlock(inblock, outblock);
                break;
            }
            // Pyr
            case LibUtilities::Pyr:
            {
                PyrBlock(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                PrismBlock(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

private:
    unsigned int m_blk;
    size_t m_nComps;

    LocalRegions::ExpansionSharedPtr m_expPtr;

    BasisDataMap<TData> m_derivativeMap;
    BasisDataMap<TData> m_fac0;
    BasisDataMap<TData> m_fac1;
    std::vector<MemoryRegion<TData>> m_df;
    static constexpr size_t m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::vector_width<TData>::value
            : 1u;

    void SegBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);
    void TriBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);
    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock);
    void HexBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);
    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock);
    void PyrBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);
    void TetBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nq0 = m_expPtr->GetNumPoints(0);

        const auto nCoord = m_expPtr->GetCoordim();

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch deriv factors data.
        auto dfPtr = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv1DKernel<ExecSpace, Implementation, DEFORMED>(
                nCoord, nq0, nElmtsPad, D0, dfPtr, inptr, outptr);

            inptr += inblock.size();
            outptr += nCoord * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nCoord, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch deriv factors data.
        auto dfPtr = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv1DKernel<ExecSpace, Implementation, DEFORMED, nCoord, nq0>(
                nElmtsPad, D0, dfPtr, inptr, outptr);

            inptr += inblock.size();
            outptr += nCoord * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);

        const auto nCoord = m_expPtr->GetCoordim();

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto f1 = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch deriv factors data.
        auto dfPtr = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
                nCoord, nq0, nq1, nElmtsPad, D0, D1, f0, f1, dfPtr, inptr,
                outptr);

            inptr += inblock.size();
            outptr += nCoord * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nCoord, unsigned int nq0, unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto f1 = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch deriv factors data.
        auto dfPtr = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED,
                              nCoord, nq0, nq1>(nElmtsPad, D0, D1, f0, f1,
                                                dfPtr, inptr, outptr);

            inptr += inblock.size();
            outptr += nCoord * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);
        const auto nq2 = m_expPtr->GetNumPoints(2);

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey(),
            m_expPtr->GetBasis(2)->GetBasisKey()};
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D2 =
            m_derivativeMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto f0  = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto f1  = m_fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f1m = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f2  = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch deriv factors data.
        auto dfPtr = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
                nq0, nq1, nq2, nElmtsPad, D0, D1, D2, f0, f1, f1m, f2, dfPtr,
                inptr, outptr);

            inptr += inblock.size();
            outptr += 3 * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey(),
            m_expPtr->GetBasis(2)->GetBasisKey()};
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D2 =
            m_derivativeMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto f0  = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto f1  = m_fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f1m = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f2  = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch deriv factors data.
        auto dfPtr = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED,
                              nq0, nq1, nq2>(nElmtsPad, D0, D1, D2, f0, f1, f1m,
                                             f2, dfPtr, inptr, outptr);
            inptr += inblock.size();
            outptr += 3 * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
