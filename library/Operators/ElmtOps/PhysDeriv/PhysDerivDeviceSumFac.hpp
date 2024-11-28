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

        // Initialize the points.
        m_zeroMap =
            GetBasisData<MemSpace, NekDouble, TData>(expansionList, eZeros);

        // Initialize the derivative matrix.
        m_derivativeMap = GetBasisData<MemSpace, NekDouble, TData>(
            expansionList, eDerivative);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Initialize index.
        size_t exp_idx = 0;

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

    LocalRegions::ExpansionSharedPtr m_expPtr;

    BasisDataMap<TData> m_zeroMap;
    BasisDataMap<TData> m_derivativeMap;
    std::vector<MemoryRegion<TData>> m_df;
    static constexpr size_t m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::vector_width<TData>::value
            : 1u;

    void SegBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator1D<LibUtilities::Seg, true>(inblock, outblock);
        }
        else
        {
            Operator1D<LibUtilities::Seg, false>(inblock, outblock);
        }
    }

    void TriBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator2D<LibUtilities::Tri, true>(inblock, outblock);
        }
        else
        {
            Operator2D<LibUtilities::Tri, false>(inblock, outblock);
        }
    }

    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator2D<LibUtilities::Quad, true>(inblock, outblock);
        }
        else
        {
            Operator2D<LibUtilities::Quad, false>(inblock, outblock);
        }
    }

    void HexBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator3D<LibUtilities::Hex, true>(inblock, outblock);
        }
        else
        {
            Operator3D<LibUtilities::Hex, false>(inblock, outblock);
        }
    }

    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator3D<LibUtilities::Prism, true>(inblock, outblock);
        }
        else
        {
            Operator3D<LibUtilities::Prism, false>(inblock, outblock);
        }
    }

    void PyrBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator3D<LibUtilities::Pyr, true>(inblock, outblock);
        }
        else
        {
            Operator3D<LibUtilities::Pyr, false>(inblock, outblock);
        }
    }

    void TetBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator3D<LibUtilities::Tet, true>(inblock, outblock);
        }
        else
        {
            Operator3D<LibUtilities::Tet, false>(inblock, outblock);
        }
    }

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

        // Reshape, if necessary.
        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
            inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
            (TData *)inptr);
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        PhysDeriv1DKernel<ExecSpace, Implementation, DEFORMED>(
            nq0, nCoord, nElmtsPad, D0, dfPtr, inptr, outptr);
    }

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
        auto Z0 = m_zeroMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto Z1 = m_zeroMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch deriv factors data.
        auto dfPtr = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Reshape, if necessary.
        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
            inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
            (TData *)inptr);
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        PhysDeriv2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
            nq0, nq1, nCoord, nElmtsPad, D0, D1, Z0, Z1, dfPtr, inptr, outptr);
    }

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
        auto Z0 = m_zeroMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto Z1 = m_zeroMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto Z2 = m_zeroMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch deriv factors data.
        auto dfPtr = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Reshape, if necessary.
        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
            inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
            (TData *)inptr);
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        PhysDeriv3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
            nq0, nq1, nq2, nElmtsPad, D0, D1, D2, Z0, Z1, Z2, dfPtr, inptr,
            outptr);
    }
};

} // namespace Nektar::Operators::detail
