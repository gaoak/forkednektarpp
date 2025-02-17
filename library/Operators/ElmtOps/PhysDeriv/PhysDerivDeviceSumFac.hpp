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

#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorPhysDerivImpl : public BlockOperatorPhysDeriv<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorPhysDerivImpl(const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorPhysDeriv<TData>(exp, dataWarehouse)
    {
    }

    void apply(BlockAccessor<TData> &inblock,
               BlockAccessor<TData> &outblock) override
    {
        // Determine shape and type of the element.
        const auto shapeType = this->m_exp->DetShapeType();

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

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BlockOperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth =
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
        const auto nq0 = this->m_exp->GetNumPoints(0);

        const auto nCoord = this->m_exp->GetCoordim();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eDerivative));

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), transpose));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv1DKernel<ExecSpace, Implementation, DEFORMED>(
                nCoord, nq0, nElmtsPad, D0, dfptr, inptr, outptr);

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
        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eDerivative));

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), transpose));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv1DKernel<ExecSpace, Implementation, DEFORMED, nCoord, nq0>(
                nElmtsPad, D0, dfptr, inptr, outptr);

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
        const auto nq0 = this->m_exp->GetNumPoints(0);
        const auto nq1 = this->m_exp->GetNumPoints(1);

        const auto nCoord = this->m_exp->GetCoordim();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eDerivative));
        auto f0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eHalfMultOnePlusZero));
        auto f1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eTwoOverOneMinusZero));

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), transpose));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
                nCoord, nq0, nq1, nElmtsPad, D0, D1, f0, f1, dfptr, inptr,
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
        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eDerivative));
        auto f0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eHalfMultOnePlusZero));
        auto f1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eTwoOverOneMinusZero));

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), transpose));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED,
                              nCoord, nq0, nq1>(nElmtsPad, D0, D1, f0, f1,
                                                dfptr, inptr, outptr);

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
        const auto nq0 = this->m_exp->GetNumPoints(0);
        const auto nq1 = this->m_exp->GetNumPoints(1);
        const auto nq2 = this->m_exp->GetNumPoints(2);

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eDerivative));
        auto D2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                eDerivative));
        auto f0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eHalfMultOnePlusZero));
        auto f1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eHalfMultOnePlusZero));
        auto f1m = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eTwoOverOneMinusZero));
        auto f2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                eTwoOverOneMinusZero));

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), transpose));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
                nq0, nq1, nq2, nElmtsPad, D0, D1, D2, f0, f1, f1m, f2, dfptr,
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
        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eDerivative));
        auto D2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                eDerivative));
        auto f0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eHalfMultOnePlusZero));
        auto f1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eHalfMultOnePlusZero));
        auto f1m = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eTwoOverOneMinusZero));
        auto f2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                eTwoOverOneMinusZero));

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), transpose));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDeriv3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED,
                              nq0, nq1, nq2>(nElmtsPad, D0, D1, D2, f0, f1, f1m,
                                             f2, dfptr, inptr, outptr);
            inptr += inblock.size();
            outptr += 3 * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
