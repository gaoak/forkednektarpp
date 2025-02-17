///////////////////////////////////////////////////////////////////////////////
//
// File: MassDeviceSumFac.hpp
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

#include "Operators/ElmtOps/OperatorMass.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/Mass/MassDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorMassImpl : public BlockOperatorMass<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorMassImpl(const LocalRegions::ExpansionSharedPtr &exp,
                          NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorMass<TData>(exp, dataWarehouse)
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
            BlockOperatorMassImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    MemoryRegion<TData> m_wsp;

    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::vector_width<TData>::value
            : 1u;

    unsigned int GetSharedWorkspaceSize(LibUtilities::ShapeType shapeType,
                                        unsigned int nElmts,
                                        [[maybe_unused]] unsigned int nq0,
                                        unsigned int nq1, unsigned int nq2,
                                        [[maybe_unused]] unsigned int nm0,
                                        unsigned int nm1, unsigned int nm2)
    {
        unsigned int wspsize = 0;

        if (shapeType == LibUtilities::Seg)
        {
            wspsize = nq0 * nElmts;
        }
        else if (shapeType == LibUtilities::Quad)
        {
            wspsize = (nq0 * nq1 + nq1) * nElmts;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = (nq0 * nq1 + std::max(nq1, nm0)) * nElmts;
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = (nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nElmts;
        }
        else if (shapeType == LibUtilities::Tet)
        {
            unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

            wspsize = (nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm01) +
                       std::max(nq2, nm0) + nm2) *
                      nElmts;
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = (nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                       std::max(nq2, nm0) + nm1) *
                      nElmts;
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = (nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                       std::max(nq2, nm0)) *
                      nElmts;
        }

        return wspsize;
    }

    MemoryRegion<TData> SetWorkspace(LibUtilities::ShapeType shapeType,
                                     unsigned int nElmts, unsigned int nq0,
                                     unsigned int nq1, unsigned int nq2,
                                     unsigned int nm0, unsigned int nm1,
                                     unsigned int nm2)
    {
        constexpr bool device_only = true;

        unsigned int wspsize = GetSharedWorkspaceSize(shapeType, nElmts, nq0,
                                                      nq1, nq2, nm0, nm1, nm2);

        return MemoryRegion<TData>::template Create<MemSpace>(
            wspsize, ExecSpace::alignment, device_only);
    }

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
        const auto nm0 = this->m_exp->GetBasisNumModes(0);
        const auto nq0 = this->m_exp->GetNumPoints(0);

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp =
                    SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, 0, 0, nm0, 0, 0);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // IProduct kernel.
            Mass1DKernel<ExecSpace, Implementation, DEFORMED>(
                nm0, nq0, nElmtsPad, B0, W0, jacptr, wspptr, inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp =
                    SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, 0, 0, nm0, 0, 0);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // IProduct kernel.
            Mass1DKernel<ExecSpace, Implementation, DEFORMED, nm0, nq0>(
                nElmtsPad, B0, W0, jacptr, wspptr, inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
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
        const auto nm0 = this->m_exp->GetBasisNumModes(0);
        const auto nm1 = this->m_exp->GetBasisNumModes(1);

        const auto nq0 = this->m_exp->GetNumPoints(0);
        const auto nq1 = this->m_exp->GetNumPoints(1);

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (this->m_exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Precompute index, if necessary.
        const bool indexing =
            SHAPE_TYPE == LibUtilities::Tri &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto index0 = indexing
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, 0))
                          : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp = SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1, 0, nm0,
                                     nm1, 0);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // IProduct kernel.
            Mass2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
                nm0, nm1, nq0, nq1, nElmtsPad, isModified, index0, B0, B1, W0,
                W1, jacptr, wspptr, inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (this->m_exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Precompute index, if necessary.
        const bool indexing =
            SHAPE_TYPE == LibUtilities::Tri &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto index0 = indexing
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, 0))
                          : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp = SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1, 0, nm0,
                                     nm1, 0);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // IProduct kernel.
            Mass2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED, nm0,
                         nm1, nq0, nq1>(nElmtsPad, isModified, index0, B0, B1,
                                        W0, W1, jacptr, wspptr, inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
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
        const auto nm0 = this->m_exp->GetBasisNumModes(0);
        const auto nm1 = this->m_exp->GetBasisNumModes(1);
        const auto nm2 = this->m_exp->GetBasisNumModes(2);

        const auto nq0 = this->m_exp->GetNumPoints(0);
        const auto nq1 = this->m_exp->GetNumPoints(1);
        const auto nq2 = this->m_exp->GetNumPoints(2);

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (this->m_exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eBasis));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eWeights));
        auto W2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Precompute index, if necessary.
        const bool indexingTet =
            SHAPE_TYPE == LibUtilities::Tet &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        const bool indexingPrism =
            SHAPE_TYPE == LibUtilities::Prism &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        const bool indexingPyr =
            SHAPE_TYPE == LibUtilities::Pyr &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto index0 = (indexingTet || indexingPrism || indexingPyr)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, nm2, 0))
                          : nullptr;
        auto index1 = (indexingTet || indexingPrism || indexingPyr)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, nm2, 1))
                          : nullptr;
        auto index2 = (indexingTet || indexingPrism)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, nm2, 2))
                          : nullptr;
        auto index3 = (indexingTet)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, nm2, 3))
                          : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp = SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1, nq2, nm0,
                                     nm1, nm2);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // IProduct kernel.
            Mass3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, isModified, index0,
                index1, index2, index3, B0, B1, B2, W0, W1, W2, jacptr, wspptr,
                inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (this->m_exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eBasis));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                eWeights));
        auto W2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Precompute index, if necessary.
        const bool indexingTet =
            SHAPE_TYPE == LibUtilities::Tet &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        const bool indexingPrism =
            SHAPE_TYPE == LibUtilities::Prism &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        const bool indexingPyr =
            SHAPE_TYPE == LibUtilities::Pyr &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto index0 = (indexingTet || indexingPrism || indexingPyr)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, nm2, 0))
                          : nullptr;
        auto index1 = (indexingTet || indexingPrism || indexingPyr)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, nm2, 1))
                          : nullptr;
        auto index2 = (indexingTet || indexingPrism)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, nm2, 2))
                          : nullptr;
        auto index3 = (indexingTet)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(SHAPE_TYPE, nm0, nm1, nm2, 3))
                          : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() == 0)
            {
                m_wsp = SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1, nq2, nm0,
                                     nm1, nm2);
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp.template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // IProduct kernel.
            Mass3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED, nm0,
                         nm1, nm2, nq0, nq1, nq2>(
                nElmtsPad, isModified, index0, index1, index2, index3, B0, B1,
                B2, W0, W1, W2, jacptr, wspptr, inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
