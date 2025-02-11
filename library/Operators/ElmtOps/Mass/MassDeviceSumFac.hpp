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
class OperatorMassImpl : public OperatorMass<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorMassImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMass<TData>(expansionList)
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        m_nComps = in.GetNumComponents();
        ASSERTL1(m_nComps == out.GetNumComponents(),
                 "Number of input and output components differ");

        // Initialize index.
        m_exp_idx = 0;

        // Loop over the blocks.
        for (m_blk = 0; m_blk < in.GetBlocks().size(); ++m_blk)
        {
            m_expPtr = this->m_expansionList->GetExp(m_exp_idx);

            // Block dependent.
            auto &inblock  = in.GetBlocks()[m_blk];
            auto &outblock = out.GetBlocks()[m_blk];

            this->BlockOperator(inblock, outblock);

            // Increment index for next element type.
            m_exp_idx += inblock.GetNumElements();
        }
    }

    size_t GetSharedWorkspaceSize(LibUtilities::ShapeType shapeType,
                                  size_t nElmts, [[maybe_unused]] size_t nq0,
                                  size_t nq1, size_t nq2,
                                  [[maybe_unused]] size_t nm0, size_t nm1,
                                  size_t nm2)
    {
        size_t wspsize = 0;

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
            size_t nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

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
                                     size_t nElmts, size_t nq0, size_t nq1,
                                     size_t nq2, size_t nm0, size_t nm1,
                                     size_t nm2)
    {
        constexpr bool device_only = true;

        size_t wspsize = GetSharedWorkspaceSize(shapeType, nElmts, nq0, nq1,
                                                nq2, nm0, nm1, nm2);

        return MemoryRegion<TData>::template Create<MemSpace>(
            wspsize, ExecSpace::alignment, device_only);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorMassImpl<ExecSpace, Implementation, TData>>(expansionList);
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

protected:
    unsigned int m_exp_idx;
    unsigned int m_blk;
    unsigned int m_nComps;

    LocalRegions::ExpansionSharedPtr m_expPtr;

    std::vector<MemoryRegion<TData>> m_wsp;

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
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nq0 = m_expPtr->GetNumPoints(0);

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(m_exp_idx, m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(
                    SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, 0, 0, nm0, 0, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
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
        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(m_exp_idx, m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(
                    SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, 0, 0, nm0, 0, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
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
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nm1 = m_expPtr->GetBasisNumModes(1);

        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);

        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(1)->GetBasisKey(), eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(m_exp_idx, m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Precompute index, if necessary.
        const bool indexing =
            SHAPE_TYPE == LibUtilities::Tri &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto index0 = indexing
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(m_expPtr, 0))
                          : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1, 0,
                                             nm0, nm1, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
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
        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(1)->GetBasisKey(), eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(m_exp_idx, m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Precompute index, if necessary.
        const bool indexing =
            SHAPE_TYPE == LibUtilities::Tri &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        auto index0 = indexing
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(m_expPtr, 0))
                          : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1, 0,
                                             nm0, nm1, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
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
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nm1 = m_expPtr->GetBasisNumModes(1);
        const auto nm2 = m_expPtr->GetBasisNumModes(2);

        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);
        const auto nq2 = m_expPtr->GetNumPoints(2);

        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(1)->GetBasisKey(), eBasis));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(2)->GetBasisKey(), eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                eWeights));
        auto W2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(m_exp_idx, m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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
                                ModeIndexKey(m_expPtr, 0))
                          : nullptr;
        auto index1 = (indexingTet || indexingPrism || indexingPyr)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(m_expPtr, 1))
                          : nullptr;
        auto index2 = (indexingTet || indexingPrism)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(m_expPtr, 2))
                          : nullptr;
        auto index3 = (indexingTet)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(m_expPtr, 3))
                          : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1,
                                             nq2, nm0, nm1, nm2));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
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
        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(1)->GetBasisKey(), eBasis));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(2)->GetBasisKey(), eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                eWeights));
        auto W2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<TData>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                eWeights));

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(m_exp_idx, m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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
                                ModeIndexKey(m_expPtr, 0))
                          : nullptr;
        auto index1 = (indexingTet || indexingPrism || indexingPyr)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(m_expPtr, 1))
                          : nullptr;
        auto index2 = (indexingTet || indexingPrism)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(m_expPtr, 2))
                          : nullptr;
        auto index3 = (indexingTet)
                          ? this->m_dataWarehouse->template GetData<ExecSpace>(
                                ModeIndexKey(m_expPtr, 3))
                          : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1,
                                             nq2, nm0, nm1, nm2));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
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
