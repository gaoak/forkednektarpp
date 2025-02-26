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

#include "Operators/ElmtOps/Mass/OperatorMass.hpp"
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
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(), eBasis)));
            m_W.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eWeights)));
        }

        if (m_dimension == 2)
        {
            // Precompute index, if necessary.
            const bool indexing =
                m_shapeType == LibUtilities::Tri &&
                std::is_same_v<Implementation, Operators::SumFacQP>;
            m_index.push_back(
                indexing ? this->m_dataWarehouse->template GetData<ExecSpace>(
                               ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], 0))
                         : nullptr);
        }
        else if (m_dimension == 3)
        {
            // Precompute index, if necessary.
            const bool indexingTet =
                m_shapeType == LibUtilities::Tet &&
                std::is_same_v<Implementation, Operators::SumFacQP>;
            const bool indexingPrism =
                m_shapeType == LibUtilities::Prism &&
                std::is_same_v<Implementation, Operators::SumFacQP>;
            const bool indexingPyr =
                m_shapeType == LibUtilities::Pyr &&
                std::is_same_v<Implementation, Operators::SumFacQP>;
            m_index.push_back(
                (indexingTet || indexingPrism || indexingPyr)
                    ? this->m_dataWarehouse->template GetData<ExecSpace>(
                          ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], m_nm[2],
                                       0))
                    : nullptr);
            m_index.push_back(
                (indexingTet || indexingPrism || indexingPyr)
                    ? this->m_dataWarehouse->template GetData<ExecSpace>(
                          ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], m_nm[2],
                                       1))
                    : nullptr);
            m_index.push_back(
                (indexingTet || indexingPrism)
                    ? this->m_dataWarehouse->template GetData<ExecSpace>(
                          ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], m_nm[2],
                                       2))
                    : nullptr);
            m_index.push_back(
                (indexingTet)
                    ? this->m_dataWarehouse->template GetData<ExecSpace>(
                          ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], m_nm[2],
                                       3))
                    : nullptr);
        }
    }

    void apply(BlockAccessor<TData> &inblock,
               BlockAccessor<TData> &outblock) override
    {
        switch (m_shapeType)
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
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::vector_width<TData>::value
            : 1u;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
    std::vector<const TData *> m_W;
    std::vector<const unsigned int *> m_index;
    MemoryRegion<TData> m_wsp;

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
        unsigned int wspsize = GetSharedWorkspaceSize(shapeType, nElmts, nq0,
                                                      nq1, nq2, nm0, nm1, nm2);

        return MemoryRegion<TData>::Create(wspsize, ExecSpace::alignment);
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
        const auto nm0 = m_nm[0];
        const auto nq0 = m_nq[0];

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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
                nm0, nq0, nElmtsPad, m_B[0], m_W[0], jacptr, wspptr, inptr,
                outptr);

            // Increment pointers.
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
                nElmtsPad, m_B[0], m_W[0], jacptr, wspptr, inptr, outptr);

            // Increment pointers.
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
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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
                nm0, nm1, nq0, nq1, nElmtsPad, m_isModified, m_index[0], m_B[0],
                m_B[1], m_W[0], m_W[1], jacptr, wspptr, inptr, outptr);

            // Increment pointers.
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
                         nm1, nq0, nq1>(nElmtsPad, m_isModified, m_index[0],
                                        m_B[0], m_B[1], m_W[0], m_W[1], jacptr,
                                        wspptr, inptr, outptr);

            // Increment pointers.
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
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];
        const auto nm2 = m_nm[2];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];
        const auto nq2 = m_nq[2];

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, m_isModified,
                m_index[0], m_index[1], m_index[2], m_index[3], m_B[0], m_B[1],
                m_B[2], m_W[0], m_W[1], m_W[2], jacptr, wspptr, inptr, outptr);

            // Increment pointers.
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
                nElmtsPad, m_isModified, m_index[0], m_index[1], m_index[2],
                m_index[3], m_B[0], m_B[1], m_B[2], m_W[0], m_W[1], m_W[2],
                jacptr, wspptr, inptr, outptr);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
