///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransDeviceSumFac.hpp
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
#include "Operators/ElmtOps/OperatorBwdTrans.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorBwdTransImpl : public OperatorBwdTrans<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
        // Initialize the basis data.
        m_basisMap =
            GetBasisData<MemSpace, NekDouble, TData>(expansionList, eBasis);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Initialize index.
        size_t exp_idx = 0;

        m_nComps = in.GetNumComponents();
        ASSERTL1(m_nComps == out.GetNumComponents(),
                 "Number of input and output components differ");

        // Loop over the blocks.
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

    size_t GetSharedWorkspaceSize(LibUtilities::ShapeType shapeType,
                                  size_t nElmts, size_t nm0, size_t nm1,
                                  size_t nm2)
    {
        size_t wspsize = 0;

        if (shapeType == LibUtilities::Quad)
        {
            wspsize = nm1 * nElmts;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = nm0 * nElmts;
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = (nm1 * nm2 + nm2) * nElmts;
        }
        else if (shapeType == LibUtilities::Tet)
        {
            wspsize = ((2 * nm1 - nm0 + 1) * nm0 / 2 + nm0) * nElmts;
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = (nm0 * nm1 + nm0) * nElmts;
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = (nm0 * nm1 + nm0) * nElmts;
        }

        return wspsize;
    }

    MemoryRegion<TData> SetWorkspace(LibUtilities::ShapeType shapeType,
                                     size_t nElmts, size_t nm0, size_t nm1,
                                     size_t nm2)
    {
        constexpr bool device_only = true;

        size_t wspsize =
            GetSharedWorkspaceSize(shapeType, nElmts, nm0, nm1, nm2);

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
            OperatorBwdTransImpl<ExecSpace, Implementation, TData>>(
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

    BasisDataMap<TData> m_basisMap;
    std::vector<MemoryRegion<TData>> m_wsp;
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index0;
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index1;
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

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // BwdTrans kernel.
            BwdTrans1DKernel<ExecSpace, Implementation>(nm0, nq0, nElmtsPad, B0,
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
              unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // BwdTrans kernel.
            BwdTrans1DKernel<ExecSpace, Implementation, nm0, nq0>(
                nElmtsPad, B0, inptr, outptr);
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

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(
                    SetWorkspace(SHAPE_TYPE, nElmtsPad, nm0, nm1, 0));
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

            // BwdTrans kernel.
            BwdTrans2DKernel<SHAPE_TYPE, ExecSpace, Implementation>(
                nm0, nm1, nq0, nq1, nElmtsPad, isModified, B0, B1, wspptr,
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
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(
                    SetWorkspace(SHAPE_TYPE, nElmtsPad, nm0, nm1, 0));
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

            // BwdTrans kernel.
            BwdTrans2DKernel<SHAPE_TYPE, ExecSpace, Implementation, nm0, nm1,
                             nq0, nq1>(nElmtsPad, isModified, B0, B1, wspptr,
                                       inptr, outptr);
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
        constexpr bool device_only = true;

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

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey(),
            m_expPtr->GetBasis(2)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto B2 =
            m_basisMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Precompute index, if necessary.
        const bool indexing =
            SHAPE_TYPE == LibUtilities::Tet &&
            std::is_same_v<Implementation, Operators::SumFacQP>;

        if (indexing)
        {
            if (m_index0.find(basisKeys) == m_index0.end())
            {
                const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
                std::vector<unsigned int> index0(nm01);
                std::vector<unsigned int> index1(nm01);
                for (unsigned int p = 0, mode_pq = 0; p < nm0; p++)
                {
                    for (unsigned int q = 0; q < nm1 - p; q++, mode_pq++)
                    {
                        index0[mode_pq] = p;
                        index1[mode_pq] = q;
                    }
                }
                m_index0[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index0, ExecSpace::alignment, device_only);
                m_index1[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index1, ExecSpace::alignment, device_only);
            }
        }

        auto index0 =
            indexing ? m_index0[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                     : nullptr;
        auto index1 =
            indexing ? m_index1[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                     : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(
                    SetWorkspace(SHAPE_TYPE, nElmtsPad, nm0, nm1, nm2));
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

            // BwdTrans kernel.
            BwdTrans3DKernel<SHAPE_TYPE, ExecSpace, Implementation>(
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, isModified, index0,
                index1, B0, B1, B2, wspptr, inptr, outptr);
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
        constexpr bool device_only = true;

        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey(),
            m_expPtr->GetBasis(2)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto B2 =
            m_basisMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Precompute index, if necessary.
        const bool indexing =
            SHAPE_TYPE == LibUtilities::Tet &&
            std::is_same_v<Implementation, Operators::SumFacQP>;

        if (indexing)
        {
            if (m_index0.find(basisKeys) == m_index0.end())
            {
                const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
                std::vector<unsigned int> index0(nm01);
                std::vector<unsigned int> index1(nm01);
                for (unsigned int p = 0, mode_pq = 0; p < nm0; p++)
                {
                    for (unsigned int q = 0; q < nm1 - p; q++, mode_pq++)
                    {
                        index0[mode_pq] = p;
                        index1[mode_pq] = q;
                    }
                }
                m_index0[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index0, ExecSpace::alignment, device_only);
                m_index1[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index1, ExecSpace::alignment, device_only);
            }
        }

        auto index0 =
            indexing ? m_index0[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                     : nullptr;
        auto index1 =
            indexing ? m_index1[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                     : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(
                    SetWorkspace(SHAPE_TYPE, nElmtsPad, nm0, nm1, nm2));
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

            // BwdTrans kernel.
            BwdTrans3DKernel<SHAPE_TYPE, ExecSpace, Implementation, nm0, nm1,
                             nm2, nq0, nq1, nq2>(nElmtsPad, isModified, index0,
                                                 index1, B0, B1, B2, wspptr,
                                                 inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
