///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledDeviceSumFac.hpp
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
#include "Operators/ElmtOps/OperatorPhysInterp1DScaled.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorPhysInterp1DScaledImpl : public OperatorPhysInterp1DScaled<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorPhysInterp1DScaledImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysInterp1DScaled<TData>(expansionList)
    {
    }

    void SetScaleFactor(double scale) final
    {
        if (this->m_scale != scale)
        {
            this->m_scale = scale;

            // Loop over the elements of expansionList.
            size_t nDim = this->m_expansionList->GetShapeDimension();

            for (size_t i = 0; i < this->m_expansionList->GetNumElmts(); ++i)
            {
                const auto expPtr = this->m_expansionList->GetExp(i);

                int npts0 = expPtr->GetBasis(0)->GetNumPoints();

                // Fetch basiskeys of the current element.
                for (size_t d = 0; d < nDim; d++)
                {
                    LibUtilities::BasisKey b =
                        expPtr->GetBasis(d)->GetBasisKey();
                    int npts = b.GetNumPoints();

                    // if delta between npts and npts0 is 1 then keep this delta
                    // for new poitns to capitalise on switch templating
                    npts = (npts0 - npts == 1) ? (int)(scale * npts0) - 1
                                               : (int)(scale * npts);

                    LibUtilities::PointsKey p(npts, b.GetPointsType());

                    // make basis using modified direction with num points as
                    // modes and new quarature points as numpoints
                    LibUtilities::BasisKey bnew(b.GetBasisType(),
                                                b.GetNumPoints(), p);

                    // If necessary initialise this  basis data in  map.
                    if (m_interpMap.find(bnew) == m_interpMap.end())
                    {
                        m_interpMap[bnew] =
                            GetBasisData<MemSpace, double, TData>(
                                expPtr->GetBasis(d), eInterp,
                                __STDCPP_DEFAULT_NEW_ALIGNMENT__, npts);
                    }
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
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
                                  size_t nElmts, [[maybe_unused]] size_t nm0,
                                  size_t nm1, size_t nm2)
    {
        size_t wspsize = 0;

        if ((shapeType == LibUtilities::Quad) ||
            (shapeType == LibUtilities::Tri))
        {
            wspsize = nm1 * nElmts;
        }
        else if ((shapeType == LibUtilities::Hex) ||
                 (shapeType == LibUtilities::Tet) ||
                 (shapeType == LibUtilities::Prism) ||
                 (shapeType == LibUtilities::Pyr))
        {
            wspsize = (nm1 * nm2 + nm2) * nElmts;
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
            OperatorPhysInterp1DScaledImpl<ExecSpace, Implementation, TData>>(
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
    size_t m_nComps;

    unsigned int m_blk;

    LocalRegions::ExpansionSharedPtr m_expPtr;

    BasisDataMap<TData> m_interpMap;
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

    template <int nm0, int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Fetch basis data.
        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        auto B0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, f necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // BwdTrans kernel.
            BwdTrans1DKernel<ExecSpace, Implementation>(nm0, nq0, nElmtsPad, B0,
                                                        inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_expPtr->GetNumPoints(0);
        const auto nq0 = (int)(nm0 * this->m_scale);

        // Fetch basis data.
        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        auto B0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, f necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            // BwdTrans kernel.
            BwdTrans1DKernel<ExecSpace, Implementation>(nm0, nq0, nElmtsPad, B0,
                                                        inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }

        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
    template <int nm0, int nm1, int nq0, int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Fetch interp data.
        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        LibUtilities::BasisKey b1 = m_expPtr->GetBasis(1)->GetBasisKey();
        LibUtilities::PointsKey p1(nq1, b1.GetPointsType());
        LibUtilities::BasisKey b1new(b1.GetBasisType(), nm1, p1);

        auto B0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();
        auto B1 = m_interpMap[b1new].template GetPtr<MemSpace, ReadOnly>();

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
                    SetWorkspace(LibUtilities::Quad, nElmtsPad, nm0, nm1, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // BwdTrans kernel.
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            BwdTrans2DKernel<LibUtilities::Quad, ExecSpace, Implementation>(
                nm0, nm1, nq0, nq1, nElmtsPad, false, B0, B1, wspptr, inptr,
                outptr);

            inptr += inblock.size();
            outptr += outblock.size();
        }
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_expPtr->GetNumPoints(0);
        const auto nm1 = m_expPtr->GetNumPoints(1);

        const auto nq0 = (int)(this->m_scale * nm0);
        // if delta between nm0 and nm1 is 1 then keep this delta
        // for new poitns to capitalise on switch templating
        const int nq1 = (nm0 - nm1 == 1) ? (int)(this->m_scale * nm0) - 1
                                         : (int)(this->m_scale * nm1);

        // Fetch interp data.
        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        LibUtilities::BasisKey b1 = m_expPtr->GetBasis(1)->GetBasisKey();
        LibUtilities::PointsKey p1(nq1, b1.GetPointsType());
        LibUtilities::BasisKey b1new(b1.GetBasisType(), nm1, p1);

        auto B0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();
        auto B1 = m_interpMap[b1new].template GetPtr<MemSpace, ReadOnly>();

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
                    SetWorkspace(LibUtilities::Quad, nElmtsPad, nm0, nm1, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // BwdTrans kernel.
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            BwdTrans2DKernel<LibUtilities::Quad, ExecSpace, Implementation>(
                nm0, nm1, nq0, nq1, nElmtsPad, false, B0, B1, wspptr, inptr,
                outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    template <int nm0, int nm1, int nm2, int nq0, int nq1, int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        LibUtilities::BasisKey b1 = m_expPtr->GetBasis(1)->GetBasisKey();
        LibUtilities::PointsKey p1(nq1, b1.GetPointsType());
        LibUtilities::BasisKey b1new(b1.GetBasisType(), nm1, p1);

        LibUtilities::BasisKey b2 = m_expPtr->GetBasis(2)->GetBasisKey();
        LibUtilities::PointsKey p2(nq2, b2.GetPointsType());
        LibUtilities::BasisKey b2new(b2.GetBasisType(), nm2, p2);

        auto B0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();
        auto B1 = m_interpMap[b1new].template GetPtr<MemSpace, ReadOnly>();
        auto B2 = m_interpMap[b2new].template GetPtr<MemSpace, ReadOnly>();

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
                    SetWorkspace(LibUtilities::Hex, nElmtsPad, nm0, nm1, nm2));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // BwdTrans kernel.
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            BwdTrans3DKernel<LibUtilities::Hex, ExecSpace, Implementation>(
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, false, nullptr,
                nullptr, B0, B1, B2, wspptr, inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_expPtr->GetNumPoints(0);
        const auto nm1 = m_expPtr->GetNumPoints(1);
        const auto nm2 = m_expPtr->GetNumPoints(2);

        const auto nq0 = (int)(this->m_scale * nm0);
        // if delta between nm0 and nm1 is 1 then keep this delta
        // for new poitns to capitalise on switch templating
        const int nq1 = (nm0 - nm1 == 1) ? (int)(this->m_scale * nm0) - 1
                                         : (int)(this->m_scale * nm1);
        const int nq2 = (nm0 - nm2 == 1) ? (int)(this->m_scale * nm0) - 1
                                         : (int)(this->m_scale * nm2);

        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        LibUtilities::BasisKey b1 = m_expPtr->GetBasis(1)->GetBasisKey();
        LibUtilities::PointsKey p1(nq1, b1.GetPointsType());
        LibUtilities::BasisKey b1new(b1.GetBasisType(), nm1, p1);

        LibUtilities::BasisKey b2 = m_expPtr->GetBasis(2)->GetBasisKey();
        LibUtilities::PointsKey p2(nq2, b2.GetPointsType());
        LibUtilities::BasisKey b2new(b2.GetBasisType(), nm2, p2);

        auto B0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();
        auto B1 = m_interpMap[b1new].template GetPtr<MemSpace, ReadOnly>();
        auto B2 = m_interpMap[b2new].template GetPtr<MemSpace, ReadOnly>();

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
                    SetWorkspace(LibUtilities::Hex, nElmtsPad, nm0, nm1, nm2));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // BwdTrans kernel.
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            BwdTrans3DKernel<LibUtilities::Hex, ExecSpace, Implementation>(
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, false, nullptr,
                nullptr, B0, B1, B2, wspptr, inptr, outptr);
            inptr += inblock.size();
            outptr += outblock.size();
        }
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
