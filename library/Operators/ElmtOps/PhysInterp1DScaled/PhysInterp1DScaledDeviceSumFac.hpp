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

#include "Operators/ElmtOps/PhysInterp1DScaled/OperatorPhysInterp1DScaled.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorPhysInterp1DScaledImpl
    : public BlockOperatorPhysInterp1DScaled<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorPhysInterp1DScaledImpl(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorPhysInterp1DScaled<TData>(exp, dataWarehouse)
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
            m_nm.push_back(exp->GetNumPoints(d));
        }
    }

    void v_SetScaleFactor(TData scale) override
    {
        this->m_scale = scale;
        m_nq.clear();
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            if (d == 0)
            {
                m_nq.push_back(this->m_scale * m_nm[0]);
            }
            else if (d == 1)
            {
                // if delta between nm0 and nm1 is 1 then keep this delta
                // for new poitns to capitalise on switch templating
                const auto nq1 =
                    (m_nm[0] - m_nm[1] == 1)
                        ? (unsigned int)(this->m_scale * m_nm[0]) - 1
                        : (unsigned int)(this->m_scale * m_nm[1]);
                m_nq.push_back(nq1);
            }
            else if (d == 2)
            {
                const auto nq2 =
                    (m_nm[0] - m_nm[2] == 1)
                        ? (unsigned int)(this->m_scale * m_nm[0]) - 1
                        : (unsigned int)(this->m_scale * m_nm[2]);
                m_nq.push_back(nq2);
            }

            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(d)->GetBasisKey(),
                                    eInterp, m_nq[d])));
        }
    }

    void apply(BlockAccessor<TData> &inblock,
               BlockAccessor<TData> &outblock) override
    {
        ASSERTL1(this->m_scale != -1.0,
                 "Scale factor has not been initialised");

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
        return std::make_unique<BlockOperatorPhysInterp1DScaledImpl<
            ExecSpace, Implementation, TData>>(exp, dataWarehouse);
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
    MemoryRegion<TData> m_wsp;

    unsigned int GetWorkspaceSize(LibUtilities::ShapeType shapeType,
                                  unsigned int nElmts,
                                  [[maybe_unused]] unsigned int nm0,
                                  unsigned int nm1, unsigned int nm2)
    {
        unsigned int wspsize = 0;

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
                                     unsigned int nElmts, unsigned int nm0,
                                     unsigned int nm1, unsigned int nm2)
    {
        unsigned int wspsize =
            GetWorkspaceSize(shapeType, nElmts, nm0, nm1, nm2);

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
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nq0 = m_nq[0];

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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

            // BwdTrans kernel.
            BwdTrans1DKernel<ExecSpace, Implementation>(nm0, nq0, nElmtsPad,
                                                        m_B[0], inptr, outptr);

            // Increment pointer.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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

            // BwdTrans kernel.
            BwdTrans1DKernel<ExecSpace, Implementation, nm0, nq0>(
                nElmtsPad, m_B[0], inptr, outptr);

            // Increment pointer.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Non-size based operator.
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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
                    SetWorkspace(LibUtilities::Quad, nElmtsPad, nm0, nm1, 0);
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

            // BwdTrans kernel.
            BwdTrans2DKernel<LibUtilities::Quad, ExecSpace, Implementation>(
                nm0, nm1, nq0, nq1, nElmtsPad, false, m_B[0], m_B[1], inptr,
                outptr, wspptr);

            // Increment pointer.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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
                    SetWorkspace(LibUtilities::Quad, nElmtsPad, nm0, nm1, 0);
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

            // BwdTrans kernel.
            BwdTrans2DKernel<LibUtilities::Quad, ExecSpace, Implementation, nm0,
                             nm1, nq0, nq1>(nElmtsPad, false, m_B[0], m_B[1],
                                            inptr, outptr, wspptr);

            // Increment pointer.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Non-size based operator.
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
                    SetWorkspace(LibUtilities::Hex, nElmtsPad, nm0, nm1, nm2);
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

            // BwdTrans kernel.
            BwdTrans3DKernel<LibUtilities::Hex, ExecSpace, Implementation>(
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, false, nullptr,
                nullptr, m_B[0], m_B[1], m_B[2], inptr, outptr, wspptr);

            // Increment pointer.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

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
                    SetWorkspace(LibUtilities::Hex, nElmtsPad, nm0, nm1, nm2);
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

            // BwdTrans kernel.
            BwdTrans3DKernel<LibUtilities::Hex, ExecSpace, Implementation, nm0,
                             nm1, nm2, nq0, nq1, nq2>(
                nElmtsPad, false, nullptr, nullptr, m_B[0], m_B[1], m_B[2],
                inptr, outptr, wspptr);

            // Increment pointer.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
