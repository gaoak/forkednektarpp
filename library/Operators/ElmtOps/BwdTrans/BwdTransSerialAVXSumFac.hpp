///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransSerialAVXSumFac.hpp
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

#include "Operators/ElmtOps/BwdTrans/BwdTransOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BwdTransBlockOpImpl : public BwdTransBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    BwdTransBlockOpImpl(const LocalRegions::ExpansionSharedPtr &exp,
                        NekDataWarehouseSharedPtr dataWarehouse)
        : BwdTransBlockOp<TData>(exp, dataWarehouse)
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
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(), eBasis)));
        }

        if ((m_shapeType == LibUtilities::eNodalTri) ||
            (m_shapeType == LibUtilities::eNodalPrism) ||
            (m_shapeType == LibUtilities::eNodalTet))
        {
            // Fetch NodalToModal Matrix if required.
            m_nodToMod = this->m_dataWarehouse->template GetData<ExecSpace>(
                VandemondeKey<simd_t>(eNodalToModal, exp->GetElmtId()));
        }
        else
        {
            m_nodToMod = (const simd_t *)nullptr;
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BwdTransBlockOpImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const simd_t *> m_B;
    const simd_t *m_nodToMod;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        // Check alignment.
        WARNINGL1(inblock.GetAlignment() == simd_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(outblock.GetAlignment() == simd_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

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
            // Nodal Triangles
            case LibUtilities::NodalTri:
            {
                NodalTriBlock(inblock, outblock);
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
            case LibUtilities::NodalTet:
            {
                NodalTetBlock(inblock, outblock);
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
            case LibUtilities::NodalPrism:
            {
                NodalPrismBlock(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    void SegBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void TriBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void NodalTriBlock(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock);

    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock);

    void HexBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock);

    void NodalPrismBlock(BlockAccessor<TData> &inblock,
                         BlockAccessor<TData> &outblock);

    void PyrBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void TetBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void NodalTetBlock(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nq0 = m_nq[0];

        const auto nmTot = nm0;
        const auto nqTot = nq0;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        BwdTrans1DWorkspace<SHAPE_TYPE>(nm0, nq0);

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // BwdTrans kernel.
                BwdTrans1DKernel<SHAPE_TYPE>(
                    nm0, nq0, m_B[0], reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
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
        // Shape size.
        constexpr auto nmTot = nm0;
        constexpr auto nqTot = nq0;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        BwdTrans1DWorkspace<SHAPE_TYPE>(nm0, nq0);

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // BwdTrans kernel.
                BwdTrans1DKernel<SHAPE_TYPE>(
                    nm0, nq0, m_B[0], reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
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

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const auto nqTot = nq0 * nq1;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // BwdTrans kernel.
                BwdTrans2DKernel<SHAPE_TYPE>(
                    nm0, nm1, nq0, nq1, m_isModified, m_B[0], m_B[1],
                    m_nodToMod, wsp0.data(),
                    reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
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
        // Shape size.
        constexpr auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        constexpr auto nqTot = nq0 * nq1;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // BwdTrans kernel.
                BwdTrans2DKernel<SHAPE_TYPE>(
                    nm0, nm1, nq0, nq1, m_isModified, m_B[0], m_B[1],
                    m_nodToMod, wsp0.data(),
                    reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
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

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const auto nqTot = nq0 * nq1 * nq2;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // BwdTrans kernel.
                BwdTrans3DKernel<SHAPE_TYPE>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, m_B[0], m_B[1],
                    m_B[2], m_nodToMod, wsp0.data(), wsp1.data(),
                    reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
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
        // Shape size.
        constexpr auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        constexpr auto nqTot = nq0 * nq1 * nq2;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // BwdTrans kernel.
                BwdTrans3DKernel<SHAPE_TYPE>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, m_B[0], m_B[1],
                    m_B[2], m_nodToMod, wsp0.data(), wsp1.data(),
                    reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
