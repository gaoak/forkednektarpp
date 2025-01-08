///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSerialAVXSumFac.hpp
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
#include "Operators/ElmtOps/OperatorIProductWRTBase.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Matrix-free implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise jacobian with paddings.
        auto locblocks = GetBlockAttributes<TData>(
            FieldState::Phys, expansionList, simd_t::width);
        m_jac = SetJacobian<MemSpace, TData>(expansionList, locblocks,
                                             ExecSpace::alignment);

        // Initialize the basis data.
        m_basisMap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eBasis, ExecSpace::alignment);
        m_weightMap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eWeights, ExecSpace::alignment);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
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

    static std::string className;

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    void BlockOperator(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock)
    {
        // Check alignment.
        WARNINGL1(inblock.GetAlignment() == simd_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(outblock.GetAlignment() == simd_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

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

    std::vector<MemoryRegion<TData>> m_jac;
    BasisDataMap<simd_t> m_basisMap;
    BasisDataMap<simd_t> m_weightMap;

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

        const auto nmTot = nm0;
        const auto nqTot = nq0;

        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian data.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int in_interleave_width  = inblock.GetInterleaveWidth();
        unsigned int out_interleave_width = outblock.GetInterleaveWidth();
        auto width_ratio                  = (in_interleave_width == 1)
                                                ? 1
                                                : in_interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, in_interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Initialize pointers.
        auto input  = (in_interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // loop over componnets
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr = jacPtr_init;
            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        out_interleave_width, chunkSize, nmTot, outptr);
                }

                // IProduct Kernel.
                IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nq0, inptr, B0, W0, jacPtr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot * simd_t::width;
                jacPtr += jacSize;
            }
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot = nm0;
        constexpr auto nqTot = nq0;

        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian data.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int in_interleave_width  = inblock.GetInterleaveWidth();
        unsigned int out_interleave_width = outblock.GetInterleaveWidth();
        auto width_ratio                  = (in_interleave_width == 1)
                                                ? 1
                                                : in_interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, in_interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Initialize pointers.
        auto input  = (in_interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr = jacPtr_init;
            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        out_interleave_width, chunkSize, nmTot, outptr);
                }

                // IProduct Kernel.
                IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nq0, inptr, B0, W0, jacPtr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot * simd_t::width;
                jacPtr += jacSize;
            }
        }
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

        const auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian data.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int in_interleave_width  = inblock.GetInterleaveWidth();
        unsigned int out_interleave_width = outblock.GetInterleaveWidth();
        auto width_ratio                  = (in_interleave_width == 1)
                                                ? 1
                                                : in_interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, in_interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        // Initialize pointers.
        auto input  = (in_interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr = jacPtr_init;
            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        out_interleave_width, chunkSize, nmTot, outptr);
                }

                // IProduct Kernel.
                IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nm1, nq0, nq1, isModified, inptr, B0, B1, W0, W1,
                    jacPtr, wsp0, outptr);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot * simd_t::width;
                jacPtr += jacSize;
            }
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nq0, int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot = nq0 * nq1;
        constexpr auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian data.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int in_interleave_width  = inblock.GetInterleaveWidth();
        unsigned int out_interleave_width = outblock.GetInterleaveWidth();
        auto width_ratio                  = (in_interleave_width == 1)
                                                ? 1
                                                : in_interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, in_interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        // Initialize pointers.
        auto input  = (in_interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr = jacPtr_init;
            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        out_interleave_width, chunkSize, nmTot, outptr);
                }

                // IProduct Kernel.
                IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nm1, nq0, nq1, isModified, inptr, B0, B1, W0, W1,
                    jacPtr, wsp0, outptr);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot * simd_t::width;
                jacPtr += jacSize;
            }
        }
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

        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
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
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W2 =
            m_weightMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian data.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int in_interleave_width  = inblock.GetInterleaveWidth();
        unsigned int out_interleave_width = outblock.GetInterleaveWidth();

        auto width_ratio = (in_interleave_width == 1)
                               ? 1
                               : in_interleave_width / simd_t::width;
        auto chunkSize   = std::max(simd_t::width, in_interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        // Initialize pointers.
        auto input  = (in_interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr = jacPtr_init;
            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        out_interleave_width, chunkSize, nmTot, outptr);
                }

                // IProduct Kernel.
                IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nm1, nm2, nq0, nq1, nq2, isModified, inptr, B0, B1, B2,
                    W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, outptr);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot * simd_t::width;
                jacPtr += jacSize;
            }
        }
    }
    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nm2, int nq0, int nq1, int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot = nq0 * nq1 * nq2;
        constexpr auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
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
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W2 =
            m_weightMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian data.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int in_interleave_width  = inblock.GetInterleaveWidth();
        unsigned int out_interleave_width = outblock.GetInterleaveWidth();
        auto width_ratio                  = (in_interleave_width == 1)
                                                ? 1
                                                : in_interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, in_interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        // Initialize pointers.
        auto input  = (in_interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr = jacPtr_init;
            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        out_interleave_width, chunkSize, nmTot, outptr);
                }

                // IProduct Kernel.
                IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nm1, nm2, nq0, nq1, nq2, isModified, inptr, B0, B1, B2,
                    W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, outptr);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot * simd_t::width;
                jacPtr += jacSize;
            }
        }
    }
};

} // namespace Nektar::Operators::detail
