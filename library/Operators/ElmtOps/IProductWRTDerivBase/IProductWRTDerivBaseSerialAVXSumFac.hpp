///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialAVXSumFac.hpp
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
#include "Operators/ElmtOps/OperatorIProductWRTDerivBase.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Sum-factorisation implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorIProductWRTDerivBaseImpl
    : public OperatorIProductWRTDerivBase<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
        // Initialise jacobian with paddings.
        auto locblocks = GetBlockAttributes<TData>(
            FieldState::Phys, expansionList, simd_t::width);
        m_jac = SetJacobian<MemSpace, TData>(expansionList, locblocks,
                                             ExecSpace::alignment);
        m_df  = SetDerivativeFactor<MemSpace, TData>(expansionList, locblocks,
                                                    ExecSpace::alignment);

        // Initialize the basis data.
        m_basisMap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eBasis, simd_t::alignment);
        m_weightMap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eWeights, simd_t::alignment);

        // Initialize the derivative matrix.
        m_derivativeMap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eDerivative, simd_t::alignment);

        // Initialize the BD data.
        m_dbasisMap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eBasisDerivative, simd_t::alignment);

        // Initialize the geometric factors.
        m_Fac0 = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eHalfMultOnePlusZero, simd_t::alignment);
        m_Fac1 = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eTwoOverOneMinusZero, simd_t::alignment);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Initialize index.
        size_t exp_idx = 0;

        m_nComps = out.GetNumComponents();
        ASSERTL1(m_nComps == in.GetNumComponents() /
                                 this->m_expansionList->GetCoordim(0),
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

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
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
    std::vector<MemoryRegion<TData>> m_df;

    BasisDataMap<simd_t> m_basisMap;
    BasisDataMap<simd_t> m_dbasisMap;
    BasisDataMap<simd_t> m_derivativeMap;
    BasisDataMap<simd_t> m_weightMap;
    BasisDataMap<simd_t> m_Fac0;
    BasisDataMap<simd_t> m_Fac1;

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

        const auto ncoord = m_expPtr->GetCoordim();

        size_t jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nq0;
        }

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto DB0 =
            m_dbasisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

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

        // Workspace for kernels.
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ncoord);
        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp0(nq0);
        auto tmpPtr =
            reinterpret_cast<typename simd_t::scalarType *>(tmp0.data());

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
            auto dfPtr  = dfPtr_init;
            for (int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (int n = 0; n < ncoord; ++n)
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            in_interleave_width, chunkSize, nq0,
                            (TData *)(inptr +
                                      n * inblock.GetNumElmtGroups() * nq0));
                    }
                    if (this->m_append)
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nm0, outptr);
                    }
                }

                StdAlignDerivBase1D<DEFORMED>(nq0, ncoord, dfPtr, df_tmp,
                                              inblock.GetNumElmtGroups() * nq0,
                                              inptr, tmpPtr);
                IProductSegKernel<false, false, DEFORMED>(
                    nm0, nq0, (const typename simd_t::vectorType *)tmpPtr, DB0,
                    W0, jacPtr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nq0;
                outptr += nm0 * simd_t::width;
                jacPtr += jacSize;
                dfPtr += jacSize * ncoord;
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
        const auto ncoord = m_expPtr->GetCoordim();

        size_t jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nq0;
        }

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto DB0 =
            m_dbasisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

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

        // Workspace for kernels.
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ncoord);
        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp0(nq0);
        auto tmpPtr =
            reinterpret_cast<typename simd_t::scalarType *>(tmp0.data());

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
            auto dfPtr  = dfPtr_init;
            for (int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (int n = 0; n < ncoord; ++n)
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            in_interleave_width, chunkSize, nq0,
                            (TData *)(inptr +
                                      n * inblock.GetNumElmtGroups() * nq0));
                    }
                    if (this->m_append)
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nm0, outptr);
                    }
                }

                StdAlignDerivBase1D<DEFORMED>(nq0, ncoord, dfPtr, df_tmp,
                                              inblock.GetNumElmtGroups() * nq0,
                                              inptr, tmpPtr);
                IProductSegKernel<false, false, DEFORMED>(
                    nm0, nq0, (const typename simd_t::vectorType *)tmpPtr, DB0,
                    W0, jacPtr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nq0;
                outptr += nm0 * simd_t::width;
                jacPtr += jacSize;
                dfPtr += jacSize * ncoord;
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

        const auto ncoord = m_expPtr->GetCoordim();

        const auto ndf   = 2u * ncoord;
        const auto nqTot = nq0 * nq1;
        const auto nmTot = m_expPtr->GetNcoeffs();

        size_t jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        [[maybe_unused]] const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

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

        // Workspace for kernels.
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ndf);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp(nq1), tmp0(nqTot),
            tmp1(nqTot);
        typename simd_t::scalarType *tmpPtr[2];
        tmpPtr[0] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp0.data());
        tmpPtr[1] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp1.data());

        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp2(nqTot);
        auto tmp2Ptr =
            reinterpret_cast<typename simd_t::scalarType *>(tmp2.data());
        auto tmp2vec =
            reinterpret_cast<typename simd_t::vectorType *>(tmp2.data());

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
            auto jacPtr        = jacPtr_init;
            auto dfPtr         = dfPtr_init;
            auto NumElmtGroups = inblock.GetNumElmtGroups();

            if constexpr (SHAPE_TYPE == LibUtilities::eQuadrilateral)
            {
                const bool colldir0 = m_expPtr->GetBasis(0)->Collocation();
                const bool colldir1 = m_expPtr->GetBasis(1)->Collocation();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < ncoord; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        if (this->m_append)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                out_interleave_width, chunkSize, nmTot, outptr);
                        }
                    }

                    StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, ncoord, dfPtr, df_tmp, NumElmtGroups * nqTot,
                        inptr, tmpPtr, (simd_t *)nullptr, (simd_t *)nullptr);
                    SumDerivTensor2DKernel<DEFORMED, simd_t>(
                        nq0, nq1,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1], W0, W1,
                        jacPtr, D0, D1, tmp2Ptr);
                    IProductQuadKernel<false, false, simd_t>(
                        nm0, nm1, nq0, nq1, tmp2vec, B0, B1, wsp, outptr, 1.0,
                        colldir0, colldir1);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }

            if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
            {
                // Get geometric factors.
                auto F0 =
                    m_Fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
                auto F1 =
                    m_Fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    if (e % width_ratio == 0)
                    {
                        // Reshape, if necessary.
                        for (int n = 0; n < ncoord; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        if (this->m_append)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                out_interleave_width, chunkSize, nmTot, outptr);
                        }
                    }

                    StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, ncoord, dfPtr, df_tmp, NumElmtGroups * nqTot,
                        inptr, tmpPtr, F0, F1);
                    SumDerivTensor2DKernel<DEFORMED, simd_t>(
                        nq0, nq1,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1], W0, W1,
                        jacPtr, D0, D1, tmp2Ptr);
                    IProductTriKernel<false, false, simd_t>(
                        nm0, nm1, nq0, nq1, isModified, tmp2vec, B0, B1, wsp,
                        outptr);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }
            // advance input by ncoord-1 componennts since have already
            // advanced one component in the above
            inptr += nqTot * NumElmtGroups * (ncoord - 1);
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nq0, int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto ncoord = m_expPtr->GetCoordim();

        const auto ndf   = 2u * ncoord;
        const auto nqTot = nq0 * nq1;
        const auto nmTot = m_expPtr->GetNcoeffs();

        size_t jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        [[maybe_unused]] const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

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

        // Workspace for kernels.
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ndf);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp(nq1), tmp0(nqTot),
            tmp1(nqTot);
        typename simd_t::scalarType *tmpPtr[2];
        tmpPtr[0] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp0.data());
        tmpPtr[1] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp1.data());

        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp2(nqTot);
        auto tmp2Ptr =
            reinterpret_cast<typename simd_t::scalarType *>(tmp2.data());
        auto tmp2vec =
            reinterpret_cast<typename simd_t::vectorType *>(tmp2.data());

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
            auto jacPtr        = jacPtr_init;
            auto dfPtr         = dfPtr_init;
            auto NumElmtGroups = inblock.GetNumElmtGroups();

            if constexpr (SHAPE_TYPE == LibUtilities::eQuadrilateral)
            {
                const bool colldir0 = m_expPtr->GetBasis(0)->Collocation();
                const bool colldir1 = m_expPtr->GetBasis(1)->Collocation();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < ncoord; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        if (this->m_append)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                out_interleave_width, chunkSize, nmTot, outptr);
                        }
                    }

                    StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, ncoord, dfPtr, df_tmp, NumElmtGroups * nqTot,
                        inptr, tmpPtr, (simd_t *)nullptr, (simd_t *)nullptr);
                    SumDerivTensor2DKernel<DEFORMED, simd_t>(
                        nq0, nq1,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1], W0, W1,
                        jacPtr, D0, D1, tmp2Ptr);
                    IProductQuadKernel<false, false, simd_t>(
                        nm0, nm1, nq0, nq1, tmp2vec, B0, B1, wsp, outptr, 1.0,
                        colldir0, colldir1);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }

            if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
            {
                // Get geometric factors.
                auto F0 =
                    m_Fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
                auto F1 =
                    m_Fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < ncoord; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        if (this->m_append)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                out_interleave_width, chunkSize, nmTot, outptr);
                        }
                    }

                    StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, ncoord, dfPtr, df_tmp, NumElmtGroups * nqTot,
                        inptr, tmpPtr, F0, F1);
                    SumDerivTensor2DKernel<DEFORMED, simd_t>(
                        nq0, nq1,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1], W0, W1,
                        jacPtr, D0, D1, tmp2Ptr);
                    IProductTriKernel<false, false, simd_t>(
                        nm0, nm1, nq0, nq1, isModified, tmp2vec, B0, B1, wsp,
                        outptr);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }
            // advance input by ncoord-1 componennts since have already
            // advanced one component in the above
            inptr += nqTot * NumElmtGroups * (ncoord - 1);
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

        const auto ndf   = 9u;
        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot = m_expPtr->GetNcoeffs();

        size_t jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        [[maybe_unused]] const bool isModified =
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
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D2 =
            m_derivativeMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W2 =
            m_weightMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

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

        // Workspace for kernels.
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ndf);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp(nq1 * nq2),
            wsp0(nq2), tmp0(nqTot), tmp1(nqTot), tmp2(nqTot);
        typename simd_t::scalarType *tmpPtr[3];
        tmpPtr[0] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp0.data());
        tmpPtr[1] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp1.data());
        tmpPtr[2] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp2.data());

        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp3(nqTot);
        auto tmp3Ptr =
            reinterpret_cast<typename simd_t::scalarType *>(tmp3.data());
        auto tmp3vec =
            reinterpret_cast<typename simd_t::vectorType *>(tmp3.data());

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
            auto jacPtr        = jacPtr_init;
            auto dfPtr         = dfPtr_init;
            auto NumElmtGroups = inblock.GetNumElmtGroups();

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                const bool colldir0 = m_expPtr->GetBasis(0)->Collocation();
                const bool colldir1 = m_expPtr->GetBasis(1)->Collocation();
                const bool colldir2 = m_expPtr->GetBasis(2)->Collocation();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < 3; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nmTot, outptr);
                    }

                    StdAlignDerivBaseHex<DEFORMED>(nq0, nq1, nq2, dfPtr, df_tmp,
                                                   NumElmtGroups * nqTot, inptr,
                                                   tmpPtr);
                    SumDerivTensor3DKernel<DEFORMED, simd_t>(
                        nq0, nq1, nq2,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1],
                        (const typename simd_t::vectorType *)tmpPtr[2], W0, W1,
                        W2, jacPtr, D0, D1, D2, tmp3Ptr);
                    IProductHexKernel<false, false, simd_t>(
                        nm0, nm1, nm2, nq0, nq1, nq2, tmp3vec, B0, B1, B2, wsp,
                        wsp0, outptr, 1.0, colldir0, colldir1, colldir2);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                // Get geometric factors.
                auto F0 =
                    m_Fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
                auto F1 =
                    m_Fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
                auto F1a =
                    m_Fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
                auto F2 =
                    m_Fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < 3; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nmTot, outptr);
                    }

                    StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, nq2, dfPtr, df_tmp, NumElmtGroups * nqTot, F0,
                        F1, F1a, F2, inptr, tmpPtr);
                    SumDerivTensor3DKernel<DEFORMED, simd_t>(
                        nq0, nq1, nq2,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1],
                        (const typename simd_t::vectorType *)tmpPtr[2], W0, W1,
                        W2, jacPtr, D0, D1, D2, tmp3Ptr);
                    IProductTetKernel<false, false, simd_t>(
                        nm0, nm1, nm2, nq0, nq1, nq2, isModified, tmp3vec, B0,
                        B1, B2, wsp, wsp0, outptr);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                // Get geometric factors.
                auto F0 =
                    m_Fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
                auto F1 =
                    m_Fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
                auto F2 =
                    m_Fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < 3; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nmTot, outptr);
                    }

                    StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, nq2, dfPtr, df_tmp, NumElmtGroups * nqTot, F0,
                        F1, (simd_t *)nullptr, F2, inptr, tmpPtr);
                    SumDerivTensor3DKernel<DEFORMED, simd_t>(
                        nq0, nq1, nq2,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1],
                        (const typename simd_t::vectorType *)tmpPtr[2], W0, W1,
                        W2, jacPtr, D0, D1, D2, tmp3Ptr);
                    IProductPyrKernel<false, false, simd_t>(
                        nm0, nm1, nm2, nq0, nq1, nq2, isModified, tmp3vec, B0,
                        B1, B2, wsp, wsp0, outptr);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                std::vector<simd_t, tinysimd::allocator<simd_t>> wsp1(nm1);

                // Get geometric factors.
                auto F0 =
                    m_Fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
                auto F2 =
                    m_Fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < 3; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nmTot, outptr);
                    }

                    StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, nq2, dfPtr, df_tmp, NumElmtGroups * nqTot, F0,
                        (simd_t *)nullptr, (simd_t *)nullptr, F2, inptr,
                        tmpPtr);
                    SumDerivTensor3DKernel<DEFORMED, simd_t>(
                        nq0, nq1, nq2,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1],
                        (const typename simd_t::vectorType *)tmpPtr[2], W0, W1,
                        W2, jacPtr, D0, D1, D2, tmp3Ptr);
                    IProductPrismKernel<false, false, simd_t>(
                        nm0, nm1, nm2, nq0, nq1, nq2, isModified, tmp3vec, B0,
                        B1, B2, wsp, wsp0, wsp1, outptr);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }
            // advance input by ncoord-1 componennts since have already
            // advanced one component in the above
            inptr += nqTot * NumElmtGroups * 2;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nm2, int nq0, int nq1, int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto ndf   = 9u;
        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot = m_expPtr->GetNcoeffs();

        size_t jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        [[maybe_unused]] const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Get Basis and weight data.
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
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D2 =
            m_derivativeMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W2 =
            m_weightMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

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

        // Workspace for kernels.
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ndf);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp(nq1 * nq2),
            wsp0(nq2), tmp0(nqTot), tmp1(nqTot), tmp2(nqTot);
        typename simd_t::scalarType *tmpPtr[3];
        tmpPtr[0] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp0.data());
        tmpPtr[1] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp1.data());
        tmpPtr[2] =
            reinterpret_cast<typename simd_t::scalarType *>(tmp2.data());
        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp3(nqTot);
        auto tmp3Ptr =
            reinterpret_cast<typename simd_t::scalarType *>(tmp3.data());
        auto tmp3vec =
            reinterpret_cast<typename simd_t::vectorType *>(tmp3.data());

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
            auto jacPtr        = jacPtr_init;
            auto dfPtr         = dfPtr_init;
            auto NumElmtGroups = inblock.GetNumElmtGroups();

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                const bool colldir0 = m_expPtr->GetBasis(0)->Collocation();
                const bool colldir1 = m_expPtr->GetBasis(1)->Collocation();
                const bool colldir2 = m_expPtr->GetBasis(2)->Collocation();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < 3; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nmTot, outptr);
                    }

                    StdAlignDerivBaseHex<DEFORMED>(nq0, nq1, nq2, dfPtr, df_tmp,
                                                   NumElmtGroups * nqTot, inptr,
                                                   tmpPtr);
                    SumDerivTensor3DKernel<DEFORMED, simd_t>(
                        nq0, nq1, nq2,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1],
                        (const typename simd_t::vectorType *)tmpPtr[2], W0, W1,
                        W2, jacPtr, D0, D1, D2, tmp3Ptr);
                    IProductHexKernel<false, false, simd_t>(
                        nm0, nm1, nm2, nq0, nq1, nq2, tmp3vec, B0, B1, B2, wsp,
                        wsp0, outptr, 1.0, colldir0, colldir1, colldir2);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                // Get geometric factors.
                auto F0 =
                    m_Fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
                auto F1 =
                    m_Fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
                auto F1a =
                    m_Fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
                auto F2 =
                    m_Fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < 3; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nmTot, outptr);
                    }

                    StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, nq2, dfPtr, df_tmp, NumElmtGroups * nqTot, F0,
                        F1, F1a, F2, inptr, tmpPtr);
                    SumDerivTensor3DKernel<DEFORMED, simd_t>(
                        nq0, nq1, nq2,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1],
                        (const typename simd_t::vectorType *)tmpPtr[2], W0, W1,
                        W2, jacPtr, D0, D1, D2, tmp3Ptr);
                    IProductTetKernel<false, false, simd_t>(
                        nm0, nm1, nm2, nq0, nq1, nq2, isModified, tmp3vec, B0,
                        B1, B2, wsp, wsp0, outptr);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                // Get geometric factors.
                auto F0 =
                    m_Fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
                auto F1 =
                    m_Fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
                auto F2 =
                    m_Fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < 3; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nmTot, outptr);
                    }

                    StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, nq2, dfPtr, df_tmp, NumElmtGroups * nqTot, F0,
                        F1, (simd_t *)nullptr, F2, inptr, tmpPtr);
                    SumDerivTensor3DKernel<DEFORMED, simd_t>(
                        nq0, nq1, nq2,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1],
                        (const typename simd_t::vectorType *)tmpPtr[2], W0, W1,
                        W2, jacPtr, D0, D1, D2, tmp3Ptr);
                    IProductPyrKernel<false, false, simd_t>(
                        nm0, nm1, nm2, nq0, nq1, nq2, isModified, tmp3vec, B0,
                        B1, B2, wsp, wsp0, outptr);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                std::vector<simd_t, tinysimd::allocator<simd_t>> wsp1(nm1);

                // Get geometric factors.
                auto F0 =
                    m_Fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
                auto F2 =
                    m_Fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
                for (int e = 0; e < NumElmtGroups; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        for (int n = 0; n < 3; ++n)
                        {
                            ReshapeStorage<ExecSpace, simd_t::width>(
                                in_interleave_width, chunkSize, nqTot,
                                (TData *)(inptr + n * NumElmtGroups * nqTot));
                        }
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            out_interleave_width, chunkSize, nmTot, outptr);
                    }

                    StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                        nq0, nq1, nq2, dfPtr, df_tmp, NumElmtGroups * nqTot, F0,
                        (simd_t *)nullptr, (simd_t *)nullptr, F2, inptr,
                        tmpPtr);
                    SumDerivTensor3DKernel<DEFORMED, simd_t>(
                        nq0, nq1, nq2,
                        (const typename simd_t::vectorType *)tmpPtr[0],
                        (const typename simd_t::vectorType *)tmpPtr[1],
                        (const typename simd_t::vectorType *)tmpPtr[2], W0, W1,
                        W2, jacPtr, D0, D1, D2, tmp3Ptr);
                    IProductPrismKernel<false, false, simd_t>(
                        nm0, nm1, nm2, nq0, nq1, nq2, isModified, tmp3vec, B0,
                        B1, B2, wsp, wsp0, wsp1, outptr);

                    // Increment pointers for the next elmt group.
                    inptr += nqTot;
                    outptr += nmTot * simd_t::width;
                    jacPtr += jacSize;
                    dfPtr += jacSize * ndf;
                }
            }
            // advance input by ncoord-1 componennts since have already
            // advanced one component in the above
            inptr += nqTot * NumElmtGroups * 2;
        }
    }
};

} // namespace Nektar::Operators::detail
