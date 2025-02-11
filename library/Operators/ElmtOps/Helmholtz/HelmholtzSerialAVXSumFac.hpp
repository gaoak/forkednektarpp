///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzSerialAVXSumFac.hpp
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

#include "Operators/ElmtOps/OperatorHelmholtz.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/Helmholtz/HelmholtzSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorHelmholtzImpl : public OperatorHelmholtz<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtz<TData>(expansionList)
    {
        auto nCoord = this->m_expansionList->GetCoordim(0);
        m_diffCoeff = std::vector<TData>(nCoord * (nCoord + 1) / 2, 0.0);

        // Set up temprary solution.
        m_diffCoeff[0] = 1.0; // D00
        if (nCoord >= 2)
        {
            m_diffCoeff[2] = 1.0; // D11
            if (nCoord == 3)
            {
                m_diffCoeff[5] = 1.0; // D22
            }
        }
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

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorHelmholtzImpl<ExecSpace, Implementation, TData>>(
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

protected:
    unsigned int m_exp_idx;
    unsigned int m_blk;
    unsigned int m_nComps;

    LocalRegions::ExpansionSharedPtr m_expPtr;

    std::vector<TData> m_diffCoeff;

    std::vector<TData> NullTDataVector;

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

        const auto nqTot = nq0;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0);

        constexpr auto ndf = 1;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Get Basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto DB0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eBasisDerivative));
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(m_exp_idx, simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Allocate workspace.
        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, B0, inptr, bwd);
                // Step 2: Inner product for mass matrix operation.
                IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                    nm0, nq0, bwdvec, B0, W0, jacptr, outptr, this->m_lambda);
                // Step 3: Take derivatives in collapsed coordinate space.
                PhysDerivTensor1DKernel(nq0, bwdvec, D0, deriv0);
                // Step 4: Apply diffusion coefficiets.
                DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                    nq0, true, this->m_diffCoeff, false, NullTDataVector, dfptr,
                    deriv0);
                // Step 5: Apply Laplacian metrics & inner product.
                IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                    nm0, nq0, deriv0vec, DB0, W0, jacptr, outptr);
                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot = nq0;
        constexpr auto nmTot = nm0;

        constexpr auto ndf = 1;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Get Basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto DB0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eBasisDerivative));
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(m_exp_idx, simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Allocate workspace.
        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, B0, inptr, bwd);
                // Step 2: Inner product for mass matrix operation.
                IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                    nm0, nq0, bwdvec, B0, W0, jacptr, outptr, this->m_lambda);
                // Step 3: Take derivatives in collapsed coordinate space.
                PhysDerivTensor1DKernel(nq0, bwdvec, D0, deriv0);
                // Step 4: Apply diffusion coefficiets.
                DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                    nq0, true, this->m_diffCoeff, false, NullTDataVector, dfptr,
                    deriv0);
                // Step 5: Apply Laplacian metrics & inner product.
                IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                    nm0, nq0, deriv0vec, DB0, W0, jacptr, outptr);
                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
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

        constexpr auto ndf = 4;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Get Basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(), eBasis));
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eDerivative));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(m_exp_idx, simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        const simd_t *h0 = nullptr, *h1 = nullptr;
        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            h0 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h1 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, isModified, B0,
                                             B1, wsp0, inptr, bwd);
                // Step 2 + 3: Get tensor derivative and apply diffusion coeff
                TensorDerivWithDiffuCoeff2DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, true, this->m_diffCoeff, false, NullTDataVector,
                    NullTDataVector, NullTDataVector, bwdvec, D0, D1, dfptr, h0,
                    h1, deriv0, deriv1);
                // Step 4: apply WJ, derivative and sum up
                SumDerivTensor2DKernel<DEFORMED, simd_t>(
                    nq0, nq1, deriv0vec, deriv1vec, W0, W1, jacptr, D0, D1, bwd,
                    this->m_lambda);
                // Step 5 : inner product without WJ
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, isModified, bwdvec, B0, B1, wsp0,
                    outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot = nq0 * nq1;
        constexpr auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

        constexpr auto ndf = 4;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Get Basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(), eBasis));
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eDerivative));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(m_exp_idx, simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        const simd_t *h0 = nullptr, *h1 = nullptr;
        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            h0 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h1 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        alignas(simd_t::alignment) TData bwd[nqTot * simd_t::width];
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        alignas(simd_t::alignment) TData deriv0[nqTot * simd_t::width];
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        alignas(simd_t::alignment) TData deriv1[nqTot * simd_t::width];
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, isModified, B0,
                                             B1, wsp0, inptr, bwd);
                // Step 2 + 3: Get tensor derivative and apply diffusion coeff
                TensorDerivWithDiffuCoeff2DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, true, this->m_diffCoeff, false, NullTDataVector,
                    NullTDataVector, NullTDataVector, bwdvec, D0, D1, dfptr, h0,
                    h1, deriv0, deriv1);
                // Step 4: apply WJ, derivative and sum up
                SumDerivTensor2DKernel<DEFORMED, simd_t>(
                    nq0, nq1, deriv0vec, deriv1vec, W0, W1, jacptr, D0, D1, bwd,
                    this->m_lambda);
                // Step 5 : inner product without WJ
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, isModified, bwdvec, B0, B1, wsp0,
                    outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
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

        constexpr auto ndf = 9;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Get Basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(), eBasis));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(), eBasis));
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eDerivative));
        auto D2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                 eDerivative));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eWeights));
        auto W2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(m_exp_idx, simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        // get geometric factors
        const simd_t *h0 = nullptr, *h1 = nullptr, *h2 = nullptr, *h3 = nullptr;
        if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
        {
            h0 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h1 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h2 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
            h3 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
        {
            h0 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h1 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
        {
            h0 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h1 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h2 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);
        auto deriv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv2);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2,
                                             isModified, B0, B1, B2, wsp0, wsp1,
                                             inptr, bwd);
                // Step 2 + 3 : Get tensor derivative and apply diffusion coeff
                TensorDerivWithDiffuCoeff3DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, nq2, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector, bwdvec,
                    D0, D1, D2, dfptr, h0, h1, h2, h3, deriv0, deriv1, deriv2);
                // Step 4: apply WJ, derivative and sum up
                SumDerivTensor3DKernel<DEFORMED, simd_t>(
                    nq0, nq1, nq2, deriv0vec, deriv1vec, deriv2vec, W0, W1, W2,
                    jacptr, D0, D1, D2, bwd, this->m_lambda);
                // Step 5 : inner product without WJ
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, isModified, bwdvec, B0, B1,
                    B2, wsp0, wsp1, wsp2, outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv2, std::align_val_t(simd_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot = nq0 * nq1 * nq2;
        constexpr auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        constexpr auto ndf = 9;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Get Basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(), eBasis));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(), eBasis));
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eDerivative));
        auto D2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                 eDerivative));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eWeights));
        auto W2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(m_exp_idx, simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        // Get geometric factors.
        const simd_t *h0 = nullptr, *h1 = nullptr, *h2 = nullptr, *h3 = nullptr;
        if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
        {
            h0 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h1 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h2 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
            h3 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
        {
            h0 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h1 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
        {
            h0 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h1 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                     eHalfMultOnePlusZero));
            h2 = this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero));
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        alignas(simd_t::alignment) TData bwd[nqTot * simd_t::width];
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        alignas(simd_t::alignment) TData deriv0[nqTot * simd_t::width];
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        alignas(simd_t::alignment) TData deriv1[nqTot * simd_t::width];
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        alignas(simd_t::alignment) TData deriv2[nqTot * simd_t::width];
        auto deriv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv2);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2,
                                             isModified, B0, B1, B2, wsp0, wsp1,
                                             inptr, bwd);
                // Step 2 + 3 : Get tensor derivative and apply diffusion coeff
                TensorDerivWithDiffuCoeff3DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, nq2, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector, bwdvec,
                    D0, D1, D2, dfptr, h0, h1, h2, h3, deriv0, deriv1, deriv2);
                // Step 4: apply WJ, derivative and sum up
                SumDerivTensor3DKernel<DEFORMED, simd_t>(
                    nq0, nq1, nq2, deriv0vec, deriv1vec, deriv2vec, W0, W1, W2,
                    jacptr, D0, D1, D2, bwd, this->m_lambda);
                // Step 5 : inner product without WJ
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, isModified, bwdvec, B0, B1,
                    B2, wsp0, wsp1, wsp2, outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }
    }
};

} // namespace Nektar::Operators::detail
