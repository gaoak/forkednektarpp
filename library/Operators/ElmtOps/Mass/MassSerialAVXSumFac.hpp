///////////////////////////////////////////////////////////////////////////////
//
// File: MassSerialAVXSumFac.hpp
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
//      Mass operator consisting of a fused BwdTrans and IProductWRTBase
//      operators.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/ElmtOps/OperatorMass.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/Mass/MassSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorMassImpl : public BlockOperatorMass<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
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
        // Check alignment.
        WARNINGL1(inblock.GetAlignment() == simd_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(outblock.GetAlignment() == simd_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

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

        const auto nmTot = nm0;
        const auto nqTot = nq0;

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Allocate workspace.
        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr = jacptr_init;
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
                IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nq0, bwdvec, B0, W0, jacptr, outptr);

                jacptr += jacSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
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

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Allocate workspace.
        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr = jacptr_init;
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
                IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nq0, bwdvec, B0, W0, jacptr, outptr);

                jacptr += jacSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
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

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const auto nqTot = nq0 * nq1;

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (this->m_exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr = jacptr_init;
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
                // Step 2: Inner product for mass matrix operation.
                IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nm1, nq0, nq1, isModified, bwdvec, B0, B1, W0, W1,
                    jacptr, wsp0, outptr);

                jacptr += jacSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
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

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (this->m_exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr = jacptr_init;
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
                // Step 2: Inner product for mass matrix operation.
                IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nm1, nq0, nq1, isModified, bwdvec, B0, B1, W0, W1,
                    jacptr, wsp0, outptr);

                jacptr += jacSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
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

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const auto nqTot = nq0 * nq1 * nq2;

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (this->m_exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eBasis));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                 eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eWeights));
        auto W2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr = jacptr_init;
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
                // Step 2: Inner product for mass matrix operation.
                IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nm1, nm2, nq0, nq1, nq2, isModified, bwdvec, B0, B1,
                    B2, W0, W1, W2, jacptr, wsp0, wsp1, wsp2, outptr);

                jacptr += jacSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
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

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (this->m_exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eBasis));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eBasis));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                 eBasis));
        auto W0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eWeights));
        auto W1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eWeights));
        auto W2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                 eWeights));

        // Fetch Jacobian.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr = jacptr_init;
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
                // Step 2: Inner product for mass matrix operation.
                IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                    nm0, nm1, nm2, nq0, nq1, nq2, isModified, bwdvec, B0, B1,
                    B2, W0, W1, W2, jacptr, wsp0, wsp1, wsp2, outptr);

                jacptr += jacSize;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
    }
};

} // namespace Nektar::Operators::detail
