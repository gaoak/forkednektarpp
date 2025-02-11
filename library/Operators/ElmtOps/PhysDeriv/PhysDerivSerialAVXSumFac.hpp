///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSerialAVXSumFac.hpp
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

#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorPhysDerivImpl : public OperatorPhysDeriv<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL0(this->m_expansionList->GetExp(0)->GetCoordim() <=
                     out.GetNumComponents(),
                 "Output field has fewer components than the coordinate!");

        m_nComps = in.GetNumComponents();
        ASSERTL1(m_nComps == out.GetNumComponents() /
                                 this->m_expansionList->GetCoordim(0),
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
            OperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
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
        const auto nq0     = m_expPtr->GetNumPoints(0);
        const auto nqTot   = nq0;
        const auto nqBlock = nqTot * simd_t::width;

        const auto nCoord = m_expPtr->GetCoordim();
        const auto ndf    = nCoord;
        int dfsize        = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int in_interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            in_interleave_width == 1 ? 1 : in_interleave_width / simd_t::width;
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

        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        for (int d = 0; d < nCoord; ++d)
        {
            outptr[d] = reinterpret_cast<typename simd_t::scalarType *>(
                output + d * compOffset);
        }

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto dfptr = dfptr_init;

            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor1DKernel(nq0, inptr, D0, outptr[0]);

                // Calculate physical derivative.
                PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(nq0, nCoord, dfptr,
                                                        outptr);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                for (int d = 0; d < nCoord; ++d)
                {
                    outptr[d] += nqBlock;
                }
            }

            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            for (int d = 0; d < nCoord; ++d) // reset to next output components
            {
                outptr[d] += (nCoord - 1) * compOffset;
            }
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nCoord, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot   = nq0;
        constexpr auto nqBlock = nqTot * simd_t::width;

        constexpr auto ndf = nCoord;
        int dfsize         = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int in_interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            in_interleave_width == 1 ? 1 : in_interleave_width / simd_t::width;
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

        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        for (int d = 0; d < nCoord; ++d)
        {
            outptr[d] = reinterpret_cast<typename simd_t::scalarType *>(
                output + d * compOffset);
        }

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor1DKernel(nq0, inptr, D0, outptr[0]);

                // Calculate physical derivative.
                PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(nq0, nCoord, dfptr,
                                                        outptr);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                for (int d = 0; d < nCoord; ++d) // automatically unrolled
                {
                    outptr[d] += nqBlock;
                }
            }

            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            for (int d = 0; d < nCoord; ++d) // reset to next output components
            {
                outptr[d] += (nCoord - 1) * compOffset;
            }
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);

        const auto nqTot   = nq0 * nq1;
        const auto nqBlock = nqTot * simd_t::width;

        const auto nCoord = m_expPtr->GetCoordim();
        const auto ndf    = 2 * nCoord;
        int dfsize        = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eDerivative));
        auto Z0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eZeros));
        auto Z1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(), eZeros));

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int in_interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            in_interleave_width == 1 ? 1 : in_interleave_width / simd_t::width;
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
        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        for (int d = 0; d < nCoord; ++d)
        {
            outptr[d] = reinterpret_cast<typename simd_t::scalarType *>(
                output + d * compOffset);
        }

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Results written to outptr0, outptr1.
                PhysDerivTensor2DKernel(nq0, nq1, inptr, D0, D1, outptr[0],
                                        outptr[1]);

                // Calculate physical derivative.
                PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(nq0, nq1, nCoord, Z0,
                                                        Z1, dfptr, outptr);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                for (int d = 0; d < nCoord; ++d) // automatically unrolled
                {
                    outptr[d] += nqBlock;
                }
            }

            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            for (int d = 0; d < nCoord; ++d) // reset to next output components
            {
                outptr[d] += (nCoord - 1) * compOffset;
            }
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nCoord, unsigned int nq0, unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot   = nq0 * nq1;
        constexpr auto nqBlock = nqTot * simd_t::width;

        constexpr auto ndf = 2 * nCoord;
        int dfsize         = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eDerivative));
        auto Z0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eZeros));
        auto Z1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(), eZeros));

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int in_interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            in_interleave_width == 1 ? 1 : in_interleave_width / simd_t::width;
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

        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        for (int d = 0; d < nCoord; ++d)
        {
            outptr[d] = reinterpret_cast<typename simd_t::scalarType *>(
                output + d * compOffset);
        }

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Results written to outptr0, outptr1.
                PhysDerivTensor2DKernel(nq0, nq1, inptr, D0, D1, outptr[0],
                                        outptr[1]);

                // Calculate physical derivative.
                PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(nq0, nq1, nCoord, Z0,
                                                        Z1, dfptr, outptr);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                for (int d = 0; d < nCoord; ++d) // automatically unrolled
                {
                    outptr[d] += nqBlock;
                }
            }

            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            for (int d = 0; d < nCoord; ++d) // reset to next output components
            {
                outptr[d] += (nCoord - 1) * compOffset;
            }
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);
        const auto nq2 = m_expPtr->GetNumPoints(2);

        const auto nqTot    = nq0 * nq1 * nq2;
        const auto nqBlocks = nqTot * simd_t::width;

        constexpr auto ndf = 9u;
        int dfsize         = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eDerivative));
        auto D2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                 eDerivative));
        auto Z0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eZeros));
        auto Z1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(), eZeros));
        auto Z2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(), eZeros));

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int in_interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            in_interleave_width == 1 ? 1 : in_interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, in_interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0;
        PhysDeriv3DWorkspace<SHAPE_TYPE>(nq0, nq1, nq2, wsp0Size, wsp1Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);

        // Initialize pointers.
        auto input  = (in_interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);

        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        outptr[0] = reinterpret_cast<typename simd_t::scalarType *>(output);

        outptr[1] = reinterpret_cast<typename simd_t::scalarType *>(output +
                                                                    compOffset);
        outptr[2] = reinterpret_cast<typename simd_t::scalarType *>(
            output + 2 * compOffset);

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor3DKernel(nq0, nq1, nq2, inptr, D0, D1, D2,
                                        outptr[0], outptr[1], outptr[2]);

                // Calculate physical derivative.
                PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, Z0, Z1, Z2, dfptr, wsp0, wsp1, outptr[0],
                    outptr[1], outptr[2]);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                outptr[0] += nqBlocks;
                outptr[1] += nqBlocks;
                outptr[2] += nqBlocks;
            }

            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            outptr[0] += 2 * compOffset;
            outptr[1] += 2 * compOffset;
            outptr[2] += 2 * compOffset;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot    = nq0 * nq1 * nq2;
        constexpr auto nqBlocks = nqTot * simd_t::width;

        constexpr auto ndf = 9u;
        int dfsize         = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch basis data.
        auto D0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(),
                                 eDerivative));
        auto D1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(),
                                 eDerivative));
        auto D2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(),
                                 eDerivative));
        auto Z0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(0)->GetBasisKey(), eZeros));
        auto Z1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(1)->GetBasisKey(), eZeros));
        auto Z2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(m_expPtr->GetBasis(2)->GetBasisKey(), eZeros));

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(m_exp_idx, simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        unsigned int in_interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            in_interleave_width == 1 ? 1 : in_interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, in_interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0;
        PhysDeriv3DWorkspace<SHAPE_TYPE>(nq0, nq1, nq2, wsp0Size, wsp1Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);

        // Initialize pointers.
        auto input  = (in_interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);

        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        outptr[0] = reinterpret_cast<typename simd_t::scalarType *>(output);

        outptr[1] = reinterpret_cast<typename simd_t::scalarType *>(output +
                                                                    compOffset);
        outptr[2] = reinterpret_cast<typename simd_t::scalarType *>(
            output + 2 * compOffset);

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        in_interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor3DKernel(nq0, nq1, nq2, inptr, D0, D1, D2,
                                        outptr[0], outptr[1], outptr[2]);

                // Calculate physical derivative.
                PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, Z0, Z1, Z2, dfptr, wsp0, wsp1, outptr[0],
                    outptr[1], outptr[2]);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                outptr[0] += nqBlocks;
                outptr[1] += nqBlocks;
                outptr[2] += nqBlocks;
            }
            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            outptr[0] += 2 * compOffset;
            outptr[1] += 2 * compOffset;
            outptr[2] += 2 * compOffset;
        }
    }
};

} // namespace Nektar::Operators::detail
