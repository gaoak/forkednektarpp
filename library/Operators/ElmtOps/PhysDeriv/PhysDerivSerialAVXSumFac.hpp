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
class BlockOperatorPhysDerivImpl : public BlockOperatorPhysDeriv<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorPhysDerivImpl(const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorPhysDeriv<TData>(exp, dataWarehouse)
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
            m_D.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eDerivative)));
            m_Z.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(), eZeros)));
        }
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
            BlockOperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
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
    std::vector<const simd_t *> m_D;
    std::vector<const simd_t *> m_Z;

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
        const auto nq0     = m_nq[0];
        const auto nqTot   = nq0;
        const auto nqBlock = nqTot * simd_t::width;

        unsigned int dfsize = m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(),
                                      m_implInterleaveWidth,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio              = (interleave_width == 1)
                                                  ? 1
                                                  : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);

        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        for (unsigned int d = 0; d < m_coordDim; ++d)
        {
            outptr[d] = reinterpret_cast<typename simd_t::scalarType *>(
                output + d * compOffset);
        }

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto dfptr = dfptr_init;

            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor1DKernel(nq0, inptr, m_D[0], outptr[0]);

                // Calculate physical derivative.
                PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(nq0, m_coordDim, dfptr,
                                                        outptr);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                for (unsigned int d = 0; d < m_coordDim; ++d)
                {
                    outptr[d] += nqBlock;
                }
            }

            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                outptr[d] += (m_coordDim - 1) * compOffset;
            }
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot   = nq0;
        constexpr auto nqBlock = nqTot * simd_t::width;

        unsigned int dfsize = m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(),
                                      m_implInterleaveWidth,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio              = (interleave_width == 1)
                                                  ? 1
                                                  : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);

        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        for (unsigned int d = 0; d < coordDim; ++d)
        {
            outptr[d] = reinterpret_cast<typename simd_t::scalarType *>(
                output + d * compOffset);
        }

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor1DKernel(nq0, inptr, m_D[0], outptr[0]);

                // Calculate physical derivative.
                PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(nq0, coordDim, dfptr,
                                                        outptr);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                for (unsigned int d = 0; d < coordDim; ++d)
                {
                    outptr[d] += nqBlock;
                }
            }

            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            for (unsigned int d = 0; d < coordDim; ++d)
            {
                outptr[d] += (coordDim - 1) * compOffset;
            }
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nqTot   = nq0 * nq1;
        const auto nqBlock = nqTot * simd_t::width;

        unsigned int dfsize = 2 * m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(),
                                      m_implInterleaveWidth,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio              = (interleave_width == 1)
                                                  ? 1
                                                  : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        for (unsigned int d = 0; d < m_coordDim; ++d)
        {
            outptr[d] = reinterpret_cast<typename simd_t::scalarType *>(
                output + d * compOffset);
        }

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Results written to outptr0, outptr1.
                PhysDerivTensor2DKernel(nq0, nq1, inptr, m_D[0], m_D[1],
                                        outptr[0], outptr[1]);

                // Calculate physical derivative.
                PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, m_coordDim, m_Z[0], m_Z[1], dfptr, outptr);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                for (unsigned int d = 0; d < m_coordDim; ++d)
                {
                    outptr[d] += nqBlock;
                }
            }

            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            for (unsigned int d = 0; d < m_coordDim;
                 ++d) // reset to next output components
            {
                outptr[d] += (m_coordDim - 1) * compOffset;
            }
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0, unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot   = nq0 * nq1;
        constexpr auto nqBlock = nqTot * simd_t::width;

        unsigned int dfsize = 2 * m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(),
                                      m_implInterleaveWidth,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio              = (interleave_width == 1)
                                                  ? 1
                                                  : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);

        auto compOffset = outblock.GetNumElmtGroups() * simd_t::width * nqTot;
        typename simd_t::scalarType *outptr[3];
        for (unsigned int d = 0; d < m_coordDim; ++d)
        {
            outptr[d] = reinterpret_cast<typename simd_t::scalarType *>(
                output + d * compOffset);
        }

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Results written to outptr0, outptr1.
                PhysDerivTensor2DKernel(nq0, nq1, inptr, m_D[0], m_D[1],
                                        outptr[0], outptr[1]);

                // Calculate physical derivative.
                PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, coordDim, m_Z[0], m_Z[1], dfptr, outptr);

                // Increment pointers for the next elmt group.
                dfptr += dfsize;
                inptr += nqTot;
                for (unsigned int d = 0; d < coordDim; ++d)
                {
                    outptr[d] += nqBlock;
                }
            }

            // advance  by ncoord-1 componennts since have already
            // advanced one component in the above
            for (unsigned int d = 0; d < coordDim; ++d)
            {
                outptr[d] += (coordDim - 1) * compOffset;
            }
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];
        const auto nq2 = m_nq[2];

        const auto nqTot    = nq0 * nq1 * nq2;
        const auto nqBlocks = nqTot * simd_t::width;

        unsigned int dfsize = 9u;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(),
                                      m_implInterleaveWidth,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio              = (interleave_width == 1)
                                                  ? 1
                                                  : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0;
        PhysDeriv3DWorkspace<SHAPE_TYPE>(nq0, nq1, nq2, wsp0Size, wsp1Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
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
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor3DKernel(nq0, nq1, nq2, inptr, m_D[0], m_D[1],
                                        m_D[2], outptr[0], outptr[1],
                                        outptr[2]);

                // Calculate physical derivative.
                PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, m_Z[0], m_Z[1], m_Z[2], dfptr, wsp0, wsp1,
                    outptr[0], outptr[1], outptr[2]);

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

        unsigned int dfsize = 9u;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(),
                                      m_implInterleaveWidth,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio              = (interleave_width == 1)
                                                  ? 1
                                                  : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0;
        PhysDeriv3DWorkspace<SHAPE_TYPE>(nq0, nq1, nq2, wsp0Size, wsp1Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
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
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto dfptr = dfptr_init;
            for (unsigned int e = 0; e < outblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor3DKernel(nq0, nq1, nq2, inptr, m_D[0], m_D[1],
                                        m_D[2], outptr[0], outptr[1],
                                        outptr[2]);

                // Calculate physical derivative.
                PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, m_Z[0], m_Z[1], m_Z[2], dfptr, wsp0, wsp1,
                    outptr[0], outptr[1], outptr[2]);

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
