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

#include "Operators/ElmtOps/PhysDeriv/PhysDerivOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class PhysDerivBlockOpImpl : public PhysDerivBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    PhysDerivBlockOpImpl(const LocalRegions::ExpansionSharedPtr &exp,
                         NekDataWarehouseSharedPtr dataWarehouse)
        : PhysDerivBlockOp<TData>(exp, dataWarehouse)
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

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            PhysDerivBlockOpImpl<ExecSpace, Implementation, TData>>(
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
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        WARNINGL1(
            m_warnOnce || (inblock.GetAlignment() == simd_t::alignment &&
                           outblock.GetAlignment() == simd_t::alignment),
            "Input or output Field are not aligned to the required alignment "
            "for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif

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
            // NodalTet
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
            // NodalPrism
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
        const auto nq0   = m_nq[0];
        const auto nqTot = nq0;

        unsigned int dfsize = m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        const auto compOffset =
            outblock.GetNumElmtGroups(m_implInterleaveWidth) * nqTot *
            outblock.GetNumHomoModes();
        simd_t *outvec[3];
        for (unsigned int d = 0; d < m_coordDim; ++d)
        {
            outvec[d] = reinterpret_cast<simd_t *>(outptr) + d * compOffset;
        }
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto dfptr = dfptr_init;

            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleave_width, chunkSize,
                                              nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor1DKernel(nq0,
                                        reinterpret_cast<const simd_t *>(inptr),
                                        m_D[0], outvec[0]);

                // Calculate physical derivative.
                PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, m_coordDim, reinterpret_cast<const simd_t *>(dfptr),
                    outvec);

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                for (unsigned int d = 0; d < m_coordDim; ++d)
                {
                    outvec[d] += nqTot;
                }
            }

            // Advance  by ncoord-1 componennts since have already
            // advanced one component in the above.
            if ((n + 1) % outblock.GetNumHomoModes() == 0)
            {
                for (unsigned int d = 0; d < m_coordDim; ++d)
                {
                    outvec[d] += (m_coordDim - 1) * compOffset;
                }
            }
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot = nq0;

        unsigned int dfsize = coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        const auto compOffset =
            outblock.GetNumElmtGroups(m_implInterleaveWidth) * nqTot *
            outblock.GetNumHomoModes();
        simd_t *outvec[3];
        for (unsigned int d = 0; d < coordDim; ++d)
        {
            outvec[d] = reinterpret_cast<simd_t *>(outptr) + d * compOffset;
        }
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto dfptr = dfptr_init;
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleave_width, chunkSize,
                                              nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor1DKernel(nq0,
                                        reinterpret_cast<const simd_t *>(inptr),
                                        m_D[0], outvec[0]);

                // Calculate physical derivative.
                PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, coordDim, reinterpret_cast<const simd_t *>(dfptr),
                    outvec);

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                for (unsigned int d = 0; d < coordDim; ++d)
                {
                    outvec[d] += nqTot;
                }
            }

            // Advance  by ncoord-1 componennts since have already
            // advanced one component in the above.
            if ((n + 1) % outblock.GetNumHomoModes() == 0)
            {
                for (unsigned int d = 0; d < m_coordDim; ++d)
                {
                    outvec[d] += (m_coordDim - 1) * compOffset;
                }
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
        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nqTot = nq0 * nq1;

        unsigned int dfsize = 2 * m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        const auto compOffset =
            outblock.GetNumElmtGroups(m_implInterleaveWidth) * nqTot *
            outblock.GetNumHomoModes();
        simd_t *outvec[3];
        for (unsigned int d = 0; d < m_coordDim; ++d)
        {
            outvec[d] = reinterpret_cast<simd_t *>(outptr) + d * compOffset;
        }
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto dfptr = dfptr_init;
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleave_width, chunkSize,
                                              nqTot, (TData *)inptr);
                }

                // Results written to outvec0, outvec1.
                PhysDerivTensor2DKernel(nq0, nq1,
                                        reinterpret_cast<const simd_t *>(inptr),
                                        m_D[0], m_D[1], outvec[0], outvec[1]);

                // Calculate physical derivative.
                PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, m_coordDim, m_Z[0], m_Z[1],
                    reinterpret_cast<const simd_t *>(dfptr), outvec);

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                for (unsigned int d = 0; d < m_coordDim; ++d)
                {
                    outvec[d] += nqTot;
                }
            }

            // Advance  by ncoord-1 componennts since have already
            // advanced one component in the above.
            if ((n + 1) % outblock.GetNumHomoModes() == 0)
            {
                for (unsigned int d = 0; d < m_coordDim; ++d)
                {
                    outvec[d] += (m_coordDim - 1) * compOffset;
                }
            }
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0, unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot = nq0 * nq1;

        unsigned int dfsize = 2 * coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Loop over components.
        const auto compOffset =
            outblock.GetNumElmtGroups(m_implInterleaveWidth) * nqTot *
            outblock.GetNumHomoModes();
        simd_t *outvec[3];
        for (unsigned int d = 0; d < coordDim; ++d)
        {
            outvec[d] = reinterpret_cast<simd_t *>(outptr) + d * compOffset;
        }
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto dfptr = dfptr_init;
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleave_width, chunkSize,
                                              nqTot, (TData *)inptr);
                }

                // Results written to outvec0, outvec1.
                PhysDerivTensor2DKernel(nq0, nq1,
                                        reinterpret_cast<const simd_t *>(inptr),
                                        m_D[0], m_D[1], outvec[0], outvec[1]);

                // Calculate physical derivative.
                PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, coordDim, m_Z[0], m_Z[1],
                    reinterpret_cast<const simd_t *>(dfptr), outvec);

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                for (unsigned int d = 0; d < coordDim; ++d)
                {
                    outvec[d] += nqTot;
                }
            }

            // Advance  by ncoord-1 componennts since have already
            // advanced one component in the above.
            if ((n + 1) % outblock.GetNumHomoModes() == 0)
            {
                for (unsigned int d = 0; d < coordDim; ++d)
                {
                    outvec[d] += (coordDim - 1) * compOffset;
                }
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
        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];
        const auto nq2 = m_nq[2];

        const auto nqTot = nq0 * nq1 * nq2;

        unsigned int dfsize = 9u;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0;
        PhysDeriv3DWorkspace<SHAPE_TYPE>(nq0, nq1, nq2, wsp0Size, wsp1Size);
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
        const auto compOffset =
            inblock.GetNumElmtGroups(m_implInterleaveWidth) * nqTot;
        simd_t *outvec[3];
        outvec[0] = reinterpret_cast<simd_t *>(outptr);
        outvec[1] = reinterpret_cast<simd_t *>(outptr) + compOffset;
        outvec[2] = reinterpret_cast<simd_t *>(outptr) + 2 * compOffset;
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            auto dfptr = dfptr_init;
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleave_width, chunkSize,
                                              nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor3DKernel(
                    nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr),
                    m_D[0], m_D[1], m_D[2], outvec[0], outvec[1], outvec[2]);

                // Calculate physical derivative.
                PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, m_Z[0], m_Z[1], m_Z[2],
                    reinterpret_cast<const simd_t *>(dfptr), wsp0.data(),
                    wsp1.data(), outvec[0], outvec[1], outvec[2]);

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                outvec[0] += nqTot;
                outvec[1] += nqTot;
                outvec[2] += nqTot;
            }

            // Advance  by ncoord-1 componennts since have already
            // advanced one component in the above.
            outvec[0] += 2 * compOffset;
            outvec[1] += 2 * compOffset;
            outvec[2] += 2 * compOffset;
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nqTot = nq0 * nq1 * nq2;

        unsigned int dfsize = 9u;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Fetch deriv factors data.
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0;
        PhysDeriv3DWorkspace<SHAPE_TYPE>(nq0, nq1, nq2, wsp0Size, wsp1Size);
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
        const auto compOffset =
            inblock.GetNumElmtGroups(m_implInterleaveWidth) * nqTot;
        simd_t *outvec[3];
        outvec[0] = reinterpret_cast<simd_t *>(outptr);
        outvec[1] = reinterpret_cast<simd_t *>(outptr) + compOffset;
        outvec[2] = reinterpret_cast<simd_t *>(outptr) + 2 * compOffset;
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            auto dfptr = dfptr_init;
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleave_width, chunkSize,
                                              nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor3DKernel(
                    nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr),
                    m_D[0], m_D[1], m_D[2], outvec[0], outvec[1], outvec[2]);

                // Calculate physical derivative.
                PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, m_Z[0], m_Z[1], m_Z[2],
                    reinterpret_cast<const simd_t *>(dfptr), wsp0.data(),
                    wsp1.data(), outvec[0], outvec[1], outvec[2]);

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                outvec[0] += nqTot;
                outvec[1] += nqTot;
                outvec[2] += nqTot;
            }

            // Advance  by ncoord-1 componennts since have already
            // advanced one component in the above.
            outvec[0] += 2 * compOffset;
            outvec[1] += 2 * compOffset;
            outvec[2] += 2 * compOffset;
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
