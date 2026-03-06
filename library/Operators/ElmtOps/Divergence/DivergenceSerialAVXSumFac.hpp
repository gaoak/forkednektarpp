///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceSerialAVXSumFac.hpp
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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "Operators/ElmtOps/Divergence/DivergenceBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/Divergence/DivergenceSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class DivergenceBlockOpImpl : public DivergenceBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    DivergenceBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          NekDataWarehouseSharedPtr dataWarehouse)
        : DivergenceBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        auto dimension = exp->GetShapeDimension();
        m_coordDim     = dimension; // required for boost_pp switch

        for (unsigned int d = 0; d < dimension; d++)
        {
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eDerivative)));
        }

        if (dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
        }
        else if (dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
        }

        // Fetch deriv factors data.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            DerivFactorKey<TData>(block_idx, m_implInterleaveWidth, false));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            DivergenceBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_coordDim;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const simd_t *> m_D;
    std::vector<const simd_t *> m_f;
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        WARNINGL1(
            m_warnOnce || (inblock.GetAlignment() % simd_t::alignment == 0 &&
                           outblock.GetAlignment() % simd_t::alignment == 0),
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

    void SegBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void TriBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTriBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                       BlockAccessor<TData, FieldState::Phys> &outblock);

    void QuadBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                   BlockAccessor<TData, FieldState::Phys> &outblock);

    void HexBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void PrismBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalPrismBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                         BlockAccessor<TData, FieldState::Phys> &outblock);

    void PyrBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void TetBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTetBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                       BlockAccessor<TData, FieldState::Phys> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_coordDim,
                                         m_nq[0]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, coordDim, nq0);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator1D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int coordDim, const unsigned int nq0)
    {
        // Shape size.
        const auto nqTot = nq0;

        unsigned int dfsize = coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

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
            auto dfptr = m_dfptr;

            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize, nqTot,
                                              (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor1DKernel(nq0,
                                        reinterpret_cast<const simd_t *>(inptr),
                                        m_D[0], outvec[0]);

                // Calculate physical derivative.
                PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, m_coordDim, reinterpret_cast<const simd_t *>(dfptr),
                    outvec);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)outvec[0] -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                outvec[0] += nqTot;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_coordDim, m_nq[0],
                                         m_nq[1]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0, unsigned int nq1>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, coordDim, nq0, nq1);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator2D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, FieldState::Phys> &outblock,
        [[maybe_unused]] const unsigned int coordDim, const unsigned int nq0,
        const unsigned int nq1)
    {
        // Shape size.
        const auto nqTot = nq0 * nq1;

        ASSERTL1(inblock.GetNumComponents() >= 2,
                 "Input block does not have at least two components");

        unsigned int dfsize = 4u;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // pointer loop over components.
        const auto compOffset =
            outblock.CompSize() * outblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr0 = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto inptr1 = inptr0 + compOffset;
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(nqTot),
            wsp1(nqTot);

        auto dfptr = m_dfptr;
        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {
            // Reshape, if necessary.
            if (e % width_ratio == 0)
            {
                ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                          interleaveWidth, chunkSize, nqTot,
                                          (TData *)inptr0);
                ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                          interleaveWidth, chunkSize, nqTot,
                                          (TData *)inptr1);
            }

            // du/dx
            PhysDerivTensor2DKernel(nq0, nq1,
                                    reinterpret_cast<const simd_t *>(inptr0),
                                    m_D[0], m_D[1], wsp0.data(), wsp1.data());

            // physical derivative.
            PhysDerivDir2DKernel<SHAPE_TYPE, DEFORMED, 0, false>(
                nq0, nq1, m_f[0], m_f[1],
                reinterpret_cast<const simd_t *>(dfptr), wsp0.data(),
                wsp1.data(), reinterpret_cast<simd_t *>(outptr));

            // dv/dy
            PhysDerivTensor2DKernel(nq0, nq1,
                                    reinterpret_cast<const simd_t *>(inptr1),
                                    m_D[0], m_D[1], wsp0.data(), wsp1.data());

            //  physical derivative.
            PhysDerivDir2DKernel<SHAPE_TYPE, DEFORMED, 1, true>(
                nq0, nq1, m_f[0], m_f[1],
                reinterpret_cast<const simd_t *>(dfptr), wsp0.data(),
                wsp1.data(), reinterpret_cast<simd_t *>(outptr));

            // Reshape back, if necessary.
            if (e % width_ratio == width_ratio - 1)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr0 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr1 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)outptr -
                        (width_ratio - 1) * nqTot * simd_t::width);
            }

            // Increment pointers for the next elmt group.
            dfptr += dfsize * simd_t::width;
            inptr0 += nqTot * simd_t::width;
            inptr1 += nqTot * simd_t::width;
            outptr += nqTot * simd_t::width;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nq[0], m_nq[1],
                                         m_nq[2]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nq0, nq1, nq2);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator3D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int nq0, const unsigned int nq1, const unsigned int nq2)
    {
        // Shape size.
        const auto nqTot = nq0 * nq1 * nq2;

        ASSERTL1(inblock.GetNumComponents() >= 3,
                 "Input block does not have at least two components");

        unsigned int dfsize = 9u;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Loop over components.
        const auto compOffset =
            outblock.CompSize() * outblock.GetNumHomoModes();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Initialize pointers.
        auto inptr0 = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto inptr1 = inptr0 + compOffset;
        auto inptr2 = inptr1 + compOffset;
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(nqTot),
            wsp1(nqTot), wsp2(nqTot);

        auto dfptr = m_dfptr;

        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {
            // Reshape, if necessary.
            if (e % width_ratio == 0)
            {
                ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                          interleaveWidth, chunkSize, nqTot,
                                          (TData *)inptr0);
                ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                          interleaveWidth, chunkSize, nqTot,
                                          (TData *)inptr1);
                ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                          interleaveWidth, chunkSize, nqTot,
                                          (TData *)inptr2);
            }

            // Get the basic derivative.
            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr0), m_D[0],
                m_D[1], m_D[2], wsp0.data(), wsp1.data(), wsp2.data());

            // du/dx
            // Calculate physical derivative.
            PhysDerivDir3DKernel<SHAPE_TYPE, DEFORMED, 0, false>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), wsp0.data(),
                wsp1.data(), wsp2.data(), reinterpret_cast<simd_t *>(outptr));

            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr1), m_D[0],
                m_D[1], m_D[2], wsp0.data(), wsp1.data(), wsp2.data());

            // dv/dy
            // Calculate physical derivative.
            PhysDerivDir3DKernel<SHAPE_TYPE, DEFORMED, 1, true>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), wsp0.data(),
                wsp1.data(), wsp2.data(), reinterpret_cast<simd_t *>(outptr));

            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr2), m_D[0],
                m_D[1], m_D[2], wsp0.data(), wsp1.data(), wsp2.data());

            // dw/dz
            // Calculate physical derivative.
            PhysDerivDir3DKernel<SHAPE_TYPE, DEFORMED, 2, true>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), wsp0.data(),
                wsp1.data(), wsp2.data(), reinterpret_cast<simd_t *>(outptr));

            // Reshape back, if necessary.
            if (e % width_ratio == width_ratio - 1)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr0 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr1 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr2 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)outptr -
                        (width_ratio - 1) * nqTot * simd_t::width);
            }

            // Increment pointers for the next elmt group.
            dfptr += dfsize * simd_t::width;
            inptr0 += nqTot * simd_t::width;
            inptr1 += nqTot * simd_t::width;
            inptr2 += nqTot * simd_t::width;
            outptr += nqTot * simd_t::width;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
