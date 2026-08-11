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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransBlockOp.hpp"

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
    BwdTransBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        NekDataWarehouseSharedPtr dataWarehouse)
        : BwdTransBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
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
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(), eBasis)));
        }

        if ((m_shapeType == LibUtilities::eNodalTri) ||
            (m_shapeType == LibUtilities::eNodalPrism) ||
            (m_shapeType == LibUtilities::eNodalTet))
        {
            std::vector<LibUtilities::BasisKey> basisKeys(
                m_dimension, LibUtilities::NullBasisKey);
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
            }

            LibUtilities::PointsType nodalType =
                (exp->IsNodalNonTensorialExp())
                    ? exp->GetNodalPointsKey().GetPointsType()
                    : LibUtilities::eNoPointsType;

            // Fetch NodalToModal Matrix if required.
            m_nodToMod =
                dataWarehouse->template GetData<MemSpace>(StdMatKey<simd_t>(
                    basisKeys, m_shapeType, eNodalToModal, nodalType));
        }
        else
        {
            m_nodToMod = (const simd_t *)nullptr;
        }

        // Workspace for kernels - also checks preconditions.
        if (m_dimension == 2)
        {
            unsigned int wsp0Size = 0;
            BwdTrans2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            m_wsp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size);
        }
        else if (m_dimension == 3)
        {
            unsigned int wsp0Size = 0, wsp1Size = 0;
            BwdTrans3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size);
            m_wsp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size);
            m_wsp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size);
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BwdTransBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
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
    std::vector<const TData *> m_B;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp0;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp1;
    const simd_t *m_nodToMod;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock) override
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

    void SegBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    void TriBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTriBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    void QuadBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    void HexBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    void PrismBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalPrismBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    void PyrBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    void TetBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTetBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nm[0], m_nq[0]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nm0, nq0);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator1D(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int nm0, const unsigned int nq0)
    {
        // Shape size.
        const auto nmTot = nm0;
        const auto nqTot = nq0;

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = (this->m_append)
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              inInterleaveWidth, chunkSize,
                                              nmTot, (TData *)inptr);
                }

                if (this->m_append)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                                  outInterleaveWidth, chunkSize,
                                                  nqTot, (TData *)outptr);
                    }

                    // BwdTrans kernel.
                    BwdTrans1DKernel<SHAPE_TYPE, true>(
                        nm0, nq0, m_B[0],
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr));
                }
                else
                {
                    // BwdTrans kernel.
                    BwdTrans1DKernel<SHAPE_TYPE, false>(
                        nm0, nq0, m_B[0],
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr));
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nm[0], m_nm[1],
                                         m_nq[0], m_nq[1]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nm0, nm1, nq0, nq1);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator2D(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
        const unsigned int nq1)
    {
        // Shape size.
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const auto nqTot = nq0 * nq1;

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = (this->m_append)
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              inInterleaveWidth, chunkSize,
                                              nmTot, (TData *)inptr);
                }

                if (this->m_append)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                                  outInterleaveWidth, chunkSize,
                                                  nqTot, (TData *)outptr);
                    }

                    // BwdTrans kernel.
                    BwdTrans2DKernel<SHAPE_TYPE, true>(
                        nm0, nm1, nq0, nq1, m_isModified, m_B[0], m_B[1],
                        m_nodToMod, m_wsp0.data(),
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr));
                }
                else
                {
                    // BwdTrans kernel.
                    BwdTrans2DKernel<SHAPE_TYPE, false>(
                        nm0, nm1, nq0, nq1, m_isModified, m_B[0], m_B[1],
                        m_nodToMod, m_wsp0.data(),
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr));
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nm[0], m_nm[1],
                                         m_nm[2], m_nq[0], m_nq[1], m_nq[2]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nm0, nm1, nm2, nq0,
                                         nq1, nq2);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator3D(
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
        const unsigned int nq0, const unsigned int nq1, const unsigned int nq2)
    {
        // Shape size.
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const auto nqTot = nq0 * nq1 * nq2;

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = (this->m_append)
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Loop over components.
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              inInterleaveWidth, chunkSize,
                                              nmTot, (TData *)inptr);
                }

                if (this->m_append)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                                  outInterleaveWidth, chunkSize,
                                                  nqTot, (TData *)outptr);
                    }

                    // BwdTrans kernel.
                    BwdTrans3DKernel<SHAPE_TYPE, true>(
                        nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, m_B[0],
                        m_B[1], m_B[2], m_nodToMod, m_wsp0.data(),
                        m_wsp1.data(), reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr));
                }
                else
                {
                    // BwdTrans kernel.
                    BwdTrans3DKernel<SHAPE_TYPE, false>(
                        nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, m_B[0],
                        m_B[1], m_B[2], m_nodToMod, m_wsp0.data(),
                        m_wsp1.data(), reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr));
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
