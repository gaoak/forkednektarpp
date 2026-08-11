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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOp.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTBaseBlockOpImpl : public IProductWRTBaseBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTBaseBlockOpImpl(const unsigned int block_idx,
                               const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTBaseBlockOp<TData>(block_idx, exp, dataWarehouse)
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
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eWeights)));
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
            m_nodToModTrans =
                dataWarehouse->template GetData<MemSpace>(StdMatKey<simd_t>(
                    basisKeys, m_shapeType, eNodalToModalTranspose, nodalType));
        }
        else
        {
            m_nodToModTrans = (const simd_t *)nullptr;
        }

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));

        // Workspace for kernels - also checks preconditions.
        if (m_dimension == 2)
        {
            unsigned int wsp0Size = 0;
            IProduct2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            m_wsp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size);
        }
        else if (m_dimension == 3)
        {
            unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
            IProduct3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size, wsp2Size);
            m_wsp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size);
            m_wsp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size);
            m_wsp2 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp2Size);
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            IProductWRTBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
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
    std::vector<const TData *> m_W;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp0;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp1;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp2;
    const simd_t *m_nodToModTrans;
    const TData *m_jacptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
                 MultiRegions::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
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
            // Prism
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
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    void TriBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    void NodalTriBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    void QuadBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    void HexBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    void PrismBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    void NodalPrismBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    void PyrBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    void TetBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    void NodalTetBlock(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nm[0], m_nq[0]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nm0, nq0);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator1D(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock,
        const unsigned int nm0, const unsigned int nq0)
    {
        // Shape size.
        const auto nmTot = nm0;
        const auto nqTot = nq0;

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
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

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;

            // Loop over element groups.
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

                if (this->m_integration)
                {
                    // IProduct Kernel.
                    if (this->m_scale == 1.0)
                    {
                        IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                            nm0, nq0, reinterpret_cast<const simd_t *>(inptr),
                            m_B[0], m_W[0],
                            reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<simd_t *>(outptr));
                    }
                    else
                    {
                        IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                            nm0, nq0, reinterpret_cast<const simd_t *>(inptr),
                            m_B[0], m_W[0],
                            reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }
                else
                {
                    // IProduct Kernel with no quadrature B^T op
                    if (this->m_scale == 1.0)
                    {
                        IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                            nm0, nq0, reinterpret_cast<const simd_t *>(inptr),
                            m_B[0], reinterpret_cast<simd_t *>(outptr));
                    }
                    else
                    {
                        IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                            nm0, nq0, reinterpret_cast<const simd_t *>(inptr),
                            m_B[0], reinterpret_cast<simd_t *>(outptr),
                            this->m_scale);
                    }
                }

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
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += nmTot * simd_t::width;
                jacptr += jacSize * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nm[0], m_nm[1],
                                         m_nq[0], m_nq[1]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nm0, nm1, nq0, nq1);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator2D(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock,
        const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
        const unsigned int nq1)
    {
        // Shape size.
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const auto nqTot = nq0 * nq1;

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
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

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;

            // Loop over element groups.
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

                // IProduct Kernel.
                if (this->m_integration)
                {
                    if (this->m_scale == 1.0)
                    {
                        IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                            nm0, nm1, nq0, nq1, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr), m_B[0],
                            m_B[1], m_W[0], m_W[1], m_nodToModTrans,
                            reinterpret_cast<const simd_t *>(jacptr),
                            m_wsp0.data(), reinterpret_cast<simd_t *>(outptr));
                    }
                    else
                    {
                        IProduct2DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                            nm0, nm1, nq0, nq1, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr), m_B[0],
                            m_B[1], m_W[0], m_W[1], m_nodToModTrans,
                            reinterpret_cast<const simd_t *>(jacptr),
                            m_wsp0.data(), reinterpret_cast<simd_t *>(outptr),
                            this->m_scale);
                    }
                }
                else
                {
                    if (this->m_scale == 1.0)
                    {
                        IProduct2DKernel<SHAPE_TYPE, false, false>(
                            nm0, nm1, nq0, nq1, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr), m_B[0],
                            m_B[1], m_nodToModTrans, m_wsp0.data(),
                            reinterpret_cast<simd_t *>(outptr));
                    }
                    else
                    {
                        IProduct2DKernel<SHAPE_TYPE, true, false>(
                            nm0, nm1, nq0, nq1, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr), m_B[0],
                            m_B[1], m_nodToModTrans, m_wsp0.data(),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }

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
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += nmTot * simd_t::width;
                jacptr += jacSize * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nm[0], m_nm[1],
                                         m_nm[2], m_nq[0], m_nq[1], m_nq[2]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nm0, nm1, nm2, nq0,
                                         nq1, nq2);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator3D(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, FieldState::Coeff> &outblock,
        const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
        const unsigned int nq0, const unsigned int nq1, const unsigned int nq2)
    {
        // Shape size.
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const auto nqTot = nq0 * nq1 * nq2;

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
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

        // Loop over components.
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            auto jacptr = m_jacptr;

            // Loop over element groups.
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

                if (this->m_integration)
                {
                    if (this->m_scale == 1.0)
                    {
                        IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                            nm0, nm1, nm2, nq0, nq1, nq2, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr), m_B[0],
                            m_B[1], m_B[2], m_W[0], m_W[1], m_W[2],
                            m_nodToModTrans,
                            reinterpret_cast<const simd_t *>(jacptr),
                            m_wsp0.data(), m_wsp1.data(), m_wsp2.data(),
                            reinterpret_cast<simd_t *>(outptr));
                    }
                    else
                    {
                        IProduct3DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                            nm0, nm1, nm2, nq0, nq1, nq2, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr), m_B[0],
                            m_B[1], m_B[2], m_W[0], m_W[1], m_W[2],
                            m_nodToModTrans,
                            reinterpret_cast<const simd_t *>(jacptr),
                            m_wsp0.data(), m_wsp1.data(), m_wsp2.data(),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }
                else
                {
                    if (this->m_scale == 1.0)
                    {
                        IProduct3DKernel<SHAPE_TYPE, false, false>(
                            nm0, nm1, nm2, nq0, nq1, nq2, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr), m_B[0],
                            m_B[1], m_B[2], m_nodToModTrans, m_wsp0.data(),
                            m_wsp1.data(), m_wsp2.data(),
                            reinterpret_cast<simd_t *>(outptr));
                    }
                    else
                    {
                        IProduct3DKernel<SHAPE_TYPE, true, false>(
                            nm0, nm1, nm2, nq0, nq1, nq2, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr), m_B[0],
                            m_B[1], m_B[2], m_nodToModTrans, m_wsp0.data(),
                            m_wsp1.data(), m_wsp2.data(),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }
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
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += nmTot * simd_t::width;
                jacptr += jacSize * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
