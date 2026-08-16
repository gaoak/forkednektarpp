///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlSerialAVXSumFac.hpp
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
#include "Operators/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp"

#include "Operators/ElmtOps/CurlCurl/CurlCurlSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class CurlCurlBlockOpImpl : public CurlCurlBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    CurlCurlBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : CurlCurlBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        auto dimension = exp->GetShapeDimension();
        m_coordDim     = dimension; // required for boost_pp switch

        ASSERTL1(m_coordDim == 2 || m_coordDim == 3,
                 "CurlCurl operator only defined for 2D and 3D.");

        for (unsigned int d = 0; d < dimension; d++)
        {
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eDerivative)));
        }

        if (dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }
        else if (dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }

        // Fetch deriv factors data.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));

        // Allocate workspace
        if (dimension == 1)
        {
            const auto nqTot = m_nq[0];
            m_wsp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
        }
        else if (dimension == 2)
        {
            const auto nqTot = m_nq[0] * m_nq[1];
            m_wsp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp2 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
        }
        else if (dimension == 3)
        {
            const auto nqTot = m_nq[0] * m_nq[1] * m_nq[2];
            m_wsp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp2 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp3 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp4 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp5 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp6 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp7 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_wsp8 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            CurlCurlBlockOpImpl<ExecSpace, Implementation, TData>>(
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
    std::vector<const TData *> m_D;
    std::vector<const simd_t *> m_f;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp0;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp1;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp2;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp3;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp4;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp5;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp6;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp7;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp8;
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
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

    void SegBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void TriBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTriBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void QuadBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void HexBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void PrismBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalPrismBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void PyrBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void TetBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTetBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_coordDim,
                                         m_nq[0]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, coordDim, nq0);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
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

        // const auto compOffset =
        //     outblock.GetNumElmtGroups(m_implInterleaveWidth) * nqTot *
        //     outblock.GetNumHomoModes();

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
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        nqTot, (TData *)inptr);
                }

                // Get the basic derivative.
                PhysDerivTensor1DKernel(
                    nq0, reinterpret_cast<const simd_t *>(inptr), m_D[0],
                    reinterpret_cast<simd_t *>(outptr));

                // Calculate physical derivative.
                PhysDerivDir1DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
                    nq0, m_coordDim, reinterpret_cast<const simd_t *>(dfptr),
                    reinterpret_cast<const simd_t *>(outptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_coordDim, m_nq[0],
                                         m_nq[1]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0, unsigned int nq1>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, coordDim, nq0, nq1);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
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
        auto inptr0  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto inptr1  = inptr0 + compOffset;
        auto outptr0 = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto outptr1 = outptr0 + compOffset;
        auto dfptr   = m_dfptr;

        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {
            auto outsimd1 = reinterpret_cast<simd_t *>(outptr1);

            // Reshape, if necessary.
            if (e % width_ratio == 0)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, chunkSize, nqTot,
                    (TData *)inptr0);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, chunkSize, nqTot,
                    (TData *)inptr1);
            }

            // du/dx
            PhysDerivTensor2DKernel(
                nq0, nq1, reinterpret_cast<const simd_t *>(inptr0), m_D[0],
                m_D[1], m_wsp0.data(), m_wsp1.data());

            // du/dy
            PhysDerivDir2DKernel<SHAPE_TYPE, false, DEFORMED, 1>(
                nq0, nq1, 2, m_f[0], m_f[1],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), m_wsp2.data());

            // dv/dx
            PhysDerivTensor2DKernel(
                nq0, nq1, reinterpret_cast<const simd_t *>(inptr1), m_D[0],
                m_D[1], m_wsp0.data(), m_wsp1.data());

            PhysDerivDir2DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
                nq0, nq1, 2, m_f[0], m_f[1],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), m_wsp0.data());

            // omega_z = dv/dx - du/dy
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                m_wsp2[i] = m_wsp0[i] - m_wsp2[i];
            }

            // q_x = d(omega_z)/dy
            PhysDerivTensor2DKernel(nq0, nq1, m_wsp2.data(), m_D[0], m_D[1],
                                    m_wsp0.data(), m_wsp1.data());
            PhysDerivDir2DKernel<SHAPE_TYPE, false, DEFORMED, 1>(
                nq0, nq1, 2, m_f[0], m_f[1],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), reinterpret_cast<simd_t *>(outptr0));

            // q_y = -d(omega_z)/dx
            PhysDerivDir2DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
                nq0, nq1, 2, m_f[0], m_f[1],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), outsimd1);
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                outsimd1[i] = -outsimd1[i];
            }

            // Reshape back, if necessary.
            if (e % width_ratio == width_ratio - 1)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr0 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr1 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)outptr0 -
                        (width_ratio - 1) * nqTot * simd_t::width);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)outptr1 -
                        (width_ratio - 1) * nqTot * simd_t::width);
            }

            // Increment pointers for the next elmt group.
            dfptr += dfsize * simd_t::width;
            inptr0 += nqTot * simd_t::width;
            inptr1 += nqTot * simd_t::width;
            outptr0 += nqTot * simd_t::width;
            outptr1 += nqTot * simd_t::width;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nq[0], m_nq[1],
                                         m_nq[2]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nq0, nq1, nq2);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int nq0, const unsigned int nq1, const unsigned int nq2)
    {
        // Shape size.
        const auto nqTot = nq0 * nq1 * nq2;

        ASSERTL1(inblock.GetNumComponents() >= 3,
                 "Input block does not have at least three components");

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
        auto inptr0  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto inptr1  = inptr0 + compOffset;
        auto inptr2  = inptr1 + compOffset;
        auto outptr0 = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto outptr1 = outptr0 + compOffset;
        auto outptr2 = outptr1 + compOffset;
        auto dfptr   = m_dfptr;

        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {
            auto outsimd0 = reinterpret_cast<simd_t *>(outptr0);
            auto outsimd1 = reinterpret_cast<simd_t *>(outptr1);
            auto outsimd2 = reinterpret_cast<simd_t *>(outptr2);

            // Reshape, if necessary.
            if (e % width_ratio == 0)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, chunkSize, nqTot,
                    (TData *)inptr0);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, chunkSize, nqTot,
                    (TData *)inptr1);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, chunkSize, nqTot,
                    (TData *)inptr2);
            }

            // omega_x = dw/dy - dv/dz
            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr1), m_D[0],
                m_D[1], m_D[2], m_wsp0.data(), m_wsp1.data(), m_wsp2.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 2>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), m_wsp2.data(), m_wsp3.data());

            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr2), m_D[0],
                m_D[1], m_D[2], m_wsp0.data(), m_wsp1.data(), m_wsp2.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 1>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), m_wsp2.data(), m_wsp0.data());
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                m_wsp3[i] = m_wsp0[i] - m_wsp3[i];
            }

            // omega_y = du/dz - dw/dx
            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr0), m_D[0],
                m_D[1], m_D[2], m_wsp0.data(), m_wsp1.data(), m_wsp2.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 2>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), m_wsp2.data(), m_wsp4.data());

            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr2), m_D[0],
                m_D[1], m_D[2], m_wsp0.data(), m_wsp1.data(), m_wsp2.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), m_wsp2.data(), m_wsp0.data());
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                m_wsp4[i] -= m_wsp0[i];
            }

            // omega_z = dv/dx - du/dy
            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr0), m_D[0],
                m_D[1], m_D[2], m_wsp0.data(), m_wsp1.data(), m_wsp2.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 1>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), m_wsp2.data(), m_wsp5.data());

            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, reinterpret_cast<const simd_t *>(inptr1), m_D[0],
                m_D[1], m_D[2], m_wsp0.data(), m_wsp1.data(), m_wsp2.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp0.data(),
                m_wsp1.data(), m_wsp2.data(), m_wsp0.data());
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                m_wsp5[i] = m_wsp0[i] - m_wsp5[i];
            }

            // Match the legacy subtraction ordering by materialising the
            // second derivatives needed for the outer curl before combining.
            //
            // omega_x -> {d(omega_x)/dy, d(omega_x)/dz}
            PhysDerivTensor3DKernel(nq0, nq1, nq2, m_wsp3.data(), m_D[0],
                                    m_D[1], m_D[2], m_wsp6.data(),
                                    m_wsp7.data(), m_wsp8.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 1>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp6.data(),
                m_wsp7.data(), m_wsp8.data(), m_wsp0.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 2>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp6.data(),
                m_wsp7.data(), m_wsp8.data(), m_wsp1.data());

            // omega_y -> {d(omega_y)/dx, d(omega_y)/dz}
            PhysDerivTensor3DKernel(nq0, nq1, nq2, m_wsp4.data(), m_D[0],
                                    m_D[1], m_D[2], m_wsp6.data(),
                                    m_wsp7.data(), m_wsp8.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp6.data(),
                m_wsp7.data(), m_wsp8.data(), m_wsp2.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 2>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp6.data(),
                m_wsp7.data(), m_wsp8.data(), m_wsp3.data());

            // omega_z -> {d(omega_z)/dx, d(omega_z)/dy}
            PhysDerivTensor3DKernel(nq0, nq1, nq2, m_wsp5.data(), m_D[0],
                                    m_D[1], m_D[2], m_wsp6.data(),
                                    m_wsp7.data(), m_wsp8.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp6.data(),
                m_wsp7.data(), m_wsp8.data(), m_wsp4.data());
            PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 1>(
                nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3],
                reinterpret_cast<const simd_t *>(dfptr), m_wsp6.data(),
                m_wsp7.data(), m_wsp8.data(), m_wsp5.data());

            for (unsigned int i = 0; i < nqTot; ++i)
            {
                outsimd0[i] = m_wsp5[i] - m_wsp3[i];
                outsimd1[i] = m_wsp1[i] - m_wsp4[i];
                outsimd2[i] = m_wsp2[i] - m_wsp0[i];
            }

            // Reshape back, if necessary.
            if (e % width_ratio == width_ratio - 1)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr0 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr1 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)inptr2 -
                        (width_ratio - 1) * nqTot * simd_t::width);

                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)outptr0 -
                        (width_ratio - 1) * nqTot * simd_t::width);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)outptr1 -
                        (width_ratio - 1) * nqTot * simd_t::width);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, nqTot,
                    (TData *)outptr2 -
                        (width_ratio - 1) * nqTot * simd_t::width);
            }

            // Increment pointers for the next elmt group.
            dfptr += dfsize * simd_t::width;
            inptr0 += nqTot * simd_t::width;
            inptr1 += nqTot * simd_t::width;
            inptr2 += nqTot * simd_t::width;
            outptr0 += nqTot * simd_t::width;
            outptr1 += nqTot * simd_t::width;
            outptr2 += nqTot * simd_t::width;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
