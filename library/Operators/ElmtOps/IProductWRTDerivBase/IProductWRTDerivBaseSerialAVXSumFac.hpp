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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp"
#include "Operators/Math/MathKernels.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, FieldState TFieldOut,
          typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TFieldOut, TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTDerivBaseBlockOpImpl(const unsigned int block_idx,
                                    const LocalRegions::ExpansionSharedPtr &exp,
                                    NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTDerivBaseBlockOp<TFieldOut, TData>(block_idx, exp,
                                                        dataWarehouse)
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
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(), eBasis)));
            m_DB.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eBasisDerivative)));
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eDerivative)));
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eWeights)));
        }

        if (m_dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(1)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
        }

        if ((m_shapeType == LibUtilities::eNodalTri) ||
            (m_shapeType == LibUtilities::eNodalTet) ||
            (m_shapeType == LibUtilities::eNodalPrism))
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

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            DerivFactorKey<TData>(block_idx, m_implInterleaveWidth, false));

        // Workspace for kernels - also checks preconditions.
        m_df = std::vector<simd_t, tinysimd::allocator<simd_t>>(m_dimension *
                                                                m_coordDim);
        if (m_dimension == 1)
        {
            const auto nqTot = m_nq[0];
            m_tmp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_tmp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
        }
        else if (m_dimension == 2)
        {
            unsigned int wsp0Size = 0;
            IProduct2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            m_wsp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size);
            const auto nqTot = m_nq[0] * m_nq[1];
            m_tmp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_tmp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_tmp2 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
        }
        else
        {
            unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
            IProduct3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size, wsp2Size);
            m_wsp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size);
            m_wsp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size);
            m_wsp2 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp2Size);
            const auto nqTot = m_nq[0] * m_nq[1] * m_nq[2];
            m_tmp0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_tmp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_tmp2 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
            m_tmp3 = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<ElmtBlockOp<FieldState::Phys, TFieldOut, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<IProductWRTDerivBaseBlockOpImpl<
            ExecSpace, Implementation, TFieldOut, TData>>(block_idx, exp,
                                                          dataWarehouse);
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
    std::vector<const simd_t *> m_B;
    std::vector<const simd_t *> m_DB;
    std::vector<const simd_t *> m_D;
    std::vector<const simd_t *> m_W;
    std::vector<const simd_t *> m_f;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_df;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp0;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp1;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp2;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_tmp0;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_tmp1;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_tmp2;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_tmp3;
    const simd_t *m_nodToModTrans;
    const TData *m_jacptr;
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, TFieldOut> &outblock) override
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
            // NodalTriangles
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
                  BlockAccessor<TData, TFieldOut> &outblock);

    void TriBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, TFieldOut> &outblock);

    void NodalTriBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                       BlockAccessor<TData, TFieldOut> &outblock);

    void QuadBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                   BlockAccessor<TData, TFieldOut> &outblock);

    void HexBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, TFieldOut> &outblock);

    void NodalPrismBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                         BlockAccessor<TData, TFieldOut> &outblock);

    void PrismBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, TFieldOut> &outblock);

    void PyrBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, TFieldOut> &outblock);

    void TetBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                  BlockAccessor<TData, TFieldOut> &outblock);

    void NodalTetBlock(BlockAccessor<TData, FieldState::Phys> &inblock,
                       BlockAccessor<TData, TFieldOut> &outblock);

    // Non-size based operator Coeff output
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, TFieldOut> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nm[0], m_nq[0]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, TFieldOut> &outblock)
    {
        Operator1D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nm0, nq0);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator1D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, TFieldOut> &outblock,
        [[maybe_unused]] const unsigned int nm0, const unsigned int nq0)
    {
        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nq0;
        }

        // Initialize pointers.
        auto inptr    = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr   = (this->m_append)
                            ? outblock.template GetPtr<MemSpace, ReadWrite>()
                            : outblock.template GetPtr<MemSpace, WriteOnly>();
        auto outndata = outblock.GetNumData();

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
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto inoffset_vec =
            inblock.CompSize() * inblock.GetNumHomoModes() / simd_t::width;
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                            nq0, (TData *)(inptr + k * inoffset));
                    }
                }
                StdAlignDerivBase1D<DEFORMED>(
                    nq0, m_coordDim, reinterpret_cast<const simd_t *>(dfptr),
                    m_df, inoffset_vec, reinterpret_cast<const simd_t *>(inptr),
                    m_tmp0.data(), reinterpret_cast<const simd_t *>(jacptr),
                    m_W[0]);
                SumDerivTensor1DKernel<false>(nq0, m_tmp0.data(), m_D[0],
                                              m_tmp1.data());

                if constexpr (TFieldOut == FieldState::Phys)
                {
                    simd_t *vecout = reinterpret_cast<simd_t *>(outptr);
                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, nq0, (TData *)outptr);
                        }

                        for (unsigned i = 0; i < nq0; ++i)
                        {
                            vecout[i] += this->m_scale * m_tmp1[i];
                        }
                    }
                    else
                    {
                        for (unsigned i = 0; i < nq0; ++i)
                        {
                            vecout[i] = this->m_scale * m_tmp1[i];
                        }
                    }
                }
                else
                {
                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, nq0, (TData *)outptr);
                        }

                        IProductSegKernel<true, true>(
                            nm0, nq0, m_tmp1.data(), m_B[0],
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    else
                    {
                        IProductSegKernel<true, false>(
                            nm0, nq0, m_tmp1.data(), m_B[0],
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                            nq0,
                            (TData *)inptr +
                                k * inblock.CompSize() *
                                    inblock.GetNumHomoModes() -
                                (width_ratio - 1) * nq0 * simd_t::width);
                    }
                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        outndata,
                        (TData *)outptr -
                            (width_ratio - 1) * outndata * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nq0 * simd_t::width;
                outptr += outndata * simd_t::width;
                jacptr += jacSize * simd_t::width;
                dfptr += jacSize * m_coordDim * simd_t::width;
            }

            // Advance input by m_coordDim-1 componennts since have already
            // advanced one component in the above.
            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (m_coordDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, TFieldOut> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nm[0], m_nm[1],
                                         m_nq[0], m_nq[1]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, TFieldOut> &outblock)
    {
        Operator2D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nm0, nm1, nq0, nq1);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator2D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, TFieldOut> &outblock,
        [[maybe_unused]] const unsigned int nm0,
        [[maybe_unused]] const unsigned int nm1, const unsigned int nq0,
        const unsigned int nq1)
    {
        // Shape size.
        const auto nqTot = nq0 * nq1;

        const auto ndf       = 2u * m_coordDim;
        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Initialize pointers.
        auto inptr    = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr   = (this->m_append)
                            ? outblock.template GetPtr<MemSpace, ReadWrite>()
                            : outblock.template GetPtr<MemSpace, WriteOnly>();
        auto outndata = outblock.GetNumData();

        // Workspace for kernels.
        simd_t *tmpPtr[2];
        tmpPtr[0] = m_tmp0.data();
        tmpPtr[1] = m_tmp1.data();

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
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto inoffset_vec =
            inblock.CompSize() * inblock.GetNumHomoModes() / simd_t::width;
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                            nqTot, (TData *)(inptr + k * inoffset));
                    }
                }

                StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, m_coordDim,
                    reinterpret_cast<const simd_t *>(dfptr), m_df, inoffset_vec,
                    reinterpret_cast<const simd_t *>(inptr), tmpPtr, m_f[0],
                    m_f[1], reinterpret_cast<const simd_t *>(jacptr), m_W[0],
                    m_W[1]);
                SumDerivTensor2DKernel<false>(nq0, nq1, tmpPtr[0], tmpPtr[1],
                                              m_D[0], m_D[1], m_tmp2.data());

                if constexpr (TFieldOut == FieldState::Phys)
                {
                    simd_t *vecout = reinterpret_cast<simd_t *>(outptr);

                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, nqTot, (TData *)outptr);
                        }

                        for (unsigned i = 0; i < nqTot; ++i)
                        {
                            vecout[i] += this->m_scale * m_tmp2[i];
                        }
                    }
                    else
                    {
                        for (unsigned i = 0; i < nqTot; ++i)
                        {
                            vecout[i] = this->m_scale * m_tmp2[i];
                        }
                    }
                }
                else
                {
                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, nqTot, (TData *)outptr);
                        }

                        IProduct2DKernel<SHAPE_TYPE, true, true>(
                            nm0, nm1, nq0, nq1, m_isModified, m_tmp2.data(),
                            m_B[0], m_B[1], m_nodToModTrans, m_wsp0.data(),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    else
                    {
                        IProduct2DKernel<SHAPE_TYPE, true, false>(
                            nm0, nm1, nq0, nq1, m_isModified, m_tmp2.data(),
                            m_B[0], m_B[1], m_nodToModTrans, m_wsp0.data(),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            (TData *)inptr + k * inoffset -
                                (width_ratio - 1) * nqTot * simd_t::width);
                    }
                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        outndata,
                        (TData *)outptr -
                            (width_ratio - 1) * outndata * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += outndata * simd_t::width;
                jacptr += jacSize * simd_t::width;
                dfptr += jacSize * ndf * simd_t::width;
            }

            // Advance input by m_coordDim-1 componennts since have already
            // advanced one component in the above.
            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (m_coordDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    // Non-size based operator with Coeff output
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, TFieldOut> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, m_nm[0], m_nm[1],
                                         m_nm[2], m_nq[0], m_nq[1], m_nq[2]);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, TFieldOut> &outblock)
    {
        Operator3D<SHAPE_TYPE, DEFORMED>(inblock, outblock, nm0, nm1, nm2, nq0,
                                         nq1, nq2);
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    NEK_FORCE_INLINE void Operator3D(
        BlockAccessor<TData, FieldState::Phys> &inblock,
        BlockAccessor<TData, TFieldOut> &outblock,
        [[maybe_unused]] const unsigned int nm0,
        [[maybe_unused]] const unsigned int nm1,
        [[maybe_unused]] const unsigned int nm2, const unsigned int nq0,
        const unsigned int nq1, const unsigned int nq2)
    {
        // Shape size.
        const auto nqTot = nq0 * nq1 * nq2;

        constexpr unsigned int ndf = 9u;
        unsigned int jacSize       = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Initialize pointers.
        auto inptr    = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr   = (this->m_append)
                            ? outblock.template GetPtr<MemSpace, ReadWrite>()
                            : outblock.template GetPtr<MemSpace, WriteOnly>();
        auto outndata = outblock.GetNumData();

        // Workspace for kernels.
        simd_t *tmpPtr[3];
        tmpPtr[0] = m_tmp0.data();
        tmpPtr[1] = m_tmp1.data();
        tmpPtr[2] = m_tmp2.data();

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
        for (unsigned int n = 0; n < outblock.GetNumComponents(); ++n)
        {
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int k = 0; k < 3; ++k)
                    {
                        ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                            nqTot, (TData *)(inptr + k * inblock.CompSize()));
                    }
                }

                StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, reinterpret_cast<const simd_t *>(dfptr),
                    m_df,
                    inblock.GetNumElmtGroups(m_implInterleaveWidth) * nqTot,
                    m_f[0], m_f[1], m_f[2], m_f[3],
                    reinterpret_cast<const simd_t *>(jacptr), m_W[0], m_W[1],
                    m_W[2], reinterpret_cast<const simd_t *>(inptr), tmpPtr);
                SumDerivTensor3DKernel<false>(nq0, nq1, nq2, tmpPtr[0],
                                              tmpPtr[1], tmpPtr[2], m_D[0],
                                              m_D[1], m_D[2], m_tmp3.data());

                if constexpr (TFieldOut == FieldState::Phys)
                {
                    simd_t *vecout = reinterpret_cast<simd_t *>(outptr);
                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, nqTot, (TData *)outptr);
                        }

                        for (unsigned i = 0; i < nqTot; ++i)
                        {
                            vecout[i] += this->m_scale * m_tmp3[i];
                        }
                    }
                    else
                    {
                        for (unsigned i = 0; i < nqTot; ++i)
                        {
                            vecout[i] = this->m_scale * m_tmp3[i];
                        }
                    }
                }
                else
                {
                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, nqTot, (TData *)outptr);
                        }

                        IProduct3DKernel<SHAPE_TYPE, true, true>(
                            nm0, nm1, nm2, nq0, nq1, nq2, m_isModified,
                            m_tmp3.data(), m_B[0], m_B[1], m_B[2],
                            m_nodToModTrans, m_wsp0.data(), m_wsp1.data(),
                            m_wsp2.data(), reinterpret_cast<simd_t *>(outptr),
                            this->m_scale);
                    }
                    else
                    {
                        IProduct3DKernel<SHAPE_TYPE, true, false>(
                            nm0, nm1, nm2, nq0, nq1, nq2, m_isModified,
                            m_tmp3.data(), m_B[0], m_B[1], m_B[2],
                            m_nodToModTrans, m_wsp0.data(), m_wsp1.data(),
                            m_wsp2.data(), reinterpret_cast<simd_t *>(outptr),
                            this->m_scale);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int k = 0; k < 3; ++k)
                    {
                        ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            (TData *)inptr + k * inblock.CompSize() -
                                (width_ratio - 1) * nqTot * simd_t::width);
                    }

                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        outndata,
                        (TData *)outptr -
                            (width_ratio - 1) * outndata * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += outndata * simd_t::width;
                jacptr += jacSize * simd_t::width;
                dfptr += jacSize * ndf * simd_t::width;
            }

            // Advance input by m_coordDim-1 componennts since have already
            // advanced one component in the above.
            inptr += 2 * inblock.CompSize();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
