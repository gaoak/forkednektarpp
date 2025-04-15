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

#include "Operators/ElmtOps/IProductWRTDerivBase/OperatorIProductWRTDerivBase.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorIProductWRTDerivBaseImpl
    : public BlockOperatorIProductWRTDerivBase<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorIProductWRTDerivBaseImpl(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorIProductWRTDerivBase<TData>(exp, dataWarehouse)
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
            m_B.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(), eBasis)));
            m_DB.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eBasisDerivative)));
            m_D.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eDerivative)));
            m_W.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eWeights)));
        }

        if (m_dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(1)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
        }

        if ((m_shapeType == LibUtilities::eNodalTri) ||
            (m_shapeType == LibUtilities::eNodalTet) ||
            (m_shapeType == LibUtilities::eNodalPrism))
        {
            // Fetch NodalToModal Matrix if required.
            m_nodToModTrans =
                this->m_dataWarehouse->template GetData<ExecSpace>(
                    VandemondeKey<simd_t>(eNodalToModalTranspose,
                                          exp->GetElmtId()));
        }
        else
        {
            m_nodToModTrans = (const simd_t *)nullptr;
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

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<BlockOperatorIProductWRTDerivBaseImpl<
            ExecSpace, Implementation, TData>>(exp, dataWarehouse);
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
    const simd_t *m_nodToModTrans;

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

    void NodalPrismBlock(BlockAccessor<TData> &inblock,
                         BlockAccessor<TData> &outblock);

    void PrismBlock(BlockAccessor<TData> &inblock,
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
        const auto nm0 = m_nm[0];
        const auto nq0 = m_nq[0];

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nq0;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                   inblock.GetNumElements())));
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

        // Workspace for kernels.
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(m_coordDim);
        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp0(nq0);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr  = reinterpret_cast<const simd_t *>(input);
        auto outptr = reinterpret_cast<simd_t *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int n = 0; n < m_coordDim; ++n)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, nq0,
                            (TData *)(inptr +
                                      n * inblock.GetNumElmtGroups() * nq0));
                    }
                }

                StdAlignDerivBase1D<DEFORMED>(nq0, m_coordDim, dfptr, df_tmp,
                                              inblock.GetNumElmtGroups() * nq0,
                                              inptr, tmp0.data());
                IProductSegKernel<false, false, DEFORMED>(
                    nm0, nq0, tmp0.data(), m_DB[0], m_W[0], jacptr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nq0;
                outptr += nm0;
                jacptr += jacSize;
                dfptr += jacSize * m_coordDim;
            }
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nq0;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                   inblock.GetNumElements())));
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

        // Workspace for kernels.
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(m_coordDim);
        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp0(nq0);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr  = reinterpret_cast<const simd_t *>(input);
        auto outptr = reinterpret_cast<simd_t *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int n = 0; n < m_coordDim; ++n)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, nq0,
                            (TData *)(inptr +
                                      n * inblock.GetNumElmtGroups() * nq0));
                    }
                }

                StdAlignDerivBase1D<DEFORMED>(nq0, m_coordDim, dfptr, df_tmp,
                                              inblock.GetNumElmtGroups() * nq0,
                                              inptr, tmp0.data());
                IProductSegKernel<false, false, DEFORMED>(
                    nm0, nq0, tmp0.data(), m_DB[0], m_W[0], jacptr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nq0;
                outptr += nm0;
                jacptr += jacSize;
                dfptr += jacSize * m_coordDim;
            }
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const auto nqTot = nq0 * nq1;

        const auto ndf       = 2u * m_coordDim;
        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                   inblock.GetNumElements())));
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

        // Workspace for kernels.
        unsigned int wspSize = 0;
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wspSize);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp(wspSize);
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ndf),
            tmp0(nqTot), tmp1(nqTot);
        simd_t *tmpPtr[2];
        tmpPtr[0] = tmp0.data();
        tmpPtr[1] = tmp1.data();

        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp2(nqTot);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr  = reinterpret_cast<const simd_t *>(input);
        auto outptr = reinterpret_cast<simd_t *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            auto jacptr        = jacptr_init;
            auto dfptr         = dfptr_init;
            auto NumElmtGroups = inblock.GetNumElmtGroups();
            for (unsigned int e = 0; e < NumElmtGroups; ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int n = 0; n < m_coordDim; ++n)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, nqTot,
                            (TData *)(inptr + n * NumElmtGroups * nqTot));
                    }
                }

                StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, m_coordDim, dfptr, df_tmp, NumElmtGroups * nqTot,
                    inptr, tmpPtr, m_f[0], m_f[1]);
                SumDerivTensor2DKernel<DEFORMED, simd_t>(
                    nq0, nq1, tmpPtr[0], tmpPtr[1], m_W[0], m_W[1], jacptr,
                    m_D[0], m_D[1], tmp2.data());
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, m_isModified, tmp2.data(), m_B[0],
                    m_B[1], m_nodToModTrans, wsp.data(), outptr, 1.0);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot;
                jacptr += jacSize;
                dfptr += jacSize * ndf;
            }

            // Advance input by m_coordDim-1 componennts since have already
            // advanced one component in the above.
            inptr += nqTot * NumElmtGroups * (m_coordDim - 1);
        }
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

        const auto ndf       = 2u * m_coordDim;
        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(),
                                      m_implInterleaveWidth,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();

        const auto width_ratio = (interleave_width == 1)
                                     ? 1
                                     : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Workspace for kernels.
        unsigned int wspSize = 0;
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wspSize);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp(wspSize);
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ndf),
            tmp0(nqTot), tmp1(nqTot);
        simd_t *tmpPtr[2];
        tmpPtr[0] = tmp0.data();
        tmpPtr[1] = tmp1.data();

        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp2(nqTot);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr  = reinterpret_cast<const simd_t *>(input);
        auto outptr = reinterpret_cast<simd_t *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            auto jacptr        = jacptr_init;
            auto dfptr         = dfptr_init;
            auto NumElmtGroups = inblock.GetNumElmtGroups();
            for (unsigned int e = 0; e < NumElmtGroups; ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int n = 0; n < m_coordDim; ++n)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, nqTot,
                            (TData *)(inptr + n * NumElmtGroups * nqTot));
                    }
                }

                StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, m_coordDim, dfptr, df_tmp, NumElmtGroups * nqTot,
                    inptr, tmpPtr, m_f[0], m_f[1]);
                SumDerivTensor2DKernel<DEFORMED, simd_t>(
                    nq0, nq1, tmpPtr[0], tmpPtr[1], m_W[0], m_W[1], jacptr,
                    m_D[0], m_D[1], tmp2.data());
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, m_isModified, tmp2.data(), m_B[0],
                    m_B[1], m_nodToModTrans, wsp.data(), outptr, 1.0);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot;
                jacptr += jacSize;
                dfptr += jacSize * ndf;
            }

            // Advance input by m_coordDim-1 componennts since have already
            // advanced one component in the above.
            inptr += nqTot * NumElmtGroups * (m_coordDim - 1);
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];
        const auto nm2 = m_nm[2];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];
        const auto nq2 = m_nq[2];

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const auto nqTot = nq0 * nq1 * nq2;

        const auto ndf       = 9u;
        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                   inblock.GetNumElements())));
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

        // Workspace for kernels.
        unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp0(nqTot),
            tmp1(nqTot), tmp2(nqTot);
        simd_t *tmpPtr[3];
        tmpPtr[0] = tmp0.data();
        tmpPtr[1] = tmp1.data();
        tmpPtr[2] = tmp2.data();
        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp3(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ndf);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr  = reinterpret_cast<const simd_t *>(input);
        auto outptr = reinterpret_cast<simd_t *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            auto jacptr        = jacptr_init;
            auto dfptr         = dfptr_init;
            auto NumElmtGroups = inblock.GetNumElmtGroups();
            for (unsigned int e = 0; e < NumElmtGroups; ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int n = 0; n < 3; ++n)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, nqTot,
                            (TData *)(inptr + n * NumElmtGroups * nqTot));
                    }
                }

                StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, dfptr, df_tmp, NumElmtGroups * nqTot, m_f[0],
                    m_f[1], m_f[2], m_f[3], inptr, tmpPtr);
                SumDerivTensor3DKernel<DEFORMED, simd_t>(
                    nq0, nq1, nq2, tmpPtr[0], tmpPtr[1], tmpPtr[2], m_W[0],
                    m_W[1], m_W[2], jacptr, m_D[0], m_D[1], m_D[2],
                    tmp3.data());
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, tmp3.data(),
                    m_B[0], m_B[1], m_B[2], m_nodToModTrans, wsp0.data(),
                    wsp1.data(), wsp2.data(), outptr, 1.0);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot;
                jacptr += jacSize;
                dfptr += jacSize * ndf;
            }

            // Advance input by m_coordDim-1 componennts since have already
            // advanced one component in the above.
            inptr += nqTot * NumElmtGroups * 2;
        }
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

        constexpr unsigned int ndf = 9u;
        unsigned int jacSize       = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                   inblock.GetNumElements())));
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

        // Workspace for kernels.
        unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp0(nqTot),
            tmp1(nqTot), tmp2(nqTot);
        simd_t *tmpPtr[3];
        tmpPtr[0] = tmp0.data();
        tmpPtr[1] = tmp1.data();
        tmpPtr[2] = tmp2.data();
        std::vector<simd_t, tinysimd::allocator<simd_t>> tmp3(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> df_tmp(ndf);

        // Initialize pointers.
        auto input  = (interleave_width == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr  = reinterpret_cast<const simd_t *>(input);
        auto outptr = reinterpret_cast<simd_t *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            auto jacptr        = jacptr_init;
            auto dfptr         = dfptr_init;
            auto NumElmtGroups = inblock.GetNumElmtGroups();
            for (unsigned int e = 0; e < NumElmtGroups; ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int n = 0; n < 3; ++n)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, nqTot,
                            (TData *)(inptr + n * NumElmtGroups * nqTot));
                    }
                }

                StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, dfptr, df_tmp, NumElmtGroups * nqTot, m_f[0],
                    m_f[1], m_f[2], m_f[3], inptr, tmpPtr);
                SumDerivTensor3DKernel<DEFORMED, simd_t>(
                    nq0, nq1, nq2, tmpPtr[0], tmpPtr[1], tmpPtr[2], m_W[0],
                    m_W[1], m_W[2], jacptr, m_D[0], m_D[1], m_D[2],
                    tmp3.data());
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, tmp3.data(),
                    m_B[0], m_B[1], m_B[2], m_nodToModTrans, wsp0.data(),
                    wsp1.data(), wsp2.data(), outptr, 1.0);

                // Increment pointers for the next elmt group.
                inptr += nqTot;
                outptr += nmTot;
                jacptr += jacSize;
                dfptr += jacSize * ndf;
            }

            // Advance input by m_coordDim-1 componennts since have already
            // advanced one component in the above.
            inptr += nqTot * NumElmtGroups * 2;
        }
    }
};

} // namespace Nektar::Operators::detail
