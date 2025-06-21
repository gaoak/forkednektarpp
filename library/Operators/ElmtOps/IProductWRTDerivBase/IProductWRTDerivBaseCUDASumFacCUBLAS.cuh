///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseCUDASumFacCUBLAS.cuh
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

#include <StdRegions/StdExpansion.h>

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp"
#include "Operators/MathKernels/MathKernels.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseCUDASumFacCUBLASHelper.cuh"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseCUDASumFacCUBLASHelper.cuh"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTDerivBaseBlockOpImpl(const LocalRegions::ExpansionSharedPtr &exp,
                                    NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTDerivBaseBlockOp<TData>(exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(), eBasis)));

            // Fetch basis derivative data.
            m_dB.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eBasisDerivative)));

            // Fetch integration weights
            m_W.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eWeights)));

            if (m_dimension == 2)
            {
                // Fetch geometric factors.
                m_f.push_back(
                    this->m_dataWarehouse->template GetData<ExecSpace>(
                        BasisDataKey<TData>(
                            this->m_exp->GetBasis(0)->GetBasisKey(),
                            eHalfMultOnePlusZero)));
                m_f.push_back(
                    this->m_dataWarehouse->template GetData<ExecSpace>(
                        BasisDataKey<TData>(
                            this->m_exp->GetBasis(1)->GetBasisKey(),
                            eTwoOverOneMinusZero)));
            }
            else if (m_dimension == 3)
            {
                // Fetch geometric factors.
                m_f.push_back(
                    this->m_dataWarehouse->template GetData<ExecSpace>(
                        BasisDataKey<TData>(
                            this->m_exp->GetBasis(0)->GetBasisKey(),
                            eHalfMultOnePlusZero)));
                m_f.push_back(
                    this->m_dataWarehouse->template GetData<ExecSpace>(
                        BasisDataKey<TData>(
                            this->m_exp->GetBasis(1)->GetBasisKey(),
                            eHalfMultOnePlusZero)));
                m_f.push_back(
                    this->m_dataWarehouse->template GetData<ExecSpace>(
                        BasisDataKey<TData>(
                            this->m_exp->GetBasis(1)->GetBasisKey(),
                            eTwoOverOneMinusZero)));
                m_f.push_back(
                    this->m_dataWarehouse->template GetData<ExecSpace>(
                        BasisDataKey<TData>(
                            this->m_exp->GetBasis(2)->GetBasisKey(),
                            eTwoOverOneMinusZero)));
            }
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
            IProductWRTDerivBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
    std::vector<const TData *> m_dB;
    std::vector<const TData *> m_W;
    std::vector<const TData *> m_f;
    MemoryRegion<TData> m_wsp;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
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
            // Hexs
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

    size_t GetWorkspaceSize(LibUtilities::ShapeType shapeType, size_t nelmt,
                            unsigned int nm0, unsigned int nm1,
                            unsigned int nm2, unsigned int nq0,
                            unsigned int nq1, unsigned int nq2)
    {
        size_t wspsize = 0;

        if (shapeType == LibUtilities::Seg)
        {
            wspsize = 2 * nq0 * nelmt;
        }
        else if (shapeType == LibUtilities::Quad)
        {
            wspsize = 4 * nelmt * (max(nq0 * nq1, nm0 * nm1));
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = 4 * nelmt * (max(nq0 * nq1, nm0 * nm1));
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = 6 * nelmt * (max(nq0 * nq1 * nq2, nm0 * nm1 * nm2));
        }
        else if (shapeType == LibUtilities::Tet)
        {
            wspsize = 6 * nelmt * (max(nq0 * nq1 * nq2, nm0 * nm1 * nm2));
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = 6 * nelmt * (max(nq0 * nq1 * nq2, nm0 * nm1 * nm2));
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = 6 * nelmt * (max(nq0 * nq1 * nq2, nm0 * nm1 * nm2));
        }

        return wspsize;
    }

    MemoryRegion<TData> SetWorkspace(LibUtilities::ShapeType shapeType,
                                     size_t nelmt, unsigned int nm0,
                                     unsigned int nm1, unsigned int nm2,
                                     unsigned int nq0, unsigned int nq1,
                                     unsigned int nq2)
    {
        size_t wspsize =
            GetWorkspaceSize(shapeType, nelmt, nm0, nm1, nm2, nq0, nq1, nq2);

        return MemoryRegion<TData>::Create(wspsize, ExecSpace::alignment);
    }

    // Fuction definitions for each shape type.
    void SegBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0         = m_nm[0];
        const auto nq0         = m_nq[0];
        const auto dim         = m_dimension;
        const auto coordDim    = m_coordDim;
        const auto ndf         = dim * coordDim;
        const auto inblocksize = inblock.size();
        const auto nelmt       = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Fetch DerivFactor data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp =
                SetWorkspace(LibUtilities::Seg, nelmt, nm0, 0, 0, nq0, 0, 0);
        }

        // Get workspace pointer.
        auto wspptr    = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        TData *wsp2ptr = wspptr + nelmt * nq0;

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Apply derivative.
            ApplyDeriv<ExecSpace>(dim, coordDim, nelmt, nq0, nm0, inblocksize,
                                  ndf, inptr, dfptr, wspptr, m_isDeformed);

            // IProduct kernel.
            IProductWRTBaseSegSumFacKernel<ExecSpace>(
                nq0, nm0, nelmt, m_dB[0], m_W[0], jacptr, wspptr, outptr,
                wsp2ptr, m_isDeformed);

            // Increment pointers.
            inptr += m_coordDim * inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock)
    {
        const auto nm0         = m_nm[0];
        const auto nm1         = m_nm[1];
        const auto nq0         = m_nq[0];
        const auto nq1         = m_nq[1];
        const auto dim         = m_dimension;
        const auto coordDim    = m_coordDim;
        const auto ndf         = dim * coordDim;
        const auto nmTot       = m_nmTot;
        const auto nqTot       = m_nqTot;
        const auto nelmt       = inblock.GetNumElements();
        const auto inblocksize = inblock.size();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Fetch DerivFactor data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Quad, nelmt, nm0, nm1, 0, nq0,
                                 nq1, 0);
        }

        // Get workspace pointers.
        auto tmpptr    = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        TData *tmp2ptr = tmpptr + nelmt * max(nqTot, nmTot);
        TData *wspptr  = tmp2ptr + nelmt * max(nqTot, nmTot);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Apply derivative.
            ApplyDeriv<ExecSpace>(dim, coordDim, nelmt, nqTot, nmTot,
                                  inblocksize, ndf, inptr, dfptr, tmpptr,
                                  m_isDeformed);

            // IProduct.
            IProductWRTBaseQuadSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nqTot, nmTot, nelmt, m_dB[0], m_B[1],
                m_W[0], m_W[1], jacptr, tmpptr, outptr, wspptr, m_isDeformed);

            IProductWRTBaseQuadSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nqTot, nmTot, nelmt, m_B[0], m_dB[1],
                m_W[0], m_W[1], jacptr, tmp2ptr, tmpptr, wspptr, m_isDeformed);

            // Sum dimensions.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, outptr, tmpptr, outptr);

            // Increment pointers.
            inptr += m_coordDim * inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void TriBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0         = m_nm[0];
        const auto nm1         = m_nm[1];
        const auto nq0         = m_nq[0];
        const auto nq1         = m_nq[1];
        const auto dim         = m_dimension;
        const auto coordDim    = m_coordDim;
        const auto ndf         = dim * coordDim;
        const auto nmTot       = m_nmTot;
        const auto nqTot       = m_nqTot;
        const auto nelmt       = inblock.GetNumElements();
        const auto inblocksize = inblock.size();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Fetch DerivFactor data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Create CUDA Streams.
        const unsigned int nStreams = 2;
        std::vector<cudaStream_t> streams(nStreams);
        for (unsigned int s = 0; s < nStreams; ++s)
        {
            cudaStreamCreate(&streams[s]);
        }

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Tri, nelmt, nm0, nm1, 0, nq0,
                                 nq1, 0);
        }

        // Get workspace pointers.
        auto tmpptr    = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        TData *tmp2ptr = tmpptr + nelmt * max(nqTot, nmTot);
        TData *wspptr  = tmp2ptr + nelmt * max(nqTot, nmTot);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Apply derivative.
            ApplyDeriv<ExecSpace>(dim, coordDim, nelmt, nqTot, nmTot,
                                  inblocksize, ndf, inptr, dfptr, tmpptr,
                                  m_isDeformed);

            // Apply factors.
            ApplyFactorTri<ExecSpace>(nelmt, nqTot, nq0, m_f[0], m_f[1], tmpptr,
                                      tmp2ptr);

            // IProduct.
            IProductWRTBaseTriSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nqTot, nmTot, nelmt, m_dB[0], m_B[1],
                m_W[0], m_W[1], jacptr, tmpptr, outptr, wspptr, m_isDeformed,
                m_isModified, streams);

            IProductWRTBaseTriSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nqTot, nmTot, nelmt, m_B[0], m_dB[1],
                m_W[0], m_W[1], jacptr, tmp2ptr, tmpptr, wspptr, m_isDeformed,
                m_isModified, streams);

            // Sum dimensions.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, tmpptr, outptr, outptr);

            // Increment pointers.
            inptr += m_coordDim * inblock.size();
            outptr += outblock.size();
        }

        // Destroy streams.
        for (unsigned int s = 0; s < nStreams; ++s)
        {
            cudaStreamDestroy(streams[s]);
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void HexBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0         = m_nm[0];
        const auto nm1         = m_nm[1];
        const auto nm2         = m_nm[2];
        const auto nq0         = m_nq[0];
        const auto nq1         = m_nq[1];
        const auto nq2         = m_nq[2];
        const auto dim         = m_dimension;
        const auto coordDim    = m_coordDim;
        const auto ndf         = dim * coordDim;
        const auto nelmt       = inblock.GetNumElements();
        const auto inblocksize = inblock.size();
        const auto nqTot       = m_nqTot;
        const auto nmTot       = m_nmTot;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Fetch DerivFactor data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Hex, nelmt, nm0, nm1, nm2, nq0,
                                 nq1, nq2);
        }

        // Get workspace pointers.
        auto tmpptr    = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        TData *tmp2ptr = tmpptr + nelmt * max(nqTot, nmTot);
        TData *tmp3ptr = tmp2ptr + nelmt * max(nqTot, nmTot);
        TData *wspptr  = tmp3ptr + nelmt * max(nqTot, nmTot);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Apply derivative.
            ApplyDeriv<ExecSpace>(dim, coordDim, nelmt, nqTot, nmTot,
                                  inblocksize, ndf, inptr, dfptr, tmpptr,
                                  m_isDeformed);

            // IProduct.
            IProductWRTBaseHexSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_dB[0],
                m_B[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, tmpptr, outptr,
                wspptr, m_isDeformed);

            IProductWRTBaseHexSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_dB[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, tmp2ptr,
                tmpptr, wspptr, m_isDeformed);

            // Add dimensions 1,2.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, tmpptr, outptr, outptr);

            IProductWRTBaseHexSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_B[1], m_dB[2], m_W[0], m_W[1], m_W[2], jacptr, tmp3ptr,
                tmpptr, wspptr, m_isDeformed);

            // Add final dimension.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, tmpptr, outptr, outptr);

            // Increment pointers.
            inptr += m_coordDim * inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nm0         = m_nm[0];
        const auto nm1         = m_nm[1];
        const auto nm2         = m_nm[2];
        const auto nq0         = m_nq[0];
        const auto nq1         = m_nq[1];
        const auto nq2         = m_nq[2];
        const auto dim         = m_dimension;
        const auto coordDim    = m_coordDim;
        const auto ndf         = dim * coordDim;
        const auto nelmt       = inblock.GetNumElements();
        const auto inblocksize = inblock.size();
        const auto nqTot       = m_nqTot;
        const auto nmTot       = m_nmTot;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Fetch DerivFactor data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Create CUDA streams.
        std::vector<cudaStream_t> streams(nm0);
        for (unsigned int i = 0; i < nm0; ++i)
        {
            cudaStreamCreate(&streams[i]);
        }

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Prism, nelmt, nm0, nm1, nm2, nq0,
                                 nq1, nq2);
        }

        // Get workspace pointers.
        auto tmpptr    = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        TData *tmp2ptr = tmpptr + nelmt * max(nqTot, nmTot);
        TData *tmp3ptr = tmp2ptr + nelmt * max(nqTot, nmTot);
        TData *wspptr  = tmp3ptr + nelmt * max(nqTot, nmTot);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Apply derivative.
            ApplyDeriv<ExecSpace>(dim, coordDim, nelmt, nqTot, nmTot,
                                  inblocksize, ndf, inptr, dfptr, tmpptr,
                                  m_isDeformed);

            // Apply factors.
            ApplyFactorPrism<ExecSpace>(nelmt, nqTot, nq0, nq1, m_f[0], m_f[3],
                                        tmpptr, tmp3ptr);

            // IProduct.
            IProductWRTBasePrismSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_dB[0],
                m_B[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, tmpptr, outptr,
                wspptr, m_isDeformed, m_isModified, streams);

            IProductWRTBasePrismSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_dB[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, tmp2ptr,
                tmpptr, wspptr, m_isDeformed, m_isModified, streams);

            // Add dimensions 1,2.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, tmpptr, outptr, outptr);

            IProductWRTBasePrismSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_B[1], m_dB[2], m_W[0], m_W[1], m_W[2], jacptr, tmp3ptr,
                tmpptr, wspptr, m_isDeformed, m_isModified, streams);

            // Add final dimension.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, tmpptr, outptr, outptr);

            // Increment pointers.
            inptr += m_coordDim * inblock.size();
            outptr += outblock.size();
        }

        // Destroy streams.
        for (unsigned int i = 0; i < nm0; ++i)
        {
            cudaStreamDestroy(streams[i]);
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void PyrBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0         = m_nm[0];
        const auto nm1         = m_nm[1];
        const auto nm2         = m_nm[2];
        const auto nq0         = m_nq[0];
        const auto nq1         = m_nq[1];
        const auto nq2         = m_nq[2];
        const auto dim         = m_dimension;
        const auto coordDim    = m_coordDim;
        const auto ndf         = dim * coordDim;
        const auto nelmt       = inblock.GetNumElements();
        const auto inblocksize = inblock.size();
        const auto nqTot       = m_nqTot;
        const auto nmTot       = m_nmTot;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Fetch DerivFactor data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Create CUDA Streams.
        const unsigned int nStreams = 32;
        std::vector<cudaStream_t> streams(nStreams);
        for (unsigned int s = 0; s < nStreams; ++s)
        {
            cudaStreamCreate(&streams[s]);
        }

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Pyr, nelmt, nm0, nm1, nm2, nq0,
                                 nq1, nq2);
        }

        // Get workspace pointers.
        auto tmpptr    = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        TData *tmp2ptr = tmpptr + nelmt * max(nqTot, nmTot);
        TData *tmp3ptr = tmp2ptr + nelmt * max(nqTot, nmTot);
        TData *wspptr  = tmp3ptr + nelmt * max(nqTot, nmTot);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Apply derivative.
            ApplyDeriv<ExecSpace>(dim, coordDim, nelmt, nqTot, nmTot,
                                  inblocksize, ndf, inptr, dfptr, tmpptr,
                                  m_isDeformed);

            // Apply factors.
            ApplyFactorPyr<ExecSpace>(nelmt, nqTot, nq0, nq1, m_f[0], m_f[1],
                                      m_f[3], tmpptr, tmp2ptr, tmp3ptr);

            // IProduct.
            IProductWRTBasePyrSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_dB[0],
                m_B[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, tmpptr, outptr,
                wspptr, m_isDeformed, m_isModified, streams);

            IProductWRTBasePyrSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_dB[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, tmp2ptr,
                tmpptr, wspptr, m_isDeformed, m_isModified, streams);

            // Add dimensions 1,2.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, tmpptr, outptr, outptr);

            IProductWRTBasePyrSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_B[1], m_dB[2], m_W[0], m_W[1], m_W[2], jacptr, tmp3ptr,
                tmpptr, wspptr, m_isDeformed, m_isModified, streams);

            // Add final dimension.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, tmpptr, outptr, outptr);

            // Increment pointers.
            inptr += m_coordDim * inblock.size();
            outptr += outblock.size();
        }

        // Destroy streams.
        for (unsigned int s = 0; s < nStreams; ++s)
        {
            cudaStreamDestroy(streams[s]);
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void TetBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0         = m_nm[0];
        const auto nm1         = m_nm[1];
        const auto nm2         = m_nm[2];
        const auto nq0         = m_nq[0];
        const auto nq1         = m_nq[1];
        const auto nq2         = m_nq[2];
        const auto dim         = m_dimension;
        const auto coordDim    = m_coordDim;
        const auto ndf         = dim * coordDim;
        const auto nelmt       = inblock.GetNumElements();
        const auto inblocksize = inblock.size();
        const auto nqTot       = m_nqTot;
        const auto nmTot       = m_nmTot;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Fetch DerivFactor data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Create CUDA Streams.
        const unsigned int nStreams = 8;
        std::vector<cudaStream_t> streams(nStreams);
        for (unsigned int s = 0; s < nStreams; ++s)
        {
            cudaStreamCreate(&streams[s]);
        }

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Tet, nelmt, nm0, nm1, nm2, nq0,
                                 nq1, nq2);
        }

        // Get workspace pointers.
        auto tmpptr    = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        TData *tmp2ptr = tmpptr + nelmt * max(nqTot, nmTot);
        TData *tmp3ptr = tmp2ptr + nelmt * max(nqTot, nmTot);
        TData *wspptr  = tmp3ptr + nelmt * max(nqTot, nmTot);

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Apply derivative.
            ApplyDeriv<ExecSpace>(dim, coordDim, nelmt, nqTot, nmTot,
                                  inblocksize, ndf, inptr, dfptr, tmpptr,
                                  m_isDeformed);

            // Apply factors.
            ApplyFactorTet<ExecSpace>(nelmt, nqTot, nq0, nq1, m_f[0], m_f[1],
                                      m_f[2], m_f[3], tmpptr, tmp2ptr, tmp3ptr);

            // IProduct.
            IProductWRTBaseTetSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_dB[0],
                m_B[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, tmpptr, outptr,
                wspptr, m_isDeformed, m_isModified, streams);

            IProductWRTBaseTetSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_dB[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, tmp2ptr,
                tmpptr, wspptr, m_isDeformed, m_isModified, streams);

            // Add dimensions 1,2.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, tmpptr, outptr, outptr);

            IProductWRTBaseTetSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_B[1], m_dB[2], m_W[0], m_W[1], m_W[2], jacptr, tmp3ptr,
                tmpptr, wspptr, m_isDeformed, m_isModified, streams);

            // Add final dimension.
            Nektar::addKernel<ExecSpace>(nelmt * nmTot, tmpptr, outptr, outptr);

            // Increment pointers.
            inptr += m_coordDim * inblock.size();
            outptr += outblock.size();
        }

        // Destroy streams.
        for (unsigned int s = 0; s < nStreams; ++s)
        {
            cudaStreamDestroy(streams[s]);
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
