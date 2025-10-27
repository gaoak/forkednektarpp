///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseCUDASumFacCUBLAS.cuh
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

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseCUDASumFacCUBLASHelper.cuh"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTBaseBlockOpImpl : public IProductWRTBaseBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTBaseBlockOpImpl(const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTBaseBlockOp<TData>(exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
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

            // Fetch integration weights
            m_W.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eWeights)));
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
            IProductWRTBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
    std::vector<const TData *> m_W;
    std::vector<const unsigned int *> m_index;
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
            wspsize = nq0 * nelmt;
        }
        else if (shapeType == LibUtilities::Quad)
        {
            wspsize = 2 * nelmt * (max(nq0 * nq1, nm0 * nm1));
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = 2 * nelmt * (max(nq0 * nq1, nm0 * nm1));
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = 3 * nelmt * (max(nq0 * nq1 * nq2, nm0 * nm1 * nm2));
        }
        else if (shapeType == LibUtilities::Tet)
        {
            wspsize = nelmt * (max(nq0 * nq1 * nq2,
                                   nq2 * nm0 * (2 * nm1 - nm0 + 1) / 2) +
                               nq2 * nq1 * nm0);
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = nelmt * nq2 * (max(nq0 * nq1, nm0 * nm1)) +
                      nq1 * nq2 * nelmt * nm0;
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = nelmt * nq2 * (max(nq0 * nq1, nm0 * nm1)) +
                      nq1 * nq2 * nelmt * nm0;
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
        const auto nm0   = m_nm[0];
        const auto nq0   = m_nq[0];
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp =
                SetWorkspace(LibUtilities::Seg, nelmt, nm0, 0, 0, nq0, 0, 0);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                      inblock.GetInterleaveWidth(),
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            // IProduct.
            IProductWRTBaseSegSumFacKernel<ExecSpace>(
                nq0, nm0, nelmt, m_B[0], m_W[0], jacptr, inptr, outptr, wspptr,
                m_isDeformed);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nqTot = m_nqTot;
        const auto nmTot = m_nmTot;
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Quad, nelmt, nm0, nm1, 0, nq0,
                                 nq1, 0);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                      inblock.GetInterleaveWidth(),
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            // IProduct.
            IProductWRTBaseQuadSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nqTot, nmTot, nelmt, m_B[0], m_B[1], m_W[0],
                m_W[1], jacptr, inptr, outptr, wspptr, m_isDeformed);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void TriBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nelmt = inblock.GetNumElements();
        const auto nqTot = m_nqTot;
        const auto nmTot = m_nmTot;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

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
            m_wsp = SetWorkspace(LibUtilities::Tri, nelmt, nm0, nm1, 0, nq0,
                                 nq1, 0);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                      inblock.GetInterleaveWidth(),
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            // IProduct.
            IProductWRTBaseTriSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nqTot, nmTot, nelmt, m_B[0], m_B[1], m_W[0],
                m_W[1], jacptr, inptr, outptr, wspptr, m_isDeformed,
                m_isModified, streams);

            // Increment pointers.
            inptr += inblock.size();
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
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nm2   = m_nm[2];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto nelmt = inblock.GetNumElements();
        const auto nqTot = m_nqTot;
        const auto nmTot = m_nmTot;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Hex, nelmt, nm0, nm1, nm2, nq0,
                                 nq1, nq2);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                      inblock.GetInterleaveWidth(),
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            // IProduct.
            IProductWRTBaseHexSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_B[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, inptr, outptr,
                wspptr, m_isDeformed);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nm2   = m_nm[2];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto nelmt = inblock.GetNumElements();
        const auto nqTot = m_nqTot;
        const auto nmTot = m_nmTot;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Create CUDA streams.
        std::vector<cudaStream_t> streams(nm0);
        for (unsigned int s = 0; s < nm0; ++s)
        {
            cudaStreamCreate(&streams[s]);
        }

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Prism, nelmt, nm0, nm1, nm2, nq0,
                                 nq1, nq2);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                      inblock.GetInterleaveWidth(),
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            // IProduct.
            IProductWRTBasePrismSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_B[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, inptr, outptr,
                wspptr, m_isDeformed, m_isModified, streams);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Destroy streams.
        for (unsigned int s = 0; s < nm0; ++s)
        {
            cudaStreamDestroy(streams[s]);
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void PyrBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nm2   = m_nm[2];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto nelmt = inblock.GetNumElements();
        const auto nqTot = m_nqTot;
        const auto nmTot = m_nmTot;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

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
            m_wsp = SetWorkspace(LibUtilities::Pyr, nelmt, nm0, nm1, nm2, nq0,
                                 nq1, nq2);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                      inblock.GetInterleaveWidth(),
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            // IProduct.
            IProductWRTBasePyrSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_B[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, inptr, outptr,
                wspptr, m_isDeformed, m_isModified, streams);

            // Increment pointers.
            inptr += inblock.size();
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
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nm2   = m_nm[2];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto nelmt = inblock.GetNumElements();
        const auto nqTot = m_nqTot;
        const auto nmTot = m_nmTot;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian data.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

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

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                      inblock.GetInterleaveWidth(),
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            // IProduct.
            IProductWRTBaseTetSumFacKernel<ExecSpace>(
                nq0, nm0, nq1, nm1, nq2, nm2, nqTot, nmTot, nelmt, m_B[0],
                m_B[1], m_B[2], m_W[0], m_W[1], m_W[2], jacptr, inptr, outptr,
                wspptr, m_isDeformed, m_isModified, streams);

            // Increment pointers.
            inptr += inblock.size();
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
