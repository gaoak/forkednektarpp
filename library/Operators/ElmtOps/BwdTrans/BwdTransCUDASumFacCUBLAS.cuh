///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransCUDASumFacCUBLAS.cuh
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

#include "Operators/ElmtOps/BwdTrans/BwdTransBlockOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/BwdTrans/BwdTransCUDASumFacCUBLASHelper.cuh"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BwdTransBlockOpImpl : public BwdTransBlockOp<TData>
{
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
        m_nmTot     = exp->GetNcoeffs();

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
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BwdTransBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_nmTot;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
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

    size_t GetSharedWorkspaceSize(LibUtilities::ShapeType shapeType,
                                  size_t nelmt, unsigned int nm0,
                                  unsigned int nm1, unsigned int nm2,
                                  unsigned int nq0, unsigned int nq1,
                                  unsigned int nq2)
    {
        size_t wspsize = 0;

        if (shapeType == LibUtilities::Seg)
        {
            wspsize = nq0 * nelmt;
        }
        else if (shapeType == LibUtilities::Quad)
        {
            wspsize = nq0 * nm1 * nelmt;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = nq1 * nm0 * nelmt;
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = (nm1 * nm2 * nq0 + nm2 * nq0 * nq1) * nelmt;
        }
        else if (shapeType == LibUtilities::Tet)
        {
            wspsize =
                (nq2 * nm0 * (2 * nm1 - nm0 + 1) / 2 + nq2 * nq1 * nm0) * nelmt;
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = (nq2 * nm1 + nq1 * nq2) * nm0 * nelmt;
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = (nm1 + nq1) * nm0 * nq2 * nelmt;
        }

        return wspsize;
    }

    MemoryRegion<TData> SetWorkspace(LibUtilities::ShapeType shapeType,
                                     size_t nelmt, unsigned int nm0,
                                     unsigned int nm1, unsigned int nm2,
                                     unsigned int nq0, unsigned int nq1,
                                     unsigned int nq2)
    {
        size_t wspsize = GetSharedWorkspaceSize(shapeType, nelmt, nm0, nm1, nm2,
                                                nq0, nq1, nq2);

        return MemoryRegion<TData>(wspsize);
    }

    // Fuction definitions for each shape type.
    void SegBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nq0   = m_nq[0];
        const auto B0    = m_B[0];
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch handle.
        auto handle = NekHandle<ExecSpace>::GetInstance();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Perform matrix-matrix multiply.
            NekGemm(handle, "N", "N", nq0, nelmt, nm0, 1.0, m_B[0], nq0, inptr,
                    nm0, 0.0, outptr, nq0);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto B0    = m_B[0];
        const auto B1    = m_B[1];
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch handle.
        auto handle = NekHandle<ExecSpace>::GetInstance();

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Quad, nelmt, nm0, nm1, 0, nq0,
                                 nq1, 0);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            NekGemm(handle, "N", "N", nq0, nm1 * nelmt, nm0, 1.0, B0, nq0,
                    inptr, nm0, 0.0, wspptr, nq0);

            NekGemmStridedBatched(handle, "N", "T", nq0, nq1, nm1, 1.0, wspptr,
                                  nq0, nq0 * nm1, B1, nq1, 0, 0.0, outptr, nq0,
                                  nq0 * nq1, nelmt);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    };

    void TriBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto B0    = m_B[0];
        const auto B1    = m_B[1];
        const auto nmTot = m_nmTot;
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch handle.
        auto handle = NekHandle<ExecSpace>::GetInstance();

        // Create CUDA Streams.
        const unsigned int nStreams = 2;
        cudaStream_t streams[nStreams];
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

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            unsigned int mode = 0;
            for (unsigned int i = 0; i < nm0; i++)
            {
                // Set stream.
                cublasSetStream(handle, streams[i % nStreams]);

                NekGemm(handle, "N", "N", nq1, nelmt, nm1 - i, 1.0,
                        B1 + mode * nq1, nq1, inptr + mode, nmTot, 0.0,
                        wspptr + i * nq1 * nelmt, nq1);

                mode += nm1 - i;
            }

            for (unsigned int s = 0; s < nStreams; ++s)
            {
                cudaStreamSynchronize(streams[s]);
            }

            // Set back to default stream.
            cublasSetStream(handle, 0);

            // Fix for modified basis by splitting top vertex mode.
            if (m_isModified)
            {
                vertexFactorTri<ExecSpace, TData>(nelmt, nmTot, nq1, wspptr,
                                                  inptr, B1);
            }

            NekGemm(handle, "N", "T", nq0, nq1 * nelmt, nm0, 1.0, B0, nq0,
                    wspptr, nq1 * nelmt, 0.0, outptr, nq0);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Destroy streams.
        for (unsigned int s = 0; s < nStreams; ++s)
        {
            cudaStreamDestroy(streams[s]);
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    };

    void HexBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nm2   = m_nm[2];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto B0    = m_B[0];
        const auto B1    = m_B[1];
        const auto B2    = m_B[2];
        const auto nelmt = inblock.GetNumElementsWithPadding();
        const auto nmTot = m_nmTot;

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch handle.
        auto handle = NekHandle<ExecSpace>::GetInstance();

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Hex, nelmt, nm0, nm1, nm2, nq0,
                                 nq1, nq2);
        }

        // Get workspace pointer.
        auto wspptr  = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        auto wsp2ptr = wspptr + nm0 * nm1 * nq2 * nelmt;

        const auto instride  = nm0 * nm1;
        const auto wspstride = nq2 * nelmt;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            NekGemmStridedBatched(handle, "N", "T", nq2, nm0 * nm1, nm2, 1.0,
                                  B2, nq2, 0, inptr, instride, nmTot, 0.0,
                                  wspptr, wspstride, nq2, nelmt);

            NekGemm(handle, "N", "T", nq1, nq2 * nelmt * nm0, nm1, 1.0, B1, nq1,
                    wspptr, nq2 * nelmt * nm0, 0.0, wsp2ptr, nq1);

            NekGemm(handle, "N", "T", nq0, nq1 * nq2 * nelmt, nm0, 1.0, B0, nq0,
                    wsp2ptr, nq1 * nq2 * nelmt, 0.0, outptr, nq0);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
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
        const auto B0    = m_B[0];
        const auto B1    = m_B[1];
        const auto B2    = m_B[2];
        const auto nmTot = m_nmTot;
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch handle.
        auto handle = NekHandle<ExecSpace>::GetInstance();

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
        auto wspptr  = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        auto wsp2ptr = wspptr + nm0 * nm1 * nq2 * nelmt;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Initialise counters.
            unsigned int mode  = 0;
            unsigned int mode1 = 0;
            for (unsigned int i = 0; i < nm0; ++i)
            {
                // Strides.
                unsigned int k         = nm2 - i;
                unsigned int strideWsp = nq2 * nelmt * nm0;

                // Pointers to corresponding block for nm0 iteration.
                const TData *tmpB2 = B2 + mode * nq2;
                const TData *tmpIn = inptr + mode1;
                TData *tmpWsp      = wspptr + i * nq2 * nelmt;

                // Set stream.
                cublasSetStream(handle, streams[i]);

                // Transform in B2.
                NekGemmStridedBatched(handle, "N", "N", nq2, nelmt, k, 1.0,
                                      tmpB2, nq2, 0, tmpIn, nmTot, k, 0.0,
                                      tmpWsp, nq2, strideWsp, nm1);

                // Advance mode counters.
                mode1 += k * nm1;
                mode += k;
            }

            // Synchronize all streams.
            for (unsigned int s = 0; s < nm0; ++s)
            {
                cudaStreamSynchronize(streams[s]);
            }

            // Set back to default stream.
            cublasSetStream(handle, 0);

            // Fix for modified basis by splitting top vertex mode.
            if (m_isModified)
            {
                vertexFactorPrism<ExecSpace>(nelmt, nmTot, nq2, nm0, nm1, nm2,
                                             wspptr, inptr, B2);
            }

            // Transform in B1.
            NekGemm(handle, "N", "T", nq1, nq2 * nelmt * nm0, nm1, 1.0, B1, nq1,
                    wspptr, nq2 * nelmt * nm0, 0.0, wsp2ptr, nq1);

            // Transform in B0.
            NekGemm(handle, "N", "T", nq0, nq1 * nq2 * nelmt, nm0, 1.0, B0, nq0,
                    wsp2ptr, nq1 * nq2 * nelmt, 0.0, outptr, nq0);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Destroy streams.
        for (unsigned int s = 0; s < nm0; ++s)
        {
            cudaStreamDestroy(streams[s]);
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void PyrBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nm2   = m_nm[2];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto B0    = m_B[0];
        const auto B1    = m_B[1];
        const auto B2    = m_B[2];
        const auto nmTot = m_nmTot;
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch handle.
        auto handle = NekHandle<ExecSpace>::GetInstance();

        // Create CUDA streams.
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
        auto wspptr  = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        auto wsp2ptr = wspptr + nq2 * nm1 * nm0 * nelmt;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Perform summation over '2' direction.
            unsigned int mode  = 0;
            unsigned int mode1 = 0;
            unsigned int cnt   = 0;
            for (unsigned int i = 0; i < nm0; ++i)
            {
                for (unsigned int j = 0; j < nm1; ++j, ++cnt)
                {
                    // Set stream.
                    cublasSetStream(handle, streams[cnt % nStreams]);

                    unsigned int ijmax = max(i, j);

                    NekGemm(handle, "N", "N", nq2, nelmt, nm2 - ijmax, 1.0,
                            B2 + mode * nq2, nq2, inptr + mode1, nmTot, 0.0,
                            wspptr + cnt * nq2 * nelmt, nq2);

                    mode += nm2 - ijmax;
                    mode1 += nm2 - ijmax;
                }

                // Increment mode in case nm1 != nm2.
                for (unsigned int j = nm1; j < nm2; ++j)
                {
                    mode += nm2 - j;
                }
            }

            // Synchronize all streams.
            for (unsigned int s = 0; s < nStreams; ++s)
            {
                cudaStreamSynchronize(streams[s]);
            }

            // Fix for modified basis by splitting top vertex mode.
            if (m_isModified)
            {
                vertexFactorPyr<ExecSpace>(nelmt, nmTot, nq2, nm1, wspptr,
                                           inptr, B2);
            }

            // Summation over '1' direction.
            mode = 0;
            for (unsigned int i = 0; i < nm0; ++i)
            {
                // Set stream.
                cublasSetStream(handle, streams[i % nStreams]);

                NekGemm(handle, "N", "T", nq1, nq2 * nelmt, nm1, 1.0, B1, nq1,
                        wspptr + mode * nq2 * nelmt, nq2 * nelmt, 0.0,
                        wsp2ptr + i * nq1 * nq2 * nelmt, nq1);

                mode += nm1;
            }

            // Synchronize all streams.
            for (unsigned int s = 0; s < nStreams; ++s)
            {
                cudaStreamSynchronize(streams[s]);
            }

            // Set back to default stream.
            cublasSetStream(handle, 0);

            // Summation over '0' direction.
            NekGemm(handle, "N", "T", nq0, nq1 * nq2 * nelmt, nm0, 1.0, B0, nq0,
                    wsp2ptr, nq1 * nq2 * nelmt, 0.0, outptr, nq0);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Destroy streams.
        for (unsigned int s = 0; s < nStreams; ++s)
        {
            cudaStreamDestroy(streams[s]);
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    };

    void TetBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nm0   = m_nm[0];
        const auto nm1   = m_nm[1];
        const auto nm2   = m_nm[2];
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto B0    = m_B[0];
        const auto B1    = m_B[1];
        const auto B2    = m_B[2];
        const auto nmTot = m_nmTot;
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch handle.
        auto handle = NekHandle<ExecSpace>::GetInstance();

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
        auto wspptr  = m_wsp.template GetPtr<MemSpace, WriteOnly>();
        auto wsp2ptr = wspptr + nq2 * nm0 * (2 * nm1 - nm0 + 1) / 2 * nelmt;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Summation over '2' direction.
            unsigned int mode  = 0;
            unsigned int mode1 = 0;
            unsigned int cnt   = 0;
            for (unsigned int i = 0; i < nm0; ++i)
            {
                for (unsigned int j = 0; j < nm1 - i; ++j, ++cnt)
                {
                    // Set stream.
                    cublasSetStream(handle, streams[cnt % nStreams]);

                    NekGemm(handle, "N", "N", nq2, nelmt, nm2 - i - j, 1.0,
                            B2 + mode * nq2, nq2, inptr + mode1, nmTot, 0.0,
                            wspptr + cnt * nq2 * nelmt, nq2);

                    mode += nm2 - i - j;
                    mode1 += nm2 - i - j;
                }

                // Increment mode in case nm1 != nm2
                mode += (nm2 - nm1) * (nm2 - nm1 + 1) / 2;
            }

            // Synchronize all streams.
            for (unsigned int s = 0; s < nStreams; ++s)
            {
                cudaStreamSynchronize(streams[s]);
            }

            // Fix for modified basis by splitting top vertex mode.
            if (m_isModified)
            {
                vertexFactorTet1<ExecSpace>(nelmt, nmTot, nq2, nm1, wspptr,
                                            inptr, B2);
            }

            // Summation over '1' direction.
            mode = 0;
            for (unsigned int i = 0; i < nm0; i++)
            {
                cublasSetStream(handle, streams[i % nStreams]);

                NekGemm(handle, "N", "T", nq1, nq2 * nelmt, nm1 - i, 1.0,
                        B1 + mode * nq1, nq1, wspptr + mode * nq2 * nelmt,
                        nq2 * nelmt, 0.0, wsp2ptr + i * nq1 * nq2 * nelmt, nq1);

                mode += nm1 - i;
            }

            // Synchronize all streams.
            for (unsigned int s = 0; s < nStreams; ++s)
            {
                cudaStreamSynchronize(streams[s]);
            }

            // Set back to default stream.
            cublasSetStream(handle, 0);

            if (m_isModified)
            {
                vertexFactorTet2<ExecSpace>(nelmt, nq1, nq2, wspptr, wsp2ptr,
                                            B1);
            }

            // Summation over '0' direction
            NekGemm(handle, "N", "T", nq0, nq1 * nq2 * nelmt, nm0, 1.0, B0, nq0,
                    wsp2ptr, nq1 * nq2 * nelmt, 0.0, outptr, nq0);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Destroy streams.
        for (unsigned int s = 0; s < nStreams; ++s)
        {
            cudaStreamDestroy(streams[s]);
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    };
};

} // namespace Nektar::Operators::detail
