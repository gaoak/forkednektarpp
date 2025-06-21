///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivCUDASumFacCUBLAS.cuh
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

#include "Operators/ElmtOps/PhysDeriv/PhysDerivCUDASumFacCUBLASHelper.cuh"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class PhysDerivBlockOpImpl : public PhysDerivBlockOp<TData>
{
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
        m_nqTot     = exp->GetTotPoints();

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_D.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(exp->GetBasis(d)->GetBasisKey(),
                                    eDerivative)));
        }

        if (m_dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                    eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                    eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                    eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                    eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                    eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<TData>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                    eTwoOverOneMinusZero)));
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
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nqTot;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_D;
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

    size_t GetWorkspaceSize(LibUtilities::ShapeType shapeType, size_t nelmt,
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
            wspsize = 2 * nelmt * nq0 * nq1;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = 2 * nelmt * nq0 * nq1;
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = 3 * nelmt * nq0 * nq1 * nq2;
        }
        else if (shapeType == LibUtilities::Tet)
        {
            wspsize = 3 * nelmt * nq0 * nq1 * nq2;
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = 3 * nelmt * nq0 * nq1 * nq2;
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = 3 * nelmt * nq0 * nq1 * nq2;
        }

        return wspsize;
    }

    MemoryRegion<TData> SetWorkspace(LibUtilities::ShapeType shapeType,
                                     size_t nelmt, unsigned int nq0,
                                     unsigned int nq1, unsigned int nq2)
    {
        size_t wspsize = GetWorkspaceSize(shapeType, nelmt, nq0, nq1, nq2);

        return MemoryRegion<TData>::Create(wspsize, ExecSpace::alignment);
    }

    void SegBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nq0   = m_nq[0];
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch deriv factors data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Seg, nelmt, nq0, 0, 0);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nelmt, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDerivSegKernel<ExecSpace>(m_coordDim, m_dimension, nq0, nelmt,
                                          outblock.size(), m_D[0], dfptr, inptr,
                                          outptr, wspptr, m_isDeformed);

            // Increment pointers.
            inptr += inblock.size();
            outptr += m_coordDim * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock)
    {
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nqTot = m_nqTot;
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch deriv factors data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Quad, nelmt, nq0, nq1, 0);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nelmt, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDerivQuadKernel<ExecSpace>(m_coordDim, m_dimension, nq0, nq1,
                                           nqTot, nelmt, outblock.size(),
                                           m_D[0], m_D[1], dfptr, inptr, outptr,
                                           wspptr, m_isDeformed);

            // Increment pointers.
            inptr += inblock.size();
            outptr += m_coordDim * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void TriBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nqTot = m_nqTot;
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch deriv factors data.
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
            m_wsp = SetWorkspace(LibUtilities::Tri, nelmt, nq0, nq1, 0);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nelmt, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDerivTriKernel<ExecSpace>(
                m_coordDim, m_dimension, nq0, nq1, nqTot, nelmt,
                outblock.size(), m_D[0], m_D[1], m_f[0], m_f[1], dfptr, inptr,
                outptr, wspptr, m_isDeformed, streams);

            // Increment pointers.
            inptr += inblock.size();
            outptr += m_coordDim * outblock.size();
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
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto nqTot = m_nqTot;
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch deriv factors data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Hex, nelmt, nq0, nq1, nq2);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nelmt, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDerivHexKernel<ExecSpace>(m_coordDim, m_dimension, nq0, nq1,
                                          nq2, nqTot, nelmt, outblock.size(),
                                          m_D[0], m_D[1], m_D[2], dfptr, inptr,
                                          outptr, wspptr, m_isDeformed);

            // Increment pointers.
            inptr += inblock.size();
            outptr += m_coordDim * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto nqTot = m_nqTot;
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch deriv factors data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Prism, nelmt, nq0, nq1, nq2);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nelmt, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDerivPrismKernel<ExecSpace>(
                m_coordDim, m_dimension, nq0, nq1, nq2, nqTot, nelmt,
                outblock.size(), m_D[0], m_D[1], m_D[2], m_f[0], m_f[3], dfptr,
                inptr, outptr, wspptr, m_isDeformed);

            // Increment pointers.
            inptr += inblock.size();
            outptr += m_coordDim * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void PyrBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto nqTot = m_nqTot;
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch deriv factors data.
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
            m_wsp = SetWorkspace(LibUtilities::Pyr, nelmt, nq0, nq1, nq2);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nelmt, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDerivPyrKernel<ExecSpace>(
                m_coordDim, m_dimension, nq0, nq1, nq2, nqTot, nelmt,
                outblock.size(), m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[3],
                dfptr, inptr, outptr, wspptr, m_isDeformed, streams);

            // Increment pointers.
            inptr += inblock.size();
            outptr += m_coordDim * outblock.size();
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
        const auto nq0   = m_nq[0];
        const auto nq1   = m_nq[1];
        const auto nq2   = m_nq[2];
        const auto nqTot = m_nqTot;
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch deriv factors data.
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Set workspace.
        if (m_wsp.size() == 0)
        {
            m_wsp = SetWorkspace(LibUtilities::Tet, nelmt, nq0, nq1, nq2);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nelmt, inblock.GetNumData(),
                (TData *)inptr);

            // Calculate derivative.
            PhysDerivTetKernel<ExecSpace>(
                m_coordDim, m_dimension, nq0, nq1, nq2, nqTot, nelmt,
                outblock.size(), m_D[0], m_D[1], m_D[2], m_f[0], m_f[1], m_f[2],
                m_f[3], dfptr, inptr, outptr, wspptr, m_isDeformed);

            // Increment pointers.
            inptr += inblock.size();
            outptr += m_coordDim * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
