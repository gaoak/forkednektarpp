///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondOpImpl.hpp
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

#include <LocalRegions/MatrixKey.h>
#include <MultiRegions/ContField.h>

#include <MultiRegions/BndCondOps/RobBndCond/RobBndCondOp.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/BndCondOps/RobBndCond/RobBndCondDeviceKernels.hpp>
#include <MultiRegions/BndCondOps/RobBndCond/RobBndCondSerialAVXKernels.hpp>

namespace Nektar::MultiRegions::detail
{

template <typename ExecSpace, typename TData>
class RobBndCondOpImpl : public RobBndCondOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    RobBndCondOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : RobBndCondOp<TData>(expansionList, components)
    {
        auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();

        // Set mapping to skip over padding elements, and record which block
        // each element belongs to and where that block starts.
        size_t i = 0, j = 0, n = 0;

        std::vector<size_t> alignmentMap(expansionList->GetNcoeffs());
        std::vector<unsigned int> elmtBlock(expansionList->GetExpSize());
        std::vector<size_t> blockStart;
        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                expansionList);
        for (unsigned int blk = 0; blk < blockAttr.size(); ++blk)
        {
            auto &block          = blockAttr[blk];
            const auto ncoeff    = block.GetNumData();
            const auto nelmt     = block.GetNumElements();
            const auto nPadElmts = block.GetNumPaddingElements();
            blockStart.push_back(j);
            for (size_t e = 0; e < nelmt; e++)
            {
                elmtBlock[n++] = blk;
                for (unsigned int c = 0; c < ncoeff; c++)
                {
                    alignmentMap[i++] = j++;
                }
            }
            j += nPadElmts * ncoeff;
        }

        const auto nBlocks = blockAttr.size();
        m_dimension        = expansionList->GetExp(0)->GetShapeDimension();
        m_blkNumEdge.assign(nBlocks, 0u);
        m_mat.resize(nBlocks);
        m_map.resize(nBlocks);
        m_offset.resize(nBlocks);

        // Edges are bucketed by the block of the element they belong to, with
        // offsets relative to that block, so each block is applied on its own
        // stream.
        if (m_dimension == 1)
        {
            std::vector<std::vector<TData>> mat(nBlocks);
            std::vector<std::vector<size_t>> map(nBlocks);
            std::vector<std::vector<size_t>> offset(nBlocks);
            for (auto &r : robinBCInfo)
            {
                auto e      = r.first;
                auto expPtr = this->m_expansionList->GetExp(e);
                auto blk    = elmtBlock[e];

                ASSERTL1(expPtr->IsBoundaryInteriorExpansion(),
                         "Not set up for non boundary-interior expansions");

                for (auto rBC = r.second; rBC; rBC = rBC->next)
                {
                    auto primCoeffs = rBC->m_robinPrimitiveCoeffs;
                    auto vertid     = rBC->m_robinID;
                    mat[blk].push_back(primCoeffs[0]);
                    map[blk].push_back(expPtr->GetVertexMap(vertid));
                    offset[blk].push_back(
                        alignmentMap[this->m_expansionList->GetCoeff_Offset(
                            e)] -
                        blockStart[blk]);
                    m_blkNumEdge[blk]++;
                    m_nBndEdge++;
                }
            }

            for (unsigned int blk = 0; blk < nBlocks; ++blk)
            {
                if (m_blkNumEdge[blk] == 0)
                {
                    continue;
                }

                m_mat[blk] = LibUtilities::MemoryRegion<
                    TData>::template FromVector<MemSpace, TData>(mat[blk]);
                m_map[blk] = LibUtilities::MemoryRegion<
                    size_t>::template FromVector<MemSpace, size_t>(map[blk]);
                m_offset[blk] = LibUtilities::MemoryRegion<
                    size_t>::template FromVector<MemSpace, size_t>(offset[blk]);
            }
        }
        else if (m_dimension == 2)
        {
            m_sign.resize(nBlocks);
            m_nEdgeCoeff.resize(nBlocks);
            m_matOffset.resize(nBlocks);
            m_mapOffset.resize(nBlocks);

            std::vector<std::vector<TData>> mat(nBlocks);
            std::vector<std::vector<size_t>> map(nBlocks);
            std::vector<std::vector<int>> sign(nBlocks);
            std::vector<std::vector<unsigned int>> nEdgeCoeff(nBlocks);
            std::vector<std::vector<size_t>> offset(nBlocks);
            std::vector<std::vector<size_t>> matOffset(nBlocks);
            std::vector<std::vector<size_t>> mapOffset(nBlocks);
            for (auto &r : robinBCInfo)
            {
                auto e      = r.first;
                auto expPtr = this->m_expansionList->GetExp(e);
                auto blk    = elmtBlock[e];

                ASSERTL1(expPtr->IsBoundaryInteriorExpansion(),
                         "Not set up for non boundary-interior expansions");

                for (auto rBC = r.second; rBC; rBC = rBC->next)
                {
                    auto primCoeffs = rBC->m_robinPrimitiveCoeffs;
                    auto edgeid     = rBC->m_robinID;
                    auto orient     = expPtr->GetTraceOrient(edgeid);
                    auto edgeExp    = expPtr->GetTraceExp(edgeid);
                    auto ncoeff     = edgeExp->GetNcoeffs();
                    m_nmaxcoeff = std::max(m_nmaxcoeff, (unsigned int)ncoeff);

                    // Initialize map and sign array
                    Array<OneD, unsigned int> tmpMap(ncoeff); // size_t
                    Array<OneD, int> tmpSign(ncoeff);
                    expPtr->GetTraceToElementMap(edgeid, tmpMap, tmpSign,
                                                 orient);
                    mapOffset[blk].push_back(map[blk].size());
                    map[blk].insert(map[blk].end(), tmpMap.begin(),
                                    tmpMap.end());
                    sign[blk].insert(sign[blk].end(), tmpSign.begin(),
                                     tmpSign.end());

                    // Initialize mat array
                    StdRegions::VarCoeffMap varcoeffs;
                    varcoeffs[StdRegions::eVarCoeffMass] = primCoeffs;
                    LocalRegions::MatrixKey mkey(
                        StdRegions::eMass, LibUtilities::eSegment, *edgeExp,
                        StdRegions::NullConstFactorMap, varcoeffs);
                    DNekScalMat &edgeMat = *edgeExp->GetLocMatrix(mkey);
                    matOffset[blk].push_back(mat[blk].size());
                    mat[blk].insert(mat[blk].end(), edgeMat.GetRawPtr(),
                                    edgeMat.GetRawPtr() + ncoeff * ncoeff);

                    // Update offset array
                    nEdgeCoeff[blk].push_back(ncoeff);
                    offset[blk].push_back(
                        alignmentMap[this->m_expansionList->GetCoeff_Offset(
                            e)] -
                        blockStart[blk]);
                    m_blkNumEdge[blk]++;
                    m_nBndEdge++;
                }
            }

            for (unsigned int blk = 0; blk < nBlocks; ++blk)
            {
                if (m_blkNumEdge[blk] == 0)
                {
                    continue;
                }

                m_mat[blk] = LibUtilities::MemoryRegion<
                    TData>::template FromVector<MemSpace, TData>(mat[blk]);
                m_map[blk] = LibUtilities::MemoryRegion<
                    size_t>::template FromVector<MemSpace, size_t>(map[blk]);
                m_sign[blk] = LibUtilities::MemoryRegion<
                    int>::template FromVector<MemSpace, int>(sign[blk]);
                m_nEdgeCoeff[blk] = LibUtilities::MemoryRegion<unsigned int>::
                    template FromVector<MemSpace, unsigned int>(
                        nEdgeCoeff[blk]);
                m_offset[blk] = LibUtilities::MemoryRegion<
                    size_t>::template FromVector<MemSpace, size_t>(offset[blk]);
                m_matOffset[blk] =
                    LibUtilities::MemoryRegion<size_t>::template FromVector<
                        MemSpace, size_t>(matOffset[blk]);
                m_mapOffset[blk] =
                    LibUtilities::MemoryRegion<size_t>::template FromVector<
                        MemSpace, size_t>(mapOffset[blk]);
            }
        }
        else if (m_dimension == 3)
        {
            for ([[maybe_unused]] auto &r : robinBCInfo)
            {
                NEKERROR(ErrorUtil::efatal,
                         "RobBndCondOpImpl: 3D Operator not yet implemented");
            }
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<RobBndCondOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    // Per-block edge tables, with offsets relative to the block. Empty for
    // blocks without a Robin edge.
    std::vector<LibUtilities::MemoryRegion<TData>> m_mat;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_map;
    std::vector<LibUtilities::MemoryRegion<int>> m_sign;
    std::vector<LibUtilities::MemoryRegion<unsigned int>> m_nEdgeCoeff;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_offset;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_matOffset;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_mapOffset;
    std::vector<size_t> m_blkNumEdge;

    unsigned int m_dimension = 0;
    unsigned int m_nmaxcoeff = 0;
    size_t m_nBndEdge        = 0;

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
                 LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        // Return if no Robin boundary condition.
        if (m_nBndEdge == 0)
        {
            return;
        }

        ASSERTL0(in.GetNumComponents() == 1,
                 "Not yet set up for multiple components");
        ASSERTL1(in.GetBlocks().size() == m_blkNumEdge.size(),
                 "Block count differs from the one the edges were built for");

        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            const auto nEdge = m_blkNumEdge[blk];
            if (nEdge == 0)
            {
                continue;
            }

            const unsigned int streamID = blk + 1;

            auto &inBlk  = in.GetBlocks()[blk];
            auto &outBlk = out.GetBlocks()[blk];
            auto inPtr   = inBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto outPtr = outBlk.template GetPtr<MemSpace, ReadWrite>(streamID);

            // if block is interlaced deInterleave block since currently
            // mapping set up assuming serial alignment
            auto inWidth = inBlk.GetInterleaveWidth();
            LibUtilities::ReshapeStorage<ExecSpace>(
                1u, inWidth, inBlk.GetNumElementsWithPadding(),
                inBlk.GetNumData(), inPtr, streamID);
            auto outWidth = outBlk.GetInterleaveWidth();
            LibUtilities::ReshapeStorage<ExecSpace>(
                1u, outWidth, outBlk.GetNumElementsWithPadding(),
                outBlk.GetNumData(), outPtr, streamID);

            // Get pointers.
            auto matPtr =
                m_mat[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
            auto mapPtr =
                m_map[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
            auto offsetPtr =
                m_offset[blk].template GetPtr<MemSpace, ReadOnly>(streamID);

            // Apply Robin boundary conditions.
            if (m_dimension == 1)
            {
                RobBndCond1DKernel<ExecSpace>(nEdge, offsetPtr, matPtr, mapPtr,
                                              inPtr, outPtr, streamID);
            }
            else if (m_dimension == 2)
            {
                auto signPtr =
                    m_sign[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
                auto ncoeffPtr =
                    m_nEdgeCoeff[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID);
                auto matOffsetPtr =
                    m_matOffset[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID);
                auto mapOffsetPtr =
                    m_mapOffset[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID);

                RobBndCond2DKernel<ExecSpace>(m_nmaxcoeff, nEdge, ncoeffPtr,
                                              offsetPtr, matOffsetPtr,
                                              mapOffsetPtr, matPtr, mapPtr,
                                              signPtr, inPtr, outPtr, streamID);
            }

            // Reshape back, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                inWidth, 1u, inBlk.GetNumElementsWithPadding(),
                inBlk.GetNumData(), inPtr, streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                inWidth, 1u, outBlk.GetNumElementsWithPadding(),
                outBlk.GetNumData(), outPtr, streamID);
            outBlk.template SetInterleaveWidth<TData>(inWidth);
        }
    }
};

} // namespace Nektar::MultiRegions::detail
