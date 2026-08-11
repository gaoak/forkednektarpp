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

#include "Operators/BndCondOps/RobBndCond/RobBndCondOp.hpp"

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondDeviceKernels.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondSerialAVXKernels.hpp"

namespace Nektar::Operators::detail
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

        // Set mapping to skip over padding elements
        size_t i = 0, j = 0;

        std::vector<size_t> alignmentMap(expansionList->GetNcoeffs());
        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                expansionList);
        for (auto &block : blockAttr)
        {
            const auto ncoeff    = block.GetNumData();
            const auto nelmt     = block.GetNumElements();
            const auto nPadElmts = block.GetNumPaddingElements();
            for (size_t e = 0; e < nelmt; e++)
            {
                for (unsigned int n = 0; n < ncoeff; n++)
                {
                    alignmentMap[i++] = j++;
                }
            }
            j += nPadElmts * ncoeff;
        }

        if (expansionList->GetExp(0)->GetShapeDimension() == 1)
        {
            // Determine size
            for (auto &r : robinBCInfo)
            {
                auto n      = r.first;
                auto expPtr = this->m_expansionList->GetExp(n);

                ASSERTL1(expPtr->IsBoundaryInteriorExpansion(),
                         "Not set up for non boundary-interior expansions");

                for (auto rBC = r.second; rBC; rBC = rBC->next)
                {
                    m_nBndEdge++;
                }
            }

            // Return if no Robin boundary condition.
            if (m_nBndEdge == 0)
            {
                return;
            }

            // Initialize data
            std::vector<TData> mat(m_nBndEdge);
            std::vector<size_t> map(m_nBndEdge);
            std::vector<size_t> offset(m_nBndEdge, 0u);
            size_t i = 0;
            for (auto &r : robinBCInfo)
            {
                auto n      = r.first;
                auto expPtr = this->m_expansionList->GetExp(n);

                for (auto rBC = r.second; rBC; rBC = rBC->next)
                {
                    auto primCoeffs = rBC->m_robinPrimitiveCoeffs;
                    auto vertid     = rBC->m_robinID;
                    mat[i]          = primCoeffs[0];
                    map[i]          = expPtr->GetVertexMap(vertid);
                    offset[i] =
                        alignmentMap[this->m_expansionList->GetCoeff_Offset(n)];
                    i++;
                }
            }

            m_mat = LibUtilities::MemoryRegion<TData>::template FromVector<
                MemSpace, TData>(mat);
            m_map = LibUtilities::MemoryRegion<size_t>::template FromVector<
                MemSpace, size_t>(map);
            m_offset = LibUtilities::MemoryRegion<size_t>::template FromVector<
                MemSpace, size_t>(offset);
        }
        else if (expansionList->GetExp(0)->GetShapeDimension() == 2)
        {
            // Determine size
            size_t matSize = 0;
            size_t mapSize = 0;
            for (auto &r : robinBCInfo)
            {
                auto n      = r.first;
                auto expPtr = this->m_expansionList->GetExp(n);

                ASSERTL1(expPtr->IsBoundaryInteriorExpansion(),
                         "Not set up for non boundary-interior expansions");

                for (auto rBC = r.second; rBC; rBC = rBC->next)
                {
                    auto edgeid  = rBC->m_robinID;
                    auto edgeExp = expPtr->GetTraceExp(edgeid);
                    auto ncoeff  = edgeExp->GetNcoeffs();
                    m_nmaxcoeff  = std::max(m_nmaxcoeff, (unsigned int)ncoeff);
                    matSize += ncoeff * ncoeff;
                    mapSize += ncoeff;
                    m_nBndEdge++;
                }
            }

            // Return if no Robin boundary condition.
            if (m_nBndEdge == 0)
            {
                return;
            }

            // Initialize data
            std::vector<TData> mat(matSize);
            std::vector<size_t> map(mapSize);
            std::vector<int> sign(mapSize);
            std::vector<unsigned int> nEdgeCoeff(m_nBndEdge, 0u);
            std::vector<size_t> offset(m_nBndEdge, 0u);
            std::vector<size_t> matOffset(m_nBndEdge, 0u);
            std::vector<size_t> mapOffset(m_nBndEdge, 0u);
            size_t i = 0;
            for (auto &r : robinBCInfo)
            {
                auto n      = r.first;
                auto expPtr = this->m_expansionList->GetExp(n);

                for (auto rBC = r.second; rBC; rBC = rBC->next)
                {
                    auto primCoeffs = rBC->m_robinPrimitiveCoeffs;
                    auto edgeid     = rBC->m_robinID;
                    auto orient     = expPtr->GetTraceOrient(edgeid);
                    auto edgeExp    = expPtr->GetTraceExp(edgeid);
                    auto ncoeff     = edgeExp->GetNcoeffs();

                    // Initialize map and sign array
                    Array<OneD, unsigned int> tmpMap(ncoeff); // size_t
                    Array<OneD, int> tmpSign(ncoeff);
                    expPtr->GetTraceToElementMap(edgeid, tmpMap, tmpSign,
                                                 orient);
                    std::copy(tmpMap.begin(), tmpMap.end(),
                              map.data() + mapOffset[i]);
                    std::copy(tmpSign.begin(), tmpSign.end(),
                              sign.data() + mapOffset[i]);

                    // Initialize mat array
                    StdRegions::VarCoeffMap varcoeffs;
                    varcoeffs[StdRegions::eVarCoeffMass] = primCoeffs;
                    LocalRegions::MatrixKey mkey(
                        StdRegions::eMass, LibUtilities::eSegment, *edgeExp,
                        StdRegions::NullConstFactorMap, varcoeffs);
                    DNekScalMat &edgeMat = *edgeExp->GetLocMatrix(mkey);
                    std::copy(edgeMat.GetRawPtr(),
                              edgeMat.GetRawPtr() + ncoeff * ncoeff,
                              mat.data() + matOffset[i]);

                    // Update offset array
                    nEdgeCoeff[i] = ncoeff;
                    offset[i] =
                        alignmentMap[this->m_expansionList->GetCoeff_Offset(n)];
                    if (i < m_nBndEdge - 1)
                    {
                        matOffset[i + 1] = matOffset[i] + ncoeff * ncoeff;
                        mapOffset[i + 1] = mapOffset[i] + ncoeff;
                    }

                    i++;
                }
            }

            m_mat = LibUtilities::MemoryRegion<TData>::template FromVector<
                MemSpace, TData>(mat);
            m_map = LibUtilities::MemoryRegion<size_t>::template FromVector<
                MemSpace, size_t>(map);
            m_sign =
                LibUtilities::MemoryRegion<int>::template FromVector<MemSpace,
                                                                     int>(sign);
            m_nEdgeCoeff =
                LibUtilities::MemoryRegion<unsigned int>::template FromVector<
                    MemSpace, unsigned int>(nEdgeCoeff);
            m_offset = LibUtilities::MemoryRegion<size_t>::template FromVector<
                MemSpace, size_t>(offset);
            m_matOffset = LibUtilities::MemoryRegion<
                size_t>::template FromVector<MemSpace, size_t>(matOffset);
            m_mapOffset = LibUtilities::MemoryRegion<
                size_t>::template FromVector<MemSpace, size_t>(mapOffset);
        }
        else if (expansionList->GetExp(0)->GetShapeDimension() == 3)
        {
            auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();

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
    LibUtilities::MemoryRegion<TData> m_mat;
    LibUtilities::MemoryRegion<size_t> m_map;
    LibUtilities::MemoryRegion<int> m_sign;
    LibUtilities::MemoryRegion<unsigned int> m_nEdgeCoeff;
    LibUtilities::MemoryRegion<size_t> m_offset;
    LibUtilities::MemoryRegion<size_t> m_matOffset;
    LibUtilities::MemoryRegion<size_t> m_mapOffset;

    unsigned int m_nmaxcoeff = 0;
    size_t m_nBndEdge        = 0;

    void v_Apply(MultiRegions::Field<TData, FieldState::Coeff> &in,
                 MultiRegions::Field<TData, FieldState::Coeff> &out) override
    {
        // Return if no Robin boundary condition.
        if (m_nBndEdge == 0)
        {
            return;
        }

        ASSERTL0(in.GetNumComponents() == 1,
                 "Not yet set up for multiple components");

        // if block is interlaced deInterleave block since currently mapping
        // set up assuming serial alignment
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &inBlk  = in.GetBlocks()[blk];
            auto &outBlk = out.GetBlocks()[blk];
            auto inPtr   = inBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto outPtr = outBlk.template GetPtr<MemSpace, ReadWrite>(streamID);

            auto inWidth = inBlk.GetInterleaveWidth();
            ReshapeStorage<ExecSpace>(1u, inWidth,
                                      inBlk.GetNumElementsWithPadding(),
                                      inBlk.GetNumData(), inPtr, streamID);
            auto outWidth = outBlk.GetInterleaveWidth();
            ReshapeStorage<ExecSpace>(1u, outWidth,
                                      outBlk.GetNumElementsWithPadding(),
                                      outBlk.GetNumData(), outPtr, streamID);
        }

        // Get pointers.
        auto matPtr       = m_mat.template GetPtr<MemSpace, ReadOnly>();
        auto mapPtr       = m_map.template GetPtr<MemSpace, ReadOnly>();
        auto signPtr      = m_sign.template GetPtr<MemSpace, ReadOnly>();
        auto ncoeffPtr    = m_nEdgeCoeff.template GetPtr<MemSpace, ReadOnly>();
        auto offsetPtr    = m_offset.template GetPtr<MemSpace, ReadOnly>();
        auto matOffsetPtr = m_matOffset.template GetPtr<MemSpace, ReadOnly>();
        auto mapOffsetPtr = m_mapOffset.template GetPtr<MemSpace, ReadOnly>();

        // Synchronize memory for all blocks.
        const unsigned int streamID0 = 1;

        auto inPtr =
            in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(streamID0);
        auto outPtr =
            out.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>(streamID0);
        for (unsigned int blk = 1; blk < in.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            in.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
            out.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
        }

        // Apply Robin boundary conditions.
        auto dimension = this->m_expansionList->GetExp(0)->GetShapeDimension();
        if (dimension == 1)
        {
            RobBndCond1DKernel<ExecSpace>(m_nBndEdge, offsetPtr, matPtr, mapPtr,
                                          inPtr, outPtr);
        }
        else if (dimension == 2)
        {
            RobBndCond2DKernel<ExecSpace>(
                m_nmaxcoeff, m_nBndEdge, ncoeffPtr, offsetPtr, matOffsetPtr,
                mapOffsetPtr, matPtr, mapPtr, signPtr, inPtr, outPtr);
        }

        // Reshape back, if necessary.
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            auto &inBlk  = in.GetBlocks()[blk];
            auto &outBlk = out.GetBlocks()[blk];
            auto inPtr   = inBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto outPtr = outBlk.template GetPtr<MemSpace, ReadWrite>(streamID);

            auto inWidth = inBlk.GetInterleaveWidth();
            ReshapeStorage<ExecSpace>(inWidth, 1u,
                                      inBlk.GetNumElementsWithPadding(),
                                      inBlk.GetNumData(), inPtr, streamID);
            ReshapeStorage<ExecSpace>(inWidth, 1u,
                                      outBlk.GetNumElementsWithPadding(),
                                      outBlk.GetNumData(), outPtr, streamID);
            outBlk.template SetInterleaveWidth<TData>(inWidth);
        }
    }
};

} // namespace Nektar::Operators::detail
