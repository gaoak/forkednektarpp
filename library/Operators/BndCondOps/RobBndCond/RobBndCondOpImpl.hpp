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

#include "Operators/BndCondOps/RobBndCond/RobBndCondDeviceKernels.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondSerialAVXKernels.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class RobBndCondOpImpl : public RobBndCondOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    RobBndCondOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : RobBndCondOp<TData>(expansionList)
    {
        auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();

        // Set mapping to skip over padding elements
        size_t i = 0, j = 0;

        std::vector<size_t> alignmentMap(expansionList->GetNcoeffs());
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, expansionList);
        for (auto &block : blocks)
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

            m_mat = MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                mat, ExecSpace::alignment);
            m_map = MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                map, ExecSpace::alignment);
            m_offset =
                MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                    offset, ExecSpace::alignment);
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

            m_mat = MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                mat, ExecSpace::alignment);
            m_map = MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                map, ExecSpace::alignment);
            m_sign = MemoryRegion<int>::template FromVector<MemSpace, int>(
                sign, ExecSpace::alignment);
            m_nEdgeCoeff = MemoryRegion<unsigned int>::template FromVector<
                MemSpace, unsigned int>(nEdgeCoeff, ExecSpace::alignment);
            m_offset =
                MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                    offset, ExecSpace::alignment);
            m_matOffset =
                MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                    matOffset, ExecSpace::alignment);
            m_mapOffset =
                MemoryRegion<size_t>::template FromVector<MemSpace, size_t>(
                    mapOffset, ExecSpace::alignment);
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
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<RobBndCondOpImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    MemoryRegion<TData> m_mat;
    MemoryRegion<size_t> m_map;
    MemoryRegion<int> m_sign;
    MemoryRegion<unsigned int> m_nEdgeCoeff;
    MemoryRegion<size_t> m_offset;
    MemoryRegion<size_t> m_matOffset;
    MemoryRegion<size_t> m_mapOffset;

    unsigned int m_nmaxcoeff = 0;
    size_t m_nBndEdge        = 0;

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out,
                 const bool &negflag) override
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
            auto &inBlk  = in.GetBlocks()[blk];
            auto &outBlk = out.GetBlocks()[blk];
            auto inPtr   = inBlk.template GetPtr<MemSpace, ReadWrite>();
            auto outPtr  = outBlk.template GetPtr<MemSpace, ReadWrite>();

            auto inWidth = inBlk.GetInterleaveWidth();
            if (inWidth != 1)
            {
                deInterleave<ExecSpace>(
                    inWidth, inBlk.GetNumElementsWithPadding() / inWidth,
                    inBlk.GetNumData(), inPtr);
                inBlk.template SetInterleaveWidth<TData>(1);
            }
            auto outWidth = outBlk.GetInterleaveWidth();
            if (outWidth != 1)
            {
                deInterleave<ExecSpace>(
                    outWidth, outBlk.GetNumElementsWithPadding() / outWidth,
                    outBlk.GetNumData(), outPtr);
                inBlk.template SetInterleaveWidth<TData>(1);
            }
        }

        // Get pointers.
        auto inPtr   = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto matPtr  = m_mat.template GetPtr<MemSpace, ReadOnly>();
        auto mapPtr  = m_map.template GetPtr<MemSpace, ReadOnly>();
        auto signPtr = m_sign.template GetPtr<MemSpace, ReadOnly>();
        auto ncoeffPtr    = m_nEdgeCoeff.template GetPtr<MemSpace, ReadOnly>();
        auto offsetPtr    = m_offset.template GetPtr<MemSpace, ReadOnly>();
        auto matOffsetPtr = m_matOffset.template GetPtr<MemSpace, ReadOnly>();
        auto mapOffsetPtr = m_mapOffset.template GetPtr<MemSpace, ReadOnly>();
        auto outPtr = out.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>();

        // Apply Robin boundary conditions.
        auto dimension = this->m_expansionList->GetExp(0)->GetShapeDimension();
        if (dimension == 1)
        {
            if (negflag)
            {
                RobBndCond1DKernel<ExecSpace, true>(
                    m_nBndEdge, offsetPtr, matPtr, mapPtr, inPtr, outPtr);
            }
            else
            {
                RobBndCond1DKernel<ExecSpace, false>(
                    m_nBndEdge, offsetPtr, matPtr, mapPtr, inPtr, outPtr);
            }
        }
        else if (dimension == 2)
        {
            if (negflag)
            {
                RobBndCond2DKernel<ExecSpace, true>(
                    m_nmaxcoeff, m_nBndEdge, ncoeffPtr, offsetPtr, matOffsetPtr,
                    mapOffsetPtr, matPtr, mapPtr, signPtr, inPtr, outPtr);
            }
            else
            {
                RobBndCond2DKernel<ExecSpace, false>(
                    m_nmaxcoeff, m_nBndEdge, ncoeffPtr, offsetPtr, matOffsetPtr,
                    mapOffsetPtr, matPtr, mapPtr, signPtr, inPtr, outPtr);
            }
        }
    }
};

} // namespace Nektar::Operators::detail
