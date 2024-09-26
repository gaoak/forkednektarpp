///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondImpl.hpp
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

#include "Operators/BndCondOps/OperatorRobBndCond.hpp"

#include "Operators/BndCondOps/RobBndCond/RobBndCondCUDAKernels.cuh"
#include "Operators/BndCondOps/RobBndCond/RobBndCondKernels.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondKokkosKernels.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondSYCLKernels.hpp"

#include <LocalRegions/MatrixKey.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorRobBndCondImpl : public OperatorRobBndCond<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorRobBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorRobBndCond<TData>(expansionList)
    {
        auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();

        // Set mapping to skip over padding elements
        int i = 0, j = 0;

        Array<OneD, int> alignmentMap(expansionList->GetNcoeffs());
        auto blocks = GetBlockAttributes(FieldState::Coeff, expansionList,
                                         ExecSpace::width);
        for (auto &block : blocks)
        {
            const auto ncoeff    = block.num_pts;
            const auto nElmts    = block.num_elements;
            const auto nPadElmts = block.num_padding_elements;
            for (unsigned int e = 0; e < nElmts; e++)
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
            Array<OneD, TData> mat(m_nBndEdge);
            Array<OneD, unsigned int> map(m_nBndEdge);
            Array<OneD, unsigned int> offset(m_nBndEdge, 0u);
            unsigned int i = 0;
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

            m_mat = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                mat, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_map =
                MemoryRegion<unsigned int>::template fromArray<MemSpace,
                                                               unsigned int>(
                    map, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_offset =
                MemoryRegion<unsigned int>::template fromArray<MemSpace,
                                                               unsigned int>(
                    offset,
                    EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
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
                    m_nmaxcoeff  = std::max(m_nmaxcoeff, (size_t)ncoeff);
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
            Array<OneD, TData> mat(matSize);
            Array<OneD, unsigned int> map(mapSize);
            Array<OneD, int> sign(mapSize);
            Array<OneD, unsigned int> nEdgeCoeff(m_nBndEdge, 0u);
            Array<OneD, unsigned int> offset(m_nBndEdge, 0u);
            Array<OneD, unsigned int> matOffset(m_nBndEdge, 0u);
            Array<OneD, unsigned int> mapOffset(m_nBndEdge, 0u);
            unsigned int i = 0;
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
                    Array<OneD, unsigned int> tmpMap(ncoeff);
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

            m_mat = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                mat, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_map =
                MemoryRegion<unsigned int>::template fromArray<MemSpace,
                                                               unsigned int>(
                    map, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_sign = MemoryRegion<int>::template fromArray<MemSpace, int>(
                sign, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_nEdgeCoeff =
                MemoryRegion<unsigned int>::template fromArray<MemSpace,
                                                               unsigned int>(
                    nEdgeCoeff,
                    EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_offset =
                MemoryRegion<unsigned int>::template fromArray<MemSpace,
                                                               unsigned int>(
                    offset,
                    EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_matOffset =
                MemoryRegion<unsigned int>::template fromArray<MemSpace,
                                                               unsigned int>(
                    matOffset,
                    EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_mapOffset =
                MemoryRegion<unsigned int>::template fromArray<MemSpace,
                                                               unsigned int>(
                    mapOffset,
                    EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
        }
        else if (expansionList->GetExp(0)->GetShapeDimension() == 3)
        {
            auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();

            for ([[maybe_unused]] auto &r : robinBCInfo)
            {
                NEKERROR(
                    ErrorUtil::efatal,
                    "OperatorRobBndCondImpl: 3D Operator not yet implemented");
            }
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &negflag) override
    {
        // Return if no Robin boundary condition.
        if (m_nBndEdge == 0)
        {
            return;
        }

        // Get pointers.
        const auto *inPtr   = in.template GetPtr<MemSpace, ReadOnly>();
        const auto *matPtr  = m_mat.template GetPtr<MemSpace, ReadOnly>();
        const auto *mapPtr  = m_map.template GetPtr<MemSpace, ReadOnly>();
        const auto *signPtr = m_sign.template GetPtr<MemSpace, ReadOnly>();
        const auto *ncoeffPtr =
            m_nEdgeCoeff.template GetPtr<MemSpace, ReadOnly>();
        const auto *offsetPtr = m_offset.template GetPtr<MemSpace, ReadOnly>();
        const auto *matOffsetPtr =
            m_matOffset.template GetPtr<MemSpace, ReadOnly>();
        const auto *mapOffsetPtr =
            m_mapOffset.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr = out.template GetPtr<MemSpace, WriteOnly>();

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

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorRobBndCondImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

protected:
    MemoryRegion<TData> m_mat;
    MemoryRegion<unsigned int> m_map;
    MemoryRegion<int> m_sign;
    MemoryRegion<unsigned int> m_nEdgeCoeff;
    MemoryRegion<unsigned int> m_offset;
    MemoryRegion<unsigned int> m_matOffset;
    MemoryRegion<unsigned int> m_mapOffset;

    size_t m_nmaxcoeff = 0;
    size_t m_nBndEdge  = 0;
};

} // namespace Nektar::Operators::detail
