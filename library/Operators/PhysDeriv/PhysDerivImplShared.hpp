///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivImplShared.hpp
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

#include "Operators/OperatorPhysDeriv.hpp"

#include "Operators/OperatorHelper.hpp"
#include "Operators/PhysDeriv/PhysDerivCUDASumFacKernels.cuh"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
              std::is_same<Implementation, Operators::SumFac>::value>::type>
class OperatorPhysDerivImpl : public OperatorPhysDeriv<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
        size_t nDim      = this->m_expansionList->GetShapeDimension();
        size_t nCoord    = this->m_expansionList->GetCoordim(0);
        size_t nTotElmts = this->m_expansionList->GetNumElmts();

        // Initialise the derivative factor.
        m_dfSize = Operator<TData>::GetGeometricFactorSize();

        auto derivFac = Operator<TData>::SetDerivativeFactor(m_dfSize);

        m_derivFac = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            derivFac, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Initialize the points.
        m_pointMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_POINT_DATA);

        // Initialize the derivative matrix.
        m_derivativeMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_DERIVATIVE_DATA);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Initialize pointers.
        const TData *inptr = in.template GetConstPtr<MemSpace>();
        TData *outptr      = out.template GetPtr<MemSpace>();

        const TData *dfptr = m_derivFac.template GetConstPtr<MemSpace>();

        size_t nSize = out.GetFieldSize();

        // Initialize index.
        size_t expIdx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        for (auto const &block : in.GetBlocks())
        {
            // Determine shape and type of the element.
            auto nElmts    = block.num_elements;
            auto nPadElmts = block.num_padding_elements;

            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nqTot        = expPtr->GetTotPoints();
            auto nDim         = expPtr->GetShapeDimension();
            auto nCoord       = expPtr->GetCoordim();
            auto shape        = expPtr->DetShapeType();
            auto deformed     = expPtr->GetMetricInfo()->GetGtype() ==
                            SpatialDomains::eDeformed;

            // Determine CUDA grid size.
            m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < expPtr->GetShapeDimension(); d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Function call to kernel functions.
            if (nDim == 1)
            {
                auto D0 = m_derivativeMap[basisKeys[0]]
                              .template GetConstPtr<MemSpace>();

                auto nq0 = expPtr->GetNumPoints(0);

                if (deformed)
                {
                    PhysDeriv1DKernel<ExecSpace, TData, true>(
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, nSize,
                        m_dfSize, D0, dfptr, inptr, outptr);
                }
                else
                {
                    PhysDeriv1DKernel<ExecSpace, TData, false>(
                        m_gridSize, m_blockSize, nq0, nCoord, nElmts, nSize,
                        m_dfSize, D0, dfptr, inptr, outptr);
                }
            }
            else if (nDim == 2)
            {
                auto D0 = m_derivativeMap[basisKeys[0]]
                              .template GetConstPtr<MemSpace>();
                auto D1 = m_derivativeMap[basisKeys[1]]
                              .template GetConstPtr<MemSpace>();
                auto Z0 =
                    m_pointMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto Z1 =
                    m_pointMap[basisKeys[1]].template GetConstPtr<MemSpace>();

                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);

                if (deformed)
                {
                    PhysDeriv2DKernel<ExecSpace, TData, true>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nCoord,
                        nElmts, nSize, m_dfSize, D0, D1, Z0, Z1, dfptr, inptr,
                        outptr);
                }
                else
                {
                    PhysDeriv2DKernel<ExecSpace, TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nCoord,
                        nElmts, nSize, m_dfSize, D0, D1, Z0, Z1, dfptr, inptr,
                        outptr);
                }
            }
            else if (nDim == 3)
            {
                auto D0 = m_derivativeMap[basisKeys[0]]
                              .template GetConstPtr<MemSpace>();
                auto D1 = m_derivativeMap[basisKeys[1]]
                              .template GetConstPtr<MemSpace>();
                auto D2 = m_derivativeMap[basisKeys[2]]
                              .template GetConstPtr<MemSpace>();
                auto Z0 =
                    m_pointMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto Z1 =
                    m_pointMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto Z2 =
                    m_pointMap[basisKeys[2]].template GetConstPtr<MemSpace>();

                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);
                auto nq2 = expPtr->GetNumPoints(2);

                if (deformed)
                {
                    PhysDeriv3DKernel<ExecSpace, TData, true>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nElmts,
                        nSize, m_dfSize, D0, D1, D2, Z0, Z1, Z2, dfptr, inptr,
                        outptr);
                }
                else
                {
                    PhysDeriv3DKernel<ExecSpace, TData, false>(
                        m_gridSize, m_blockSize, shape, nq0, nq1, nq2, nElmts,
                        nSize, m_dfSize, D0, D1, D2, Z0, Z1, Z2, dfptr, inptr,
                        outptr);
                }
            }

            // Increment pointer and index for next element type.
            dfptr += deformed ? nqTot * nElmts : nElmts;
            outptr += (nPadElmts + nElmts) * nqTot;
            inptr += (nPadElmts + nElmts) * nqTot;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    static std::string className;

private:
    BasisDataMap<TData> m_pointMap;
    BasisDataMap<TData> m_derivativeMap;

    MemoryRegion<TData> m_derivFac;

    size_t m_dfSize;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
