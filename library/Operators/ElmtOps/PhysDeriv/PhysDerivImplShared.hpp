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

#include "Operators/Common/OperatorHelper.hpp"
#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivCUDASumFacKernels.cuh"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)>::type>
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
        const TData *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr      = out.template GetPtr<MemSpace, ReadWrite>();

        const TData *dfPtr = m_derivFac.template GetPtr<MemSpace, ReadOnly>();

        size_t nSize = out.GetFieldSize();

        // Initialize index.
        size_t exp_idx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        for (auto const &block : in.GetBlocks())
        {
            // Block dependent
            auto const nElmts    = block.num_elements;
            auto const nPadElmts = block.num_padding_elements;

            // Determine shape and type of the element.
            auto const expPtr    = this->m_expansionList->GetExp(exp_idx);
            auto const shape     = expPtr->DetShapeType();
            auto const dimension = expPtr->GetShapeDimension();
            auto const deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            auto const nqTot  = expPtr->GetTotPoints();
            auto const nCoord = expPtr->GetCoordim();

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < expPtr->GetShapeDimension(); d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Function call to kernel functions.
            if (dimension == 1)
            {
                auto D0 = m_derivativeMap[basisKeys[0]]
                              .template GetPtr<MemSpace, ReadOnly>();

                auto nq0 = expPtr->GetNumPoints(0);

                if (deformed)
                {
                    PhysDeriv1DKernel<ExecSpace, TData, true>(
                        nq0, nCoord, nElmts + nPadElmts, nSize, m_dfSize, D0,
                        dfPtr, inPtr, outPtr);
                }
                else
                {
                    PhysDeriv1DKernel<ExecSpace, TData, false>(
                        nq0, nCoord, nElmts + nPadElmts, nSize, m_dfSize, D0,
                        dfPtr, inPtr, outPtr);
                }
            }
            else if (dimension == 2)
            {
                auto D0 = m_derivativeMap[basisKeys[0]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto D1 = m_derivativeMap[basisKeys[1]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto Z0 = m_pointMap[basisKeys[0]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto Z1 = m_pointMap[basisKeys[1]]
                              .template GetPtr<MemSpace, ReadOnly>();

                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);

                if (deformed)
                {
                    PhysDeriv2DKernel<ExecSpace, TData, true>(
                        shape, nq0, nq1, nCoord, nElmts + nPadElmts, nSize,
                        m_dfSize, D0, D1, Z0, Z1, dfPtr, inPtr, outPtr);
                }
                else
                {
                    PhysDeriv2DKernel<ExecSpace, TData, false>(
                        shape, nq0, nq1, nCoord, nElmts + nPadElmts, nSize,
                        m_dfSize, D0, D1, Z0, Z1, dfPtr, inPtr, outPtr);
                }
            }
            else if (dimension == 3)
            {
                auto D0 = m_derivativeMap[basisKeys[0]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto D1 = m_derivativeMap[basisKeys[1]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto D2 = m_derivativeMap[basisKeys[2]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto Z0 = m_pointMap[basisKeys[0]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto Z1 = m_pointMap[basisKeys[1]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto Z2 = m_pointMap[basisKeys[2]]
                              .template GetPtr<MemSpace, ReadOnly>();

                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);
                auto nq2 = expPtr->GetNumPoints(2);

                if (deformed)
                {
                    PhysDeriv3DKernel<ExecSpace, TData, true>(
                        shape, nq0, nq1, nq2, nElmts + nPadElmts, nSize,
                        m_dfSize, D0, D1, D2, Z0, Z1, Z2, dfPtr, inPtr, outPtr);
                }
                else
                {
                    PhysDeriv3DKernel<ExecSpace, TData, false>(
                        shape, nq0, nq1, nq2, nElmts + nPadElmts, nSize,
                        m_dfSize, D0, D1, D2, Z0, Z1, Z2, dfPtr, inPtr, outPtr);
                }
            }

            // Increment pointer and index for next element type.
            dfPtr += deformed ? nqTot * nElmts : nElmts;
            inPtr += (nElmts + nPadElmts) * nqTot;
            outPtr += (nElmts + nPadElmts) * nqTot;
            exp_idx += nElmts;
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    BasisDataMap<TData> m_pointMap;
    BasisDataMap<TData> m_derivativeMap;
    MemoryRegion<TData> m_derivFac;
    size_t m_dfSize;
};

} // namespace Nektar::Operators::detail
