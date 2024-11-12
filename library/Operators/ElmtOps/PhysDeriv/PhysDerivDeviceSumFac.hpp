///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceSumFac.hpp
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
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivSYCLSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorPhysDerivImpl : public OperatorPhysDeriv<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
        // Initialise the derivative factor.
        auto transpose = std::is_same_v<Implementation, Operators::SumFacQP>;
        auto locblocks = GetBlockAttributes<TData>(
            FieldState::Phys, expansionList, m_implInterleaveWidth);
        m_df = SetDerivativeFactor<MemSpace, TData>(
            expansionList, locblocks, ExecSpace::alignment, transpose);

        // Initialize the points.
        m_zeroMap = GetBasisData<MemSpace, TData>(expansionList, eZeros);

        // Initialize the derivative matrix.
        m_derivativeMap =
            GetBasisData<MemSpace, TData>(expansionList, eDerivative);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialize index.
        size_t exp_idx = 0;

        for (size_t blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            // Initialize pointers.
            auto inPtr  = (in.GetBlocks()[blk].GetInterleaveWidth() ==
                          m_implInterleaveWidth)
                              ? in.template GetPtr<MemSpace, ReadOnly>(blk)
                              : in.template GetPtr<MemSpace, ReadWrite>(blk);
            auto outPtr = out.template GetPtr<MemSpace, WriteOnly>(blk);
            auto dfPtr  = m_df[blk].template GetPtr<MemSpace, ReadOnly>();

            // Block dependent.
            auto &inblock        = in.GetBlocks()[blk];
            auto &outblock       = out.GetBlocks()[blk];
            const auto nElmts    = inblock.GetNumElements();
            const auto nElmtsPad = inblock.GetNumElementsWithPadding();

            // Determine shape and type of the element.
            const auto expPtr   = this->m_expansionList->GetExp(exp_idx);
            const auto shape    = expPtr->DetShapeType();
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto nCoord = expPtr->GetCoordim();
            const auto nq0    = expPtr->GetNumPoints(0);
            const auto nq1    = (dimension > 1) ? expPtr->GetNumPoints(1) : 0;
            const auto nq2    = (dimension > 2) ? expPtr->GetNumPoints(2) : 0;

            const auto D0 = m_derivativeMap[expPtr->GetBasis(0)->GetBasisKey()]
                                .template GetPtr<MemSpace, ReadOnly>();
            const auto D1 =
                (dimension > 1)
                    ? m_derivativeMap[expPtr->GetBasis(1)->GetBasisKey()]
                          .template GetPtr<MemSpace, ReadOnly>()
                    : nullptr;
            const auto D2 =
                (dimension > 2)
                    ? m_derivativeMap[expPtr->GetBasis(2)->GetBasisKey()]
                          .template GetPtr<MemSpace, ReadOnly>()
                    : nullptr;
            const auto Z0 = m_zeroMap[expPtr->GetBasis(0)->GetBasisKey()]
                                .template GetPtr<MemSpace, ReadOnly>();
            const auto Z1 = (dimension > 1)
                                ? m_zeroMap[expPtr->GetBasis(1)->GetBasisKey()]
                                      .template GetPtr<MemSpace, ReadOnly>()
                                : nullptr;
            const auto Z2 = (dimension > 2)
                                ? m_zeroMap[expPtr->GetBasis(2)->GetBasisKey()]
                                      .template GetPtr<MemSpace, ReadOnly>()
                                : nullptr;

            constexpr bool SharedMemory = true;

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inPtr);
            inblock.SetInterleaveWidth(m_implInterleaveWidth);
            outblock.SetInterleaveWidth(m_implInterleaveWidth);

            // Function call to kernel functions.
            if (dimension == 1)
            {
                if (deformed)
                {
                    constexpr bool Deformed = true;

                    PhysDeriv1DKernel<ExecSpace, Implementation, Deformed>(
                        nq0, nCoord, nElmtsPad, D0, dfPtr, inPtr, outPtr);
                }
                else
                {
                    constexpr bool Deformed = false;

                    PhysDeriv1DKernel<ExecSpace, Implementation, Deformed>(
                        nq0, nCoord, nElmtsPad, D0, dfPtr, inPtr, outPtr);
                }
            }
            else if (dimension == 2)
            {
                if (deformed)
                {
                    constexpr bool Deformed = true;

                    PhysDeriv2DKernel<ExecSpace, Implementation, Deformed,
                                      SharedMemory>(shape, nq0, nq1, nCoord,
                                                    nElmtsPad, D0, D1, Z0, Z1,
                                                    dfPtr, inPtr, outPtr);
                }
                else
                {
                    constexpr bool Deformed = false;

                    PhysDeriv2DKernel<ExecSpace, Implementation, Deformed,
                                      SharedMemory>(shape, nq0, nq1, nCoord,
                                                    nElmtsPad, D0, D1, Z0, Z1,
                                                    dfPtr, inPtr, outPtr);
                }
            }
            else if (dimension == 3)
            {
                if (deformed)
                {
                    constexpr bool Deformed = true;

                    PhysDeriv3DKernel<ExecSpace, Implementation, Deformed,
                                      SharedMemory>(
                        shape, nq0, nq1, nq2, nElmtsPad, D0, D1, D2, Z0, Z1, Z2,
                        dfPtr, inPtr, outPtr);
                }
                else
                {
                    constexpr bool Deformed = false;

                    PhysDeriv3DKernel<ExecSpace, Implementation, Deformed,
                                      SharedMemory>(
                        shape, nq0, nq1, nq2, nElmtsPad, D0, D1, D2, Z0, Z1, Z2,
                        dfPtr, inPtr, outPtr);
                }
            }

            // Increment index for next element type.
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
    BasisDataMap<TData> m_zeroMap;
    BasisDataMap<TData> m_derivativeMap;
    std::vector<MemoryRegion<TData>> m_df;
    static constexpr size_t m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::vector_width<TData>::value
            : 1u;
};

} // namespace Nektar::Operators::detail
