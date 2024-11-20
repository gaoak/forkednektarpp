///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransDeviceSumFac.hpp
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
#include "Operators/ElmtOps/OperatorBwdTrans.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/BwdTrans/BwdTransCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/BwdTrans/BwdTransKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransSYCLSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorBwdTransImpl : public OperatorBwdTrans<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
        // Initialize the basis data.
        m_basisMap =
            GetBasisData<MemSpace, NekDouble, TData>(expansionList, eBasis);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        const auto dimension = this->m_expansionList->GetShapeDimension();

        const bool device_only = true;

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (size_t blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            // Block dependent.
            auto &inblock        = in.GetBlocks()[blk];
            auto &outblock       = out.GetBlocks()[blk];
            const auto nElmtsPad = inblock.GetNumElementsWithPadding();

            // Initialize pointers.
            auto inPtr = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                             ? inblock.template GetPtr<MemSpace, ReadOnly>()
                             : inblock.template GetPtr<MemSpace, ReadWrite>();
            auto outPtr = outblock.template GetPtr<MemSpace, WriteOnly>();

            // Determine shape and type of the element.
            const auto expPtr    = this->m_expansionList->GetExp(exp_idx);
            const auto shapeType = expPtr->DetShapeType();
            const auto nm0       = expPtr->GetBasisNumModes(0);
            const auto nq0       = expPtr->GetNumPoints(0);
            const auto nm1 = (dimension > 1) ? expPtr->GetBasisNumModes(1) : 0;
            const auto nq1 = (dimension > 1) ? expPtr->GetNumPoints(1) : 0;
            const auto nm2 = (dimension > 2) ? expPtr->GetBasisNumModes(2) : 0;
            const auto nq2 = (dimension > 2) ? expPtr->GetNumPoints(2) : 0;
            const auto basis0 = m_basisMap[expPtr->GetBasis(0)->GetBasisKey()]
                                    .template GetPtr<MemSpace, ReadOnly>();
            const auto basis1 =
                (dimension > 1) ? m_basisMap[expPtr->GetBasis(1)->GetBasisKey()]
                                      .template GetPtr<MemSpace, ReadOnly>()
                                : nullptr;
            const auto basis2 =
                (dimension > 2) ? m_basisMap[expPtr->GetBasis(2)->GetBasisKey()]
                                      .template GetPtr<MemSpace, ReadOnly>()
                                : nullptr;

            // Flag for collapsed coordinate correction.
            bool correct = expPtr->GetBasis(0)->GetBasisType() ==
                           LibUtilities::eModified_A;

            // Set workspace.
            if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
            {
                if (m_wsp.size() <= blk)
                {
                    m_wsp.push_back(
                        SetWorkspace(shapeType, nElmtsPad, nm0, nm1, nm2));
                }
            }

            // Get workspace pointer.
            auto wspPtr =
                std::is_same_v<Implementation, Operators::SumFac>
                    ? m_wsp[blk].template GetPtr<MemSpace, WriteOnly>()
                    : nullptr;

            constexpr bool SharedMemory = true;

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inPtr);
            inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
            outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

            // Function call to kernel functions.
            if (dimension == 1)
            {
                BwdTrans1DKernel<ExecSpace, Implementation, SharedMemory>(
                    nm0, nq0, nElmtsPad, basis0, inPtr, outPtr);
            }
            else if (dimension == 2)
            {
                BwdTrans2DKernel<ExecSpace, Implementation, SharedMemory>(
                    shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct, basis0,
                    basis1, wspPtr, inPtr, outPtr);
            }
            else if (dimension == 3)
            {
#if !defined(NEKTAR_USE_QP_1D_KERNEL)
                const bool indexing = false;
#else
                const bool indexing =
                    shapeType == LibUtilities::Tet &&
                    std::is_same_v<Implementation, Operators::SumFacQP>;
#endif
                std::vector<LibUtilities::BasisKey> basisKeys{
                    expPtr->GetBasis(0)->GetBasisKey(),
                    expPtr->GetBasis(1)->GetBasisKey(),
                    expPtr->GetBasis(2)->GetBasisKey()};

                // Precompute index, if necessary.
                if (indexing)
                {
                    if (m_index0.find(basisKeys) == m_index0.end())
                    {
                        const unsigned int nm01 =
                            (2u * nm1 - nm0 + 1u) * nm0 / 2u;
                        std::vector<unsigned int> index0(nm01);
                        std::vector<unsigned int> index1(nm01);
                        for (unsigned int p = 0, mode_pq = 0; p < nm0; p++)
                        {
                            for (unsigned int q = 0; q < nm1 - p;
                                 q++, mode_pq++)
                            {
                                index0[mode_pq] = p;
                                index1[mode_pq] = q;
                            }
                        }
                        m_index0[basisKeys] =
                            MemoryRegion<unsigned int>::template FromVector<
                                MemSpace>(index0, ExecSpace::alignment,
                                          device_only);
                        m_index1[basisKeys] =
                            MemoryRegion<unsigned int>::template FromVector<
                                MemSpace>(index1, ExecSpace::alignment,
                                          device_only);
                    }
                }

                auto index0 = indexing
                                  ? m_index0[basisKeys]
                                        .template GetPtr<MemSpace, ReadOnly>()
                                  : nullptr;
                auto index1 = indexing
                                  ? m_index1[basisKeys]
                                        .template GetPtr<MemSpace, ReadOnly>()
                                  : nullptr;

                BwdTrans3DKernel<ExecSpace, Implementation, SharedMemory>(
                    shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, correct,
                    index0, index1, basis0, basis1, basis2, wspPtr, inPtr,
                    outPtr);
            }

            // Increment index for next element type.
            exp_idx += inblock.GetNumElements();
        }
    }

    size_t GetSharedWorkspaceSize(LibUtilities::ShapeType shapeType,
                                  size_t nElmtsPad, size_t nm0, size_t nm1,
                                  size_t nm2)
    {
        size_t wspsize = 0;

        if (shapeType == LibUtilities::Quad)
        {
            wspsize = nm1 * nElmtsPad;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = nm0 * nElmtsPad;
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = (nm1 * nm2 + nm2) * nElmtsPad;
        }
        else if (shapeType == LibUtilities::Tet)
        {
            wspsize = ((2 * nm1 - nm0 + 1) * nm0 / 2 + nm0) * nElmtsPad;
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = (nm0 * nm1 + nm0) * nElmtsPad;
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = (nm0 * nm1 + nm0) * nElmtsPad;
        }

        return wspsize;
    }

    MemoryRegion<TData> SetWorkspace(LibUtilities::ShapeType shapeType,
                                     size_t nElmtsPad, size_t nm0, size_t nm1,
                                     size_t nm2)
    {
        const bool device_only = true;

        size_t wspsize =
            GetSharedWorkspaceSize(shapeType, nElmtsPad, nm0, nm1, nm2);

        return MemoryRegion<TData>::template Create<MemSpace>(
            wspsize, ExecSpace::alignment, device_only);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorBwdTransImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    BasisDataMap<TData> m_basisMap;
    std::vector<MemoryRegion<TData>> m_wsp;
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index0;
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index1;
    static constexpr size_t m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::vector_width<TData>::value
            : 1u;
};

} // namespace Nektar::Operators::detail
