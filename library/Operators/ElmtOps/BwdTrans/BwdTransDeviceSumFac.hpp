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
#include "Operators/Field/MemoryRegion.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/BwdTrans/BwdTransCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/BwdTrans/BwdTransKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransSYCLSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFacQP>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::SYCL>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::SYCL>::value &&
               std::is_same<Implementation, Operators::SumFacQP>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value &&
               std::is_same<Implementation, Operators::SumFacQP>::value)>::type>
class OperatorBwdTransImpl : public OperatorBwdTrans<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
        // Initialize the basis data.
        m_basisMap = GetBasisData<MemSpace, TData>(expansionList, eBasis);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Copy memory to the device, if necessary and get raw pointers.
        const TData *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            auto &inblock  = in.GetBlocks()[block_idx];
            auto &outblock = out.GetBlocks()[block_idx];
            const auto nElmtsPad =
                inblock.num_padding_elements + inblock.num_elements;

            // Determine shape and type of the element.
            const auto expPtr    = this->m_expansionList->GetExp(exp_idx);
            const auto shapeType = expPtr->DetShapeType();
            const auto dimension = expPtr->GetShapeDimension();
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
            TData *wspPtr = SetWorkspace(shapeType, nElmtsPad, nm0, nm1, nm2);

            constexpr bool SharedMemory = true;

            // Reshape, if necessary.
            if constexpr (std::is_same<Implementation,
                                       Operators::SumFac>::value)
            {
                ReshapeStorage<ExecSpace,
                               NektarSpaces::vector_width<TData>::value>(
                    inblock.GetInterleaveWidth(), nElmtsPad, inblock.num_pts,
                    (TData *)inPtr);
                inblock.SetInterleaveWidth(
                    NektarSpaces::vector_width<TData>::value);
                outblock.SetInterleaveWidth(
                    NektarSpaces::vector_width<TData>::value);
            }

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
                BwdTrans3DKernel<ExecSpace, Implementation, SharedMemory>(
                    shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, correct,
                    basis0, basis1, basis2, wspPtr, inPtr, outPtr);
            }

            // Increment pointer and index for next element type.
            inPtr += inblock.block_size;
            outPtr += outblock.block_size;
            exp_idx += inblock.num_elements;
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

    TData *SetWorkspace(LibUtilities::ShapeType shapeType, size_t nElmtsPad,
                        size_t nm0, size_t nm1, size_t nm2)
    {
        TData *wspptr = nullptr;

        if constexpr (std::is_same<Implementation, Operators::SumFac>::value)
        {
            const bool device_only = true;

            size_t wspsize =
                GetSharedWorkspaceSize(shapeType, nElmtsPad, nm0, nm1, nm2);

            if (m_wspsize < wspsize)
            {
                m_wspsize = wspsize;

                m_wsp = MemoryRegion<TData>::template create<MemSpace>(
                    m_wspsize, ExecSpace::alignment, device_only);
            }

            if (m_wspsize > 0)
            {
                wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();
            }
        }

        return wspptr;
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
    MemoryRegion<TData> m_wsp;

    size_t m_wspsize = 0;
};

} // namespace Nektar::Operators::detail
