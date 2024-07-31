///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransImplShared.hpp
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

#include "Operators/ElmtOps/OperatorBwdTrans.hpp"

#include "Operators/Common/OperatorHelper.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/BwdTrans/BwdTransSYCLSumFacKernels.hpp"
#include "Operators/Field/MemoryRegion.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::SYCL>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value &&
               std::is_same<Implementation, Operators::SumFac>::value)>::type>
class OperatorBwdTransImpl : public OperatorBwdTrans<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
        // Initialize the basis data.
        m_basisMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_BASIS_DATA);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Copy memory to the device, if necessary and get raw pointers.
        const TData *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();

        TData *wspPtr = nullptr;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            auto const &inblock  = in.GetBlocks()[block_idx];
            auto const &outblock = out.GetBlocks()[block_idx];
            auto const nElmts =
                inblock.num_elements + inblock.num_padding_elements;

            // Determine shape and type of the element.
            auto const expPtr    = this->m_expansionList->GetExp(exp_idx);
            auto const shapeType = expPtr->DetShapeType();
            auto const dimension = expPtr->GetShapeDimension();
            auto const nmTot     = expPtr->GetNcoeffs();
            auto const nqTot     = expPtr->GetTotPoints();

            // Flag for collapsed coordinate correction.
            bool correct = expPtr->GetBasis(0)->GetBasisType() ==
                           LibUtilities::eModified_A;

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < dimension; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Function call to kernel functions.
            if (dimension == 1)
            {
                auto basis0 = m_basisMap[basisKeys[0]]
                                  .template GetPtr<MemSpace, ReadOnly>();

                auto nm0 = expPtr->GetBasisNumModes(0);
                auto nq0 = expPtr->GetNumPoints(0);
                BwdTrans1DKernel<ExecSpace, TData>(nm0, nq0, nElmts, basis0,
                                                   inPtr, outPtr);
            }
            else if (dimension == 2)
            {
                auto basis0 = m_basisMap[basisKeys[0]]
                                  .template GetPtr<MemSpace, ReadOnly>();
                auto basis1 = m_basisMap[basisKeys[1]]
                                  .template GetPtr<MemSpace, ReadOnly>();

                auto nm0 = expPtr->GetBasisNumModes(0);
                auto nm1 = expPtr->GetBasisNumModes(1);
                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);

                if constexpr (!FLAG_QP)
                {
                    size_t wspsize = 0;

                    if (shapeType == LibUtilities::Quad)
                    {
                        wspsize = nm1 * nElmts;
                    }
                    else if (shapeType == LibUtilities::Tri)
                    {
                        wspsize = nm0 * nElmts;
                    }

                    if (m_wspsize < wspsize)
                    {
                        m_wspsize = wspsize;
                        m_wsp = MemoryRegion<TData>::template create<MemSpace>(
                            m_wspsize,
                            EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
                    }

                    wspPtr = m_wsp.template GetPtr<MemSpace, WriteOnly>();
                }

                BwdTrans2DKernel<ExecSpace, TData>(
                    shapeType, nm0, nm1, nq0, nq1, nElmts, correct, basis0,
                    basis1, wspPtr, inPtr, outPtr);
            }
            else if (dimension == 3)
            {
                auto basis0 = m_basisMap[basisKeys[0]]
                                  .template GetPtr<MemSpace, ReadOnly>();
                auto basis1 = m_basisMap[basisKeys[1]]
                                  .template GetPtr<MemSpace, ReadOnly>();
                auto basis2 = m_basisMap[basisKeys[2]]
                                  .template GetPtr<MemSpace, ReadOnly>();

                auto nm0 = expPtr->GetBasisNumModes(0);
                auto nm1 = expPtr->GetBasisNumModes(1);
                auto nm2 = expPtr->GetBasisNumModes(2);
                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);
                auto nq2 = expPtr->GetNumPoints(2);

                if constexpr (!FLAG_QP)
                {
                    size_t wspsize = 0;

                    if (shapeType == LibUtilities::Hex)
                    {
                        wspsize = (nm1 * nm2 + nm2) * nElmts;
                    }
                    else if (shapeType == LibUtilities::Tet)
                    {
                        wspsize =
                            ((2 * nm1 - nm0 + 1) * nm0 / 2 + nm0) * nElmts;
                    }
                    else if (shapeType == LibUtilities::Prism)
                    {
                        wspsize = (nm0 * nm1 + nm0) * nElmts;
                    }
                    else if (shapeType == LibUtilities::Pyr)
                    {
                        wspsize = (nm0 * nm1 + nm0) * nElmts;
                    }

                    if (m_wspsize < wspsize)
                    {
                        m_wspsize = wspsize;

                        m_wsp = MemoryRegion<TData>::template create<MemSpace>(
                            m_wspsize,
                            EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
                    }

                    wspPtr = m_wsp.template GetPtr<MemSpace, WriteOnly>();
                }

                BwdTrans3DKernel<ExecSpace, TData>(
                    shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmts, correct,
                    basis0, basis1, basis2, wspPtr, inPtr, outPtr);
            }

            // Increment pointer and index for next element type.
            inPtr += inblock.block_size;
            outPtr += outblock.block_size;
            exp_idx += inblock.num_elements;
        }
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
