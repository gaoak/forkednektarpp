///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseImplShared.hpp
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
#include "Operators/ElmtOps/OperatorIProductWRTBase.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSYCLSumFacKernels.hpp"

#define FLAG_QP false // TODO: to be removed

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
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise the jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        auto jac       = Operator<TData>::SetJacobian(jacSize);

        m_jac = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            jac, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());

        // Initialize the basis data.
        m_basisMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_BASIS_DATA);
        m_weightMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_WEIGHT_DATA);
    }

    ~OperatorIProductWRTBaseImpl(void)
    {
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               const TData lambda = 1.0) override
    {
        ASSERTL1(in.GetVecWidth() == out.GetVecWidth(),
                 "Input and output widths are different but kernel is not "
                 "setup for this (yet)");

        // Copy memory to the device, if necessary and get raw pointers.
        const TData *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr      = (lambda == 1.0)
                                 ? out.template GetPtr<MemSpace, WriteOnly>()
                                 : out.template GetPtr<MemSpace, ReadWrite>();

        const TData *jacPtr = m_jac.template GetPtr<MemSpace, ReadOnly>();

        // Initialize index.
        size_t exp_idx = 0;
        size_t width   = in.GetVecWidth();

        // Loop over the blocks.
        for (const auto &block : in.GetBlocks())
        {
            // Block dependent
            const auto nElmts    = block.num_elements;
            const auto nElmtsPad = block.num_elmt_groups * width;

            // Determine shape and type of the element.
            const auto expPtr    = this->m_expansionList->GetExp(exp_idx);
            const auto shapeType = expPtr->DetShapeType();
            const auto dimension = expPtr->GetShapeDimension();
            const auto deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto nqTot = expPtr->GetTotPoints();
            const auto nmTot = expPtr->GetNcoeffs();
            const auto nm0   = expPtr->GetBasisNumModes(0);
            const auto nq0   = expPtr->GetNumPoints(0);
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
            const auto w0 = m_weightMap[expPtr->GetBasis(0)->GetBasisKey()]
                                .template GetPtr<MemSpace, ReadOnly>();
            const auto w1 =
                (dimension > 1)
                    ? m_weightMap[expPtr->GetBasis(1)->GetBasisKey()]
                          .template GetPtr<MemSpace, ReadOnly>()
                    : nullptr;
            const auto w2 =
                (dimension > 2)
                    ? m_weightMap[expPtr->GetBasis(2)->GetBasisKey()]
                          .template GetPtr<MemSpace, ReadOnly>()
                    : nullptr;

            // Flag for collapsed coordinate correction.
            bool correct = expPtr->GetBasis(0)->GetBasisType() ==
                           LibUtilities::eModified_A;

            // Set workspace.
            TData *wspPtr =
                SetWorkspace(shapeType, nElmtsPad, nq0, nq1, nq2, nm1, nm2);

            // Function call to kernel functions.
            if (dimension == 1)
            {
                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, false, false,
                                                true>(nm0, nq0, nElmtsPad,
                                                      basis0, w0, jacPtr, inPtr,
                                                      outPtr);
                    }
                    else
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, true, false,
                                                true>(nm0, nq0, nElmtsPad,
                                                      basis0, w0, jacPtr, inPtr,
                                                      outPtr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, false, false,
                                                false>(nm0, nq0, nElmtsPad,
                                                       basis0, w0, jacPtr,
                                                       inPtr, outPtr);
                    }
                    else
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, true, false,
                                                false>(nm0, nq0, nElmtsPad,
                                                       basis0, w0, jacPtr,
                                                       inPtr, outPtr, lambda);
                    }
                }
            }
            else if (dimension == 2)
            {
                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, false, false,
                                                true>(
                            shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                            basis0, basis1, w0, w1, jacPtr, wspPtr, inPtr,
                            outPtr);
                    }
                    else
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, true, false,
                                                true>(
                            shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                            basis0, basis1, w0, w1, jacPtr, wspPtr, inPtr,
                            outPtr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, false, false,
                                                false>(
                            shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                            basis0, basis1, w0, w1, jacPtr, wspPtr, inPtr,
                            outPtr);
                    }
                    else
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, true, false,
                                                false>(
                            shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                            basis0, basis1, w0, w1, jacPtr, wspPtr, inPtr,
                            outPtr, lambda);
                    }
                }
            }
            else if (dimension == 3)
            {
                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, false, false,
                                                true>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                            correct, basis0, basis1, basis2, w0, w1, w2, jacPtr,
                            wspPtr, inPtr, outPtr);
                    }
                    else
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, true, false,
                                                true>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                            correct, basis0, basis1, basis2, w0, w1, w2, jacPtr,
                            wspPtr, inPtr, outPtr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, false, false,
                                                false>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                            correct, basis0, basis1, basis2, w0, w1, w2, jacPtr,
                            wspPtr, inPtr, outPtr);
                    }
                    else
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, true, false,
                                                false>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                            correct, basis0, basis1, basis2, w0, w1, w2, jacPtr,
                            wspPtr, inPtr, outPtr, lambda);
                    }
                }
            }

            // Increment pointer and index for next element type.
            jacPtr += deformed ? nqTot * nElmts : nElmts;
            inPtr += nqTot * nElmtsPad;
            outPtr += nmTot * nElmtsPad;
            exp_idx += nElmts;
        }
    }

    size_t GetSharedWorkspaceSize(LibUtilities::ShapeType shapeType,
                                  size_t nElmts, size_t nq0, size_t nq1,
                                  size_t nq2, size_t nm1, size_t nm2)
    {
        size_t wspsize = 0;

        if (shapeType == LibUtilities::Quad)
        {
            wspsize = nq1 * nElmts;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = nq0 * nElmts;
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = (nq2 * nq1 + nq2) * nElmts;
        }
        else if (shapeType == LibUtilities::Tet)
        {
            wspsize = (nq2 * nq1 + nq2 + nm2) * nElmts;
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = (nq2 * nq1 + nq2 + nm1) * nElmts;
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = (nq2 * nq1 + nq2) * nElmts;
        }

        return wspsize;
    }

    TData *SetWorkspace(LibUtilities::ShapeType shapeType, size_t nElmts,
                        size_t nq0, size_t nq1, size_t nq2, size_t nm1,
                        size_t nm2)
    {
        TData *wspptr = nullptr;

        if constexpr (!FLAG_QP)
        {
            size_t wspsize = GetSharedWorkspaceSize(shapeType, nElmts, nq0, nq1,
                                                    nq2, nm1, nm2);

            if (m_wspsize < wspsize)
            {
                m_wspsize = wspsize;

                m_wsp = MemoryRegion<TData>::template create<MemSpace>(
                    m_wspsize,
                    EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
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
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    BasisDataMap<TData> m_basisMap;
    BasisDataMap<TData> m_weightMap;
    MemoryRegion<TData> m_jac;
    MemoryRegion<TData> m_wsp;
    size_t m_wspsize = 0;
};

} // namespace Nektar::Operators::detail
