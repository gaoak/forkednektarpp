///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseImplShared.hpp
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
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/OperatorIProductWRTDerivBase.hpp"

#define FLAG_QP false // TODO: to be removed

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value &&
               std::is_same<Implementation, Operators::SumFac>::value)>::type>
class OperatorIProductWRTDerivBaseImpl
    : public OperatorIProductWRTDerivBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
        size_t nDim   = this->m_expansionList->GetShapeDimension();
        size_t nCoord = this->m_expansionList->GetCoordim(0);

        // Initialise the jacobian and the derivative factor.
        m_dfSize = Operator<TData>::GetGeometricFactorSize();

        auto jac      = Operator<TData>::SetJacobian(m_dfSize);
        auto derivFac = Operator<TData>::SetDerivativeFactor(m_dfSize);

        // Initialise the jacobian.
        m_jac = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            jac, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Initialise the derivative factor.
        m_derivFac = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            derivFac, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Initialize the basis data.
        m_basisMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_BASIS_DATA);
        m_dbasisMap = GetBasisData<MemSpace, TData>(
            expansionList, BASIS_BASIS_DERIVATIVE_DATA);
        m_weightMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_WEIGHT_DATA);
        m_pointMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_POINT_DATA);
        m_derivativeMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_DERIVATIVE_DATA);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               bool APPEND = false) override
    {
        // Copy memory to the device, if necessary and get raw pointers.
        const TData *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr      = APPEND ? out.template GetPtr<MemSpace, ReadWrite>()
                                    : out.template GetPtr<MemSpace, WriteOnly>();

        const TData *jacPtr = m_jac.template GetPtr<MemSpace, ReadOnly>();
        const TData *dfPtr  = m_derivFac.template GetPtr<MemSpace, ReadOnly>();

        // Zero output.
        if (!APPEND)
        {
            out.initialize(0);
        }

        // Initialize the workspace memory.
        auto nSize = in.GetFieldSize();
        if (m_tmpsize < nSize)
        {
            m_tmpsize = nSize;
            m_tmp     = MemoryRegion<TData>::template create<MemSpace>(
                nSize * this->m_expansionList->GetExp(0)->GetCoordim());
        }

        TData *tmpPtr = m_tmp.template GetPtr<MemSpace, WriteOnly>();

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (const auto &block : in.GetBlocks())
        {
            // Block dependent
            const auto nElmts    = block.num_elements;
            const auto nPadElmts = block.num_padding_elements;

            // Determine shape and type of the element.
            const auto expPtr    = this->m_expansionList->GetExp(exp_idx);
            const auto shapeType = expPtr->DetShapeType();
            const auto dimension = expPtr->GetShapeDimension();
            const auto deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto nqTot  = expPtr->GetTotPoints();
            const auto nmTot  = expPtr->GetNcoeffs();
            const auto nCoord = expPtr->GetCoordim();
            const auto nm0    = expPtr->GetBasisNumModes(0);
            const auto nq0    = expPtr->GetNumPoints(0);
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
            const auto dbasis0 = m_dbasisMap[expPtr->GetBasis(0)->GetBasisKey()]
                                     .template GetPtr<MemSpace, ReadOnly>();
            const auto dbasis1 =
                (dimension > 1)
                    ? m_dbasisMap[expPtr->GetBasis(1)->GetBasisKey()]
                          .template GetPtr<MemSpace, ReadOnly>()
                    : nullptr;
            const auto dbasis2 =
                (dimension > 2)
                    ? m_dbasisMap[expPtr->GetBasis(2)->GetBasisKey()]
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
            const auto Z0 = m_pointMap[expPtr->GetBasis(0)->GetBasisKey()]
                                .template GetPtr<MemSpace, ReadOnly>();
            const auto Z1 = (dimension > 1)
                                ? m_pointMap[expPtr->GetBasis(1)->GetBasisKey()]
                                      .template GetPtr<MemSpace, ReadOnly>()
                                : nullptr;
            const auto Z2 = (dimension > 2)
                                ? m_pointMap[expPtr->GetBasis(2)->GetBasisKey()]
                                      .template GetPtr<MemSpace, ReadOnly>()
                                : nullptr;

            // Flag for collapsed coordinate correction.
            bool correct = expPtr->GetBasis(0)->GetBasisType() ==
                           LibUtilities::eModified_A;

            // Set workspace.
            TData *wspPtr = SetWorkspace(shapeType, nElmts + nPadElmts, nq0,
                                         nq1, nq2, nm1, nm2);

            // Function call to kernel functions.
            if (dimension == 1)
            {
                if (deformed)
                {
                    IProductWRTDerivBase1DKernel<ExecSpace, TData, true>(
                        nq0, nCoord, nElmts, nSize, m_dfSize, dfPtr, inPtr,
                        tmpPtr);
                    IProductWRTBase1DKernel<ExecSpace, TData, false, true,
                                            true>(nm0, nq0, nElmts + nPadElmts,
                                                  dbasis0, w0, jacPtr, tmpPtr,
                                                  outPtr);
                }
                else
                {
                    IProductWRTDerivBase1DKernel<ExecSpace, TData, false>(
                        nq0, nCoord, nElmts, nSize, m_dfSize, dfPtr, inPtr,
                        tmpPtr);
                    IProductWRTBase1DKernel<ExecSpace, TData, false, true,
                                            false>(nm0, nq0, nElmts + nPadElmts,
                                                   dbasis0, w0, jacPtr, tmpPtr,
                                                   outPtr);
                }
            }
            else if (dimension == 2)
            {

                if (deformed)
                {
                    IProductWRTDerivBase2DKernel<ExecSpace, TData, true>(
                        shapeType, nq0, nq1, nCoord, nElmts, nSize, m_dfSize,
                        Z0, Z1, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase2DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                        correct, dbasis0, basis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr, outPtr);
                    IProductWRTBase2DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                        correct, basis0, dbasis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr + nSize, outPtr);
                }
                else
                {
                    IProductWRTDerivBase2DKernel<ExecSpace, TData, false>(
                        shapeType, nq0, nq1, nCoord, nElmts, nSize, m_dfSize,
                        Z0, Z1, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase2DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                        correct, dbasis0, basis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr, outPtr);
                    IProductWRTBase2DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                        correct, basis0, dbasis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr + nSize, outPtr);
                }
            }
            else if (dimension == 3)
            {
                if (deformed)
                {
                    IProductWRTDerivBase3DKernel<ExecSpace, TData, true>(
                        shapeType, nq0, nq1, nq2, nCoord, nElmts, nSize,
                        m_dfSize, Z0, Z1, Z2, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, dbasis0, basis1, basis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, basis0, dbasis1, basis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr + nSize, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, basis0, basis1, dbasis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr + 2 * nSize, outPtr);
                }
                else
                {
                    IProductWRTDerivBase3DKernel<ExecSpace, TData, false>(
                        shapeType, nq0, nq1, nq2, nCoord, nElmts, nSize,
                        m_dfSize, Z0, Z1, Z2, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, dbasis0, basis1, basis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, basis0, dbasis1, basis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr + nSize, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, basis0, basis1, dbasis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr + 2 * nSize, outPtr);
                }
            }

            // Increment pointer and index for next element type.
            jacPtr += deformed ? nqTot * nElmts : nElmts;
            dfPtr += deformed ? nqTot * nElmts : nElmts;

            tmpPtr += nqTot * (nElmts + nPadElmts);
            inPtr += nqTot * (nElmts + nPadElmts);
            outPtr += nmTot * (nElmts + nPadElmts);
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
                    m_wspsize, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
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
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    BasisDataMap<TData> m_basisMap;
    BasisDataMap<TData> m_dbasisMap;
    BasisDataMap<TData> m_weightMap;
    BasisDataMap<TData> m_derivativeMap;
    BasisDataMap<TData> m_pointMap;

    MemoryRegion<TData> m_jac;
    MemoryRegion<TData> m_derivFac;
    MemoryRegion<TData> m_wsp;
    MemoryRegion<TData> m_tmp;

    size_t m_dfSize;
    size_t m_wspsize = 0;
    size_t m_tmpsize = 0;
};

} // namespace Nektar::Operators::detail
