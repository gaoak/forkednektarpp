///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseDeviceSumFac.hpp
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
#include "Operators/ElmtOps/OperatorIProductWRTDerivBase.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSYCLSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseSYCLSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorIProductWRTDerivBaseImpl
    : public OperatorIProductWRTDerivBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
        // Initialise the jacobian and the derivative factor.
        auto transpose = std::is_same_v<Implementation, Operators::SumFacQP>;
        auto locblocks = GetBlockAttributes<TData>(
            FieldState::Phys, expansionList, m_implInterleaveWidth);
        m_jac = SetJacobian<MemSpace, TData>(expansionList, locblocks,
                                             ExecSpace::alignment);
        m_df  = SetDerivativeFactor<MemSpace, TData>(
            expansionList, locblocks, ExecSpace::alignment, transpose);

        // Initialize the basis data.
        m_basisMap =
            GetBasisData<MemSpace, NekDouble, TData>(expansionList, eBasis);
        m_dbasisMap = GetBasisData<MemSpace, NekDouble, TData>(
            expansionList, eBasisDerivative);
        m_weightMap =
            GetBasisData<MemSpace, NekDouble, TData>(expansionList, eWeights);
        m_pointMap =
            GetBasisData<MemSpace, NekDouble, TData>(expansionList, eZeros);
        m_derivativeMap = GetBasisData<MemSpace, NekDouble, TData>(
            expansionList, eDerivative);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               bool APPEND = false) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        const bool device_only = true;

        // Zero output.
        if (!APPEND)
        {
            out.template Initialize<MemSpace>(0);
        }

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (size_t blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            // Block dependent.
            auto &inblock        = in.GetBlocks()[blk];
            auto &outblock       = out.GetBlocks()[blk];
            const auto nElmts    = inblock.GetNumElements();
            const auto nElmtsPad = inblock.GetNumElementsWithPadding();

            // Initialize pointers.
            auto inPtr = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                             ? inblock.template GetPtr<MemSpace, ReadOnly>()
                             : inblock.template GetPtr<MemSpace, ReadWrite>();
            auto outPtr = APPEND
                              ? outblock.template GetPtr<MemSpace, ReadWrite>()
                              : outblock.template GetPtr<MemSpace, WriteOnly>();
            auto jacPtr = m_jac[blk].template GetPtr<MemSpace, ReadOnly>();
            auto dfPtr  = m_df[blk].template GetPtr<MemSpace, ReadOnly>();

            // Determine shape and type of the element.
            const auto expPtr    = this->m_expansionList->GetExp(exp_idx);
            const auto shapeType = expPtr->DetShapeType();
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
            if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
            {
                if (m_wsp.size() <= blk)
                {
                    m_wsp.push_back(SetWorkspace(shapeType, nElmtsPad, nq0, nq1,
                                                 nq2, nm0, nm1, nm2));
                }
            }

            if (m_tmp.size() <= blk)
            {
                m_tmp.push_back(MemoryRegion<TData>::template Create<MemSpace>(
                    nCoord * inblock.size(), ExecSpace::alignment,
                    device_only));
            }

            // Get workspace pointer.
            auto wspPtr =
                std::is_same_v<Implementation, Operators::SumFac>
                    ? m_wsp[blk].template GetPtr<MemSpace, WriteOnly>()
                    : nullptr;
            TData *tmpPtr = m_tmp[blk].template GetPtr<MemSpace, WriteOnly>();

            constexpr bool SharedMemory = true;
            constexpr bool Scale        = false;
            constexpr bool Append       = true;

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inPtr);
            if (nCoord > 1)
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(), nElmtsPad,
                    inblock.GetNumData(), (TData *)inPtr + inblock.size());
            }
            if (nCoord > 2)
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(), nElmtsPad,
                    inblock.GetNumData(), (TData *)inPtr + 2 * inblock.size());
            }
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                outblock.GetInterleaveWidth(), nElmtsPad, outblock.GetNumData(),
                outPtr);
            inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
            outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

            // Function call to kernel functions.
            if (dimension == 1)
            {
                if (deformed)
                {
                    constexpr bool Deformed = true;

                    IProductWRTDerivBase1DKernel<ExecSpace, Implementation,
                                                 Deformed>(
                        nq0, nCoord, nElmtsPad, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase1DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        nm0, nq0, nElmtsPad, dbasis0, w0, jacPtr, tmpPtr,
                        outPtr);
                }
                else
                {
                    constexpr bool Deformed = false;

                    IProductWRTDerivBase1DKernel<ExecSpace, Implementation,
                                                 Deformed>(
                        nq0, nCoord, nElmtsPad, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase1DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        nm0, nq0, nElmtsPad, dbasis0, w0, jacPtr, tmpPtr,
                        outPtr);
                }
            }
            else if (dimension == 2)
            {
#if !defined(NEKTAR_USE_QP_1D_KERNEL)
                const bool indexing = false;
#else
                const bool indexing =
                    shapeType == LibUtilities::Tri &&
                    std::is_same_v<Implementation, Operators::SumFacQP>;
#endif
                std::vector<LibUtilities::BasisKey> basisKeys{
                    expPtr->GetBasis(0)->GetBasisKey(),
                    expPtr->GetBasis(1)->GetBasisKey()};

                // Precompute index, if necessary.
                if (indexing)
                {
                    if (m_index0.find(basisKeys) == m_index0.end())
                    {
                        const unsigned int nm01 =
                            (2u * nm1 - nm0 + 1u) * nm0 / 2u;
                        std::vector<unsigned int> index0(nm01);
                        for (unsigned int p = 0, mode_pq = 0; p < nm0; p++)
                        {
                            for (unsigned int q = 0; q < nm1 - p;
                                 q++, mode_pq++)
                            {
                                index0[mode_pq] = p;
                            }
                        }
                        m_index0[basisKeys] =
                            MemoryRegion<unsigned int>::template FromVector<
                                MemSpace>(index0, ExecSpace::alignment,
                                          device_only);
                    }
                }

                auto index0 = indexing
                                  ? m_index0[basisKeys]
                                        .template GetPtr<MemSpace, ReadOnly>()
                                  : nullptr;

                if (deformed)
                {
                    constexpr bool Deformed = true;

                    IProductWRTDerivBase2DKernel<ExecSpace, Implementation,
                                                 Deformed>(
                        shapeType, nq0, nq1, nCoord, nElmtsPad, Z0, Z1, dfPtr,
                        inPtr, tmpPtr);
                    IProductWRTBase2DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                        index0, dbasis0, basis1, w0, w1, jacPtr, wspPtr, tmpPtr,
                        outPtr);
                    IProductWRTBase2DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                        index0, basis0, dbasis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr + nElmtsPad * nqTot, outPtr);
                }
                else
                {
                    constexpr bool Deformed = false;

                    IProductWRTDerivBase2DKernel<ExecSpace, Implementation,
                                                 Deformed>(
                        shapeType, nq0, nq1, nCoord, nElmtsPad, Z0, Z1, dfPtr,
                        inPtr, tmpPtr);
                    IProductWRTBase2DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                        index0, dbasis0, basis1, w0, w1, jacPtr, wspPtr, tmpPtr,
                        outPtr);
                    IProductWRTBase2DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                        index0, basis0, dbasis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr + nElmtsPad * nqTot, outPtr);
                }
            }
            else if (dimension == 3)
            {
#if !defined(NEKTAR_USE_QP_1D_KERNEL)
                const bool indexingTet   = false;
                const bool indexingPrism = false;
                const bool indexingPyr   = false;
#else
                const bool indexingTet =
                    shapeType == LibUtilities::Tet &&
                    std::is_same_v<Implementation, Operators::SumFacQP>;
                const bool indexingPrism =
                    shapeType == LibUtilities::Prism &&
                    std::is_same_v<Implementation, Operators::SumFacQP>;
                const bool indexingPyr =
                    shapeType == LibUtilities::Pyr &&
                    std::is_same_v<Implementation, Operators::SumFacQP>;
#endif
                std::vector<LibUtilities::BasisKey> basisKeys{
                    expPtr->GetBasis(0)->GetBasisKey(),
                    expPtr->GetBasis(1)->GetBasisKey(),
                    expPtr->GetBasis(2)->GetBasisKey()};

                // Precompute index, if necessary.
                if (indexingTet)
                {
                    if (m_index0.find(basisKeys) == m_index0.end())
                    {
                        const unsigned int nm01 =
                            (2u * nm1 - nm0 + 1u) * nm0 / 2u;
                        std::vector<unsigned int> index0(nm01);
                        std::vector<unsigned int> index1(nmTot);
                        std::vector<unsigned int> index2(nmTot);
                        for (unsigned int p = 0, mode_pq = 0, mode_pqr = 0;
                             p < nm0; p++)
                        {
                            for (unsigned int q = 0; q < nm1 - p;
                                 q++, mode_pq++)
                            {
                                index0[mode_pq] = p;
                                for (unsigned int r = 0; r < nm2 - p - q;
                                     r++, mode_pqr++)
                                {
                                    index1[mode_pqr] = p;
                                    index2[mode_pqr] = q;
                                }
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
                        m_index2[basisKeys] =
                            MemoryRegion<unsigned int>::template FromVector<
                                MemSpace>(index2, ExecSpace::alignment,
                                          device_only);
                    }
                }

                if (indexingPrism)
                {
                    if (m_index0.find(basisKeys) == m_index0.end())
                    {
                        std::vector<unsigned int> index0(nmTot);
                        std::vector<unsigned int> index1(nmTot);
                        std::vector<unsigned int> index2(nmTot);
                        for (unsigned int p = 0, mode_pqr = 0; p < nm0; p++)
                        {
                            for (unsigned int q = 0u; q < nm1; q++)
                            {
                                for (unsigned int r = 0u; r < nm2 - p;
                                     r++, mode_pqr++)
                                {
                                    unsigned int mode_pr =
                                        (2u * nm2 - p + 1u) * p / 2u;
                                    unsigned int mode_pqr =
                                        mode_pr * nm1 + (nm2 - p) * q + r;
                                    index0[mode_pqr] = p;
                                    index1[mode_pqr] = q;
                                    index2[mode_pqr] = r;
                                }
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
                        m_index2[basisKeys] =
                            MemoryRegion<unsigned int>::template FromVector<
                                MemSpace>(index2, ExecSpace::alignment,
                                          device_only);
                    }
                }

                if (indexingPyr)
                {
                    if (m_index0.find(basisKeys) == m_index0.end())
                    {
                        std::vector<unsigned int> index0(nmTot);
                        std::vector<unsigned int> index1(nmTot);
                        m_index0[basisKeys] =
                            MemoryRegion<unsigned int>::template FromVector<
                                MemSpace>(index0, ExecSpace::alignment,
                                          device_only);
                        m_index1[basisKeys] =
                            MemoryRegion<unsigned int>::template FromVector<
                                MemSpace>(index1, ExecSpace::alignment,
                                          device_only);
                        for (unsigned int p = 0, mode_pqr = 0; p < nm0; p++)
                        {
                            for (unsigned int q = 0u; q < nm1; q++)
                            {
                                for (unsigned int r = 0;
                                     r < nm2 - std::max(p, q); r++, mode_pqr++)
                                {
                                    index0[mode_pqr] = p;
                                    index1[mode_pqr] = q;
                                }
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

                auto index0 = (indexingTet || indexingPrism || indexingPyr)
                                  ? m_index0[basisKeys]
                                        .template GetPtr<MemSpace, ReadOnly>()
                                  : nullptr;
                auto index1 = (indexingTet || indexingPrism || indexingPyr)
                                  ? m_index1[basisKeys]
                                        .template GetPtr<MemSpace, ReadOnly>()
                                  : nullptr;
                auto index2 = (indexingTet || indexingPrism)
                                  ? m_index2[basisKeys]
                                        .template GetPtr<MemSpace, ReadOnly>()
                                  : nullptr;

                if (deformed)
                {
                    constexpr bool Deformed = true;

                    IProductWRTDerivBase3DKernel<ExecSpace, Implementation,
                                                 Deformed>(
                        shapeType, nq0, nq1, nq2, nCoord, nElmtsPad, Z0, Z1, Z2,
                        dfPtr, inPtr, tmpPtr);
                    IProductWRTBase3DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                        correct, index0, index1, index2, dbasis0, basis1,
                        basis2, w0, w1, w2, jacPtr, wspPtr, tmpPtr, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                        correct, index0, index1, index2, basis0, dbasis1,
                        basis2, w0, w1, w2, jacPtr, wspPtr,
                        tmpPtr + nElmtsPad * nqTot, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                        correct, index0, index1, index2, basis0, basis1,
                        dbasis2, w0, w1, w2, jacPtr, wspPtr,
                        tmpPtr + 2 * nElmtsPad * nqTot, outPtr);
                }
                else
                {
                    constexpr bool Deformed = false;

                    IProductWRTDerivBase3DKernel<ExecSpace, Implementation,
                                                 Deformed>(
                        shapeType, nq0, nq1, nq2, nCoord, nElmtsPad, Z0, Z1, Z2,
                        dfPtr, inPtr, tmpPtr);
                    IProductWRTBase3DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                        correct, index0, index1, index2, dbasis0, basis1,
                        basis2, w0, w1, w2, jacPtr, wspPtr, tmpPtr, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                        correct, index0, index1, index2, basis0, dbasis1,
                        basis2, w0, w1, w2, jacPtr, wspPtr,
                        tmpPtr + nElmtsPad * nqTot, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, Implementation, Scale,
                                            Append, Deformed, SharedMemory>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                        correct, index0, index1, index2, basis0, basis1,
                        dbasis2, w0, w1, w2, jacPtr, wspPtr,
                        tmpPtr + 2 * nElmtsPad * nqTot, outPtr);
                }
            }

            // Increment index for next element type.
            exp_idx += nElmts;
        }
    }

    size_t GetSharedWorkspaceSize(LibUtilities::ShapeType shapeType,
                                  size_t nElmts, [[maybe_unused]] size_t nq0,
                                  size_t nq1, size_t nq2,
                                  [[maybe_unused]] size_t nm0, size_t nm1,
                                  size_t nm2)
    {
        size_t wspsize = 0;

        if (shapeType == LibUtilities::Quad)
        {
            wspsize = nq1 * nElmts;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = nq1 * nElmts;
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

    MemoryRegion<TData> SetWorkspace(LibUtilities::ShapeType shapeType,
                                     size_t nElmts, size_t nq0, size_t nq1,
                                     size_t nq2, size_t nm0, size_t nm1,
                                     size_t nm2)
    {
        size_t wspsize = GetSharedWorkspaceSize(shapeType, nElmts, nq0, nq1,
                                                nq2, nm0, nm1, nm2);

        const bool device_only = true;

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
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    BasisDataMap<TData> m_basisMap;
    BasisDataMap<TData> m_dbasisMap;
    BasisDataMap<TData> m_weightMap;
    BasisDataMap<TData> m_derivativeMap;
    BasisDataMap<TData> m_pointMap;

    std::vector<MemoryRegion<TData>> m_jac;
    std::vector<MemoryRegion<TData>> m_df;
    std::vector<MemoryRegion<TData>> m_wsp;
    std::vector<MemoryRegion<TData>> m_tmp;

    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index0;
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index1;
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index2;

    static constexpr size_t m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::vector_width<TData>::value
            : 1u;
};

} // namespace Nektar::Operators::detail
