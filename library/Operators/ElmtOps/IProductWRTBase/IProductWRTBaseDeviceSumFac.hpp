///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseDeviceSumFac.hpp
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
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSYCLSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise the jacobian.
        auto locblocks = GetBlockAttributes<TData>(
            FieldState::Phys, expansionList, m_implInterleaveWidth);
        m_jac = SetJacobian<MemSpace, TData>(expansionList, locblocks,
                                             ExecSpace::alignment);

        // Initialize the basis data.
        m_basisMap  = GetBasisData<MemSpace, TData>(expansionList, eBasis);
        m_weightMap = GetBasisData<MemSpace, TData>(expansionList, eWeights);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               const TData lambda = 1.0) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

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
            auto outPtr = (lambda == 1.0)
                              ? outblock.template GetPtr<MemSpace, WriteOnly>()
                              : outblock.template GetPtr<MemSpace, ReadWrite>();
            auto jacPtr = m_jac[blk].template GetPtr<MemSpace, ReadOnly>();

            // Determine shape and type of the element.
            const auto expPtr    = this->m_expansionList->GetExp(exp_idx);
            const auto shapeType = expPtr->DetShapeType();
            const auto deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
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
            if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
            {
                if (m_wsp.size() <= blk)
                {
                    m_wsp.push_back(SetWorkspace(shapeType, nElmtsPad, nq0, nq1,
                                                 nq2, nm0, nm1, nm2));
                }
            }

            // Get workspace pointer.
            auto wspPtr =
                std::is_same_v<Implementation, Operators::SumFac>
                    ? m_wsp[blk].template GetPtr<MemSpace, WriteOnly>()
                    : nullptr;

            constexpr bool SharedMemory = true;
            constexpr bool Append       = false;

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inPtr);
            inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
            if (lambda == 1.0)
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    outblock.GetInterleaveWidth(), nElmtsPad,
                    outblock.GetNumData(), outPtr);
            }
            outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

            // Function call to kernel functions.
            if (dimension == 1)
            {
                if (deformed)
                {
                    constexpr bool Deformed = true;

                    if (lambda == 1.0)
                    {
                        constexpr bool Scale = false;

                        IProductWRTBase1DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            nm0, nq0, nElmtsPad, basis0, w0, jacPtr, inPtr,
                            outPtr);
                    }
                    else
                    {
                        constexpr bool Scale = true;

                        IProductWRTBase1DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            nm0, nq0, nElmtsPad, basis0, w0, jacPtr, inPtr,
                            outPtr, lambda);
                    }
                }
                else
                {
                    constexpr bool Deformed = false;

                    if (lambda == 1.0)
                    {
                        constexpr bool Scale = false;

                        IProductWRTBase1DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            nm0, nq0, nElmtsPad, basis0, w0, jacPtr, inPtr,
                            outPtr);
                    }
                    else
                    {
                        constexpr bool Scale = true;

                        IProductWRTBase1DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            nm0, nq0, nElmtsPad, basis0, w0, jacPtr, inPtr,
                            outPtr, lambda);
                    }
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
                    const bool device_only = true;

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

                    if (lambda == 1.0)
                    {
                        constexpr bool Scale = false;

                        IProductWRTBase2DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                            index0, basis0, basis1, w0, w1, jacPtr, wspPtr,
                            inPtr, outPtr);
                    }
                    else
                    {
                        constexpr bool Scale = true;

                        IProductWRTBase2DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                            index0, basis0, basis1, w0, w1, jacPtr, wspPtr,
                            inPtr, outPtr, lambda);
                    }
                }
                else
                {
                    constexpr bool Deformed = false;

                    if (lambda == 1.0)
                    {
                        constexpr bool Scale = false;

                        IProductWRTBase2DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                            index0, basis0, basis1, w0, w1, jacPtr, wspPtr,
                            inPtr, outPtr);
                    }
                    else
                    {
                        constexpr bool Scale = true;

                        IProductWRTBase2DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            shapeType, nm0, nm1, nq0, nq1, nElmtsPad, correct,
                            index0, basis0, basis1, w0, w1, jacPtr, wspPtr,
                            inPtr, outPtr, lambda);
                    }
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
                    const bool device_only = true;

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
                    const bool device_only = true;

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
                    const bool device_only = true;

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

                    if (lambda == 1.0)
                    {
                        constexpr bool Scale = false;

                        IProductWRTBase3DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                            correct, index0, index1, index2, basis0, basis1,
                            basis2, w0, w1, w2, jacPtr, wspPtr, inPtr, outPtr);
                    }
                    else
                    {
                        constexpr bool Scale = true;

                        IProductWRTBase3DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                            correct, index0, index1, index2, basis0, basis1,
                            basis2, w0, w1, w2, jacPtr, wspPtr, inPtr, outPtr,
                            lambda);
                    }
                }
                else
                {
                    constexpr bool Deformed = false;

                    if (lambda == 1.0)
                    {
                        constexpr bool Scale = false;

                        IProductWRTBase3DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                            correct, index0, index1, index2, basis0, basis1,
                            basis2, w0, w1, w2, jacPtr, wspPtr, inPtr, outPtr);
                    }
                    else
                    {
                        constexpr bool Scale = true;

                        IProductWRTBase3DKernel<ExecSpace, Implementation,
                                                Scale, Append, Deformed,
                                                SharedMemory>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad,
                            correct, index0, index1, index2, basis0, basis1,
                            basis2, w0, w1, w2, jacPtr, wspPtr, inPtr, outPtr,
                            lambda);
                    }
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
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    BasisDataMap<TData> m_basisMap;
    BasisDataMap<TData> m_weightMap;
    std::vector<MemoryRegion<TData>> m_jac;
    std::vector<MemoryRegion<TData>> m_wsp;
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
