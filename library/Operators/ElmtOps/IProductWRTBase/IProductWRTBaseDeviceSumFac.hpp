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

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacKernels.hpp"

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
        m_basisMap =
            GetBasisData<MemSpace, NekDouble, TData>(expansionList, eBasis);
        m_weightMap =
            GetBasisData<MemSpace, NekDouble, TData>(expansionList, eWeights);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (m_blk = 0; m_blk < in.GetBlocks().size(); ++m_blk)
        {
            m_expPtr = this->m_expansionList->GetExp(exp_idx);

            // Block dependent.
            auto &inblock  = in.GetBlocks()[m_blk];
            auto &outblock = out.GetBlocks()[m_blk];

            this->BlockOperator(inblock, outblock);

            // Increment index for next element type.
            exp_idx += inblock.GetNumElements();
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

        constexpr bool device_only = true;

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

    void BlockOperator(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock)
    {
        // Determine shape and type of the element.
        const auto shapeType = m_expPtr->DetShapeType();

        switch (shapeType)
        {
            // Segment
            case LibUtilities::Seg:
            {
                SegBlock(inblock, outblock);
                break;
            }
            // Quads
            case LibUtilities::Quad:
            {
                QuadBlock(inblock, outblock);
                break;
            }
            // Triangles
            case LibUtilities::Tri:
            {
                TriBlock(inblock, outblock);
                break;
            }
            // Hexes
            case LibUtilities::Hex:
            {
                HexBlock(inblock, outblock);
                break;
            }
            // Tet
            case LibUtilities::Tet:
            {
                TetBlock(inblock, outblock);
                break;
            }
            // Pyr
            case LibUtilities::Pyr:
            {
                PyrBlock(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                PrismBlock(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

private:
    unsigned int m_blk;

    LocalRegions::ExpansionSharedPtr m_expPtr;

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

    void SegBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator1D<LibUtilities::Seg, true>(inblock, outblock);
        }
        else
        {
            Operator1D<LibUtilities::Seg, false>(inblock, outblock);
        }
    }
    void TriBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator2D<LibUtilities::Tri, true>(inblock, outblock);
        }
        else
        {
            Operator2D<LibUtilities::Tri, false>(inblock, outblock);
        }
    }
    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator2D<LibUtilities::Quad, true>(inblock, outblock);
        }
        else
        {
            Operator2D<LibUtilities::Quad, false>(inblock, outblock);
        }
    }
    void HexBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator3D<LibUtilities::Hex, true>(inblock, outblock);
        }
        else
        {
            Operator3D<LibUtilities::Hex, false>(inblock, outblock);
        }
    }
    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator3D<LibUtilities::Prism, true>(inblock, outblock);
        }
        else
        {
            Operator3D<LibUtilities::Prism, false>(inblock, outblock);
        }
    }
    void PyrBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator3D<LibUtilities::Pyr, true>(inblock, outblock);
        }
        else
        {
            Operator3D<LibUtilities::Pyr, false>(inblock, outblock);
        }
    }
    void TetBlock(BlockAccessor<TData> &inblock, BlockAccessor<TData> &outblock)
    {
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        if (deformed)
        {
            Operator3D<LibUtilities::Tet, true>(inblock, outblock);
        }
        else
        {
            Operator3D<LibUtilities::Tet, false>(inblock, outblock);
        }
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        constexpr bool SharedMemory = true;
        constexpr bool Append       = false;

        // Shape size.
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nq0 = m_expPtr->GetNumPoints(0);

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto basis0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto w0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian data.
        auto jacptr = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = (this->m_lambda == 1.0)
                          ? outblock.template GetPtr<MemSpace, WriteOnly>()
                          : outblock.template GetPtr<MemSpace, ReadWrite>();

        // Reshape, if necessary.
        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
            inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
            (TData *)inptr);
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        if (this->m_lambda == 1.0)
        {
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                outblock.GetInterleaveWidth(), nElmtsPad, outblock.GetNumData(),
                outptr);
        }
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // IProduct kernel.
        if (this->m_lambda == 1.0)
        {
            constexpr bool Scale = false;

            IProductWRTBase1DKernel<ExecSpace, Implementation, Scale, Append,
                                    DEFORMED, SharedMemory>(
                nm0, nq0, nElmtsPad, basis0, w0, jacptr, inptr, outptr);
        }
        else
        {
            constexpr bool Scale = true;

            IProductWRTBase1DKernel<ExecSpace, Implementation, Scale, Append,
                                    DEFORMED, SharedMemory>(
                nm0, nq0, nElmtsPad, basis0, w0, jacptr, inptr, outptr,
                this->m_lambda);
        }
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        constexpr bool device_only  = true;
        constexpr bool SharedMemory = true;
        constexpr bool Append       = false;

        // Shape size.
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nm1 = m_expPtr->GetBasisNumModes(1);

        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);

        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto basis0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto basis1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto w0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto w1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian data.
        auto jacptr = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = (this->m_lambda == 1.0)
                          ? outblock.template GetPtr<MemSpace, WriteOnly>()
                          : outblock.template GetPtr<MemSpace, ReadWrite>();

        // Reshape, if necessary.
        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
            inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
            (TData *)inptr);
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        if (this->m_lambda == 1.0)
        {
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                outblock.GetInterleaveWidth(), nElmtsPad, outblock.GetNumData(),
                outptr);
        }
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Precompute index, if necessary.
        const bool indexing =
            SHAPE_TYPE == LibUtilities::Tri &&
            std::is_same_v<Implementation, Operators::SumFacQP>;

        if (indexing)
        {
            if (m_index0.find(basisKeys) == m_index0.end())
            {
                const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
                std::vector<unsigned int> index0(nm01);
                for (unsigned int p = 0, mode_pq = 0; p < nm0; p++)
                {
                    for (unsigned int q = 0; q < nm1 - p; q++, mode_pq++)
                    {
                        index0[mode_pq] = p;
                    }
                }
                m_index0[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index0, ExecSpace::alignment, device_only);
            }
        }

        auto index0 =
            indexing ? m_index0[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                     : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1, 0,
                                             nm0, nm1, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // IProduct kernel.
        if (this->m_lambda == 1.0)
        {
            constexpr bool Scale = false;

            IProductWRTBase2DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                    Scale, Append, DEFORMED, SharedMemory>(
                nm0, nm1, nq0, nq1, nElmtsPad, isModified, index0, basis0,
                basis1, w0, w1, jacptr, wspptr, inptr, outptr);
        }
        else
        {
            constexpr bool Scale = true;

            IProductWRTBase2DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                    Scale, Append, DEFORMED, SharedMemory>(
                nm0, nm1, nq0, nq1, nElmtsPad, isModified, index0, basis0,
                basis1, w0, w1, jacptr, wspptr, inptr, outptr, this->m_lambda);
        }
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        constexpr bool device_only  = true;
        constexpr bool SharedMemory = true;
        constexpr bool Append       = false;

        // Shape size.
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nm1 = m_expPtr->GetBasisNumModes(1);
        const auto nm2 = m_expPtr->GetBasisNumModes(2);

        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);
        const auto nq2 = m_expPtr->GetNumPoints(2);

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey(),
            m_expPtr->GetBasis(2)->GetBasisKey()};
        auto basis0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto basis1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto basis2 =
            m_basisMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto w0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto w1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto w2 =
            m_weightMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian data.
        auto jacptr = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = (this->m_lambda == 1.0)
                          ? outblock.template GetPtr<MemSpace, WriteOnly>()
                          : outblock.template GetPtr<MemSpace, ReadWrite>();

        // Reshape, if necessary.
        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
            inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
            (TData *)inptr);
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        if (this->m_lambda == 1.0)
        {
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                outblock.GetInterleaveWidth(), nElmtsPad, outblock.GetNumData(),
                outptr);
        }
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Precompute index, if necessary.
        const bool indexingTet =
            SHAPE_TYPE == LibUtilities::Tet &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        const bool indexingPrism =
            SHAPE_TYPE == LibUtilities::Prism &&
            std::is_same_v<Implementation, Operators::SumFacQP>;
        const bool indexingPyr =
            SHAPE_TYPE == LibUtilities::Pyr &&
            std::is_same_v<Implementation, Operators::SumFacQP>;

        // Precompute index, if necessary.
        if (indexingTet)
        {
            if (m_index0.find(basisKeys) == m_index0.end())
            {
                const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
                std::vector<unsigned int> index0(nm01);
                std::vector<unsigned int> index1(nmTot);
                std::vector<unsigned int> index2(nmTot);
                for (unsigned int p = 0, mode_pq = 0, mode_pqr = 0; p < nm0;
                     p++)
                {
                    for (unsigned int q = 0; q < nm1 - p; q++, mode_pq++)
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
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index0, ExecSpace::alignment, device_only);
                m_index1[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index1, ExecSpace::alignment, device_only);
                m_index2[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index2, ExecSpace::alignment, device_only);
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
                        for (unsigned int r = 0u; r < nm2 - p; r++, mode_pqr++)
                        {
                            unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;
                            unsigned int mode_pqr =
                                mode_pr * nm1 + (nm2 - p) * q + r;
                            index0[mode_pqr] = p;
                            index1[mode_pqr] = q;
                            index2[mode_pqr] = r;
                        }
                    }
                }
                m_index0[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index0, ExecSpace::alignment, device_only);
                m_index1[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index1, ExecSpace::alignment, device_only);
                m_index2[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index2, ExecSpace::alignment, device_only);
            }
        }

        if (indexingPyr)
        {
            if (m_index0.find(basisKeys) == m_index0.end())
            {
                std::vector<unsigned int> index0(nmTot);
                std::vector<unsigned int> index1(nmTot);
                m_index0[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index0, ExecSpace::alignment, device_only);
                m_index1[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index1, ExecSpace::alignment, device_only);
                for (unsigned int p = 0, mode_pqr = 0; p < nm0; p++)
                {
                    for (unsigned int q = 0u; q < nm1; q++)
                    {
                        for (unsigned int r = 0; r < nm2 - std::max(p, q);
                             r++, mode_pqr++)
                        {
                            index0[mode_pqr] = p;
                            index1[mode_pqr] = q;
                        }
                    }
                }
                m_index0[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index0, ExecSpace::alignment, device_only);
                m_index1[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index1, ExecSpace::alignment, device_only);
            }
        }

        auto index0 =
            (indexingTet || indexingPrism || indexingPyr)
                ? m_index0[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                : nullptr;
        auto index1 =
            (indexingTet || indexingPrism || indexingPyr)
                ? m_index1[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                : nullptr;
        auto index2 =
            (indexingTet || indexingPrism)
                ? m_index2[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nq0, nq1,
                                             nq2, nm0, nm1, nm2));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // IProduct kernel.
        if (this->m_lambda == 1.0)
        {
            constexpr bool Scale = false;

            IProductWRTBase3DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                    Scale, Append, DEFORMED, SharedMemory>(
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, isModified, index0,
                index1, index2, basis0, basis1, basis2, w0, w1, w2, jacptr,
                wspptr, inptr, outptr);
        }
        else
        {
            constexpr bool Scale = true;

            IProductWRTBase3DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                    Scale, Append, DEFORMED, SharedMemory>(
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, isModified, index0,
                index1, index2, basis0, basis1, basis2, w0, w1, w2, jacptr,
                wspptr, inptr, outptr, this->m_lambda);
        }
    }
};

} // namespace Nektar::Operators::detail
