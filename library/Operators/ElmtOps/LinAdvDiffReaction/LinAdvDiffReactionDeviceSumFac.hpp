///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionDeviceSumFac.hpp
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
#include "Operators/ElmtOps/OperatorLinAdvDiffReaction.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorLinAdvDiffReactionImpl : public OperatorLinAdvDiffReaction<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorLinAdvDiffReactionImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinAdvDiffReaction<TData>(expansionList),
          m_diffCoeff(MemoryRegion<TData>::template Create<MemSpace>(
              "LinAdvDiffReaction diffCoeff",
              expansionList->GetCoordim(0) * expansionList->GetCoordim(0),
              ExecSpace::alignment))
    {
        auto nCoord = this->m_expansionList->GetCoordim(0);

        m_diffCoeff.template Initialize<MemSpace>(0);

        TData *diffCoeff =
            m_diffCoeff.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

        if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
        {
            for (size_t d = 0; d < nCoord; d++)
            {
                diffCoeff[d * nCoord + d] = 1.0;
            }
        }
        else if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            diffCoeff[0] = 1.0; // D00
            if (nCoord >= 2)
            {
                diffCoeff[2] = 1.0; // D11
                if (nCoord == 3)
                {
                    diffCoeff[5] = 1.0; // D22
                }
            }
        }

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
        m_weightMap =
            GetBasisData<MemSpace, NekDouble, TData>(expansionList, eWeights);
        m_derivativeMap = GetBasisData<MemSpace, NekDouble, TData>(
            expansionList, eDerivative);

        // Initialize the geometric factors.
        m_fac0 = GetBasisData<MemSpace, NekDouble, TData>(
            expansionList, eHalfMultOnePlusZero, ExecSpace::alignment);
        m_fac1 = GetBasisData<MemSpace, NekDouble, TData>(
            expansionList, eTwoOverOneMinusZero, ExecSpace::alignment);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Initialize index.
        size_t exp_idx = 0;

        m_nComps = out.GetNumComponents();
        ASSERTL1(m_nComps == in.GetNumComponents(),
                 "Number of input and output "
                 "components differ");

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
                                  size_t nElmts, [[maybe_unused]] size_t ncoord,
                                  [[maybe_unused]] size_t nq0, size_t nq1,
                                  size_t nq2, [[maybe_unused]] size_t nm0,
                                  size_t nm1, size_t nm2)
    {
        size_t wspsize = 0;

        if (shapeType == LibUtilities::Seg)
        {
            wspsize = (1 + ncoord) * nq0 * nElmts;
        }
        else if (shapeType == LibUtilities::Quad)
        {
            wspsize = ((1 + ncoord) * nq0 * nq1 + nq1) * nElmts;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = ((1 + ncoord) * nq0 * nq1 + std::max(nq1, nm0)) * nElmts;
        }
        else if (shapeType == LibUtilities::Hex)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2) * nElmts;
        }
        else if (shapeType == LibUtilities::Tet)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + nq1 * nq2 + nq2 + nm2) * nElmts;
        }
        else if (shapeType == LibUtilities::Prism)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                       std::max(nq2, nm0) + nm1) *
                      nElmts;
        }
        else if (shapeType == LibUtilities::Pyr)
        {
            wspsize = (4 * nq0 * nq1 * nq2 + std::max(nq1 * nq2, nm0 * nm1) +
                       std::max(nq2, nm0)) *
                      nElmts;
        }

        return wspsize;
    }

    MemoryRegion<TData> SetWorkspace(LibUtilities::ShapeType shapeType,
                                     size_t nElmts, size_t ncoord, size_t nq0,
                                     size_t nq1, size_t nq2, size_t nm0,
                                     size_t nm1, size_t nm2)
    {
        size_t wspsize = GetSharedWorkspaceSize(shapeType, nElmts, ncoord, nq0,
                                                nq1, nq2, nm0, nm1, nm2);

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
            OperatorLinAdvDiffReactionImpl<ExecSpace, Implementation, TData>>(
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

    void v_SetAdvVel(const int nVel, const Array<OneD, NekDouble> &Vel) override
    {
        if (m_advVel.GetNumComponents() != nVel)
        {
            // set up a physBlockAttributes which will be
            std::vector<BlockAttributes> physBlockAttributes =
                GetBlockAttributes<TData>(FieldState::Phys,
                                          this->m_expansionList, 1);
            m_advVel =
                Field<TData, FieldState::Phys>::template Create<MemSpace>(
                    "Advection Field", physBlockAttributes, nVel,
                    ExecSpace::alignment);
        }

        // set interleave width to 1.
        for (m_blk = 0; m_blk < m_advVel.GetBlocks().size(); ++m_blk)
        {
            auto &blk = m_advVel.GetBlocks()[m_blk];

            blk.template SetInterleaveWidth<TData>(1);
        }
        m_advVel.template CopyArray<NektarSpaces::HostSpace>(Vel);
    }

private:
    MemoryRegion<TData> m_diffCoeff;

    unsigned int m_blk;
    size_t m_nComps;

    LocalRegions::ExpansionSharedPtr m_expPtr;

    Field<TData, FieldState::Phys> m_advVel;
    BasisDataMap<TData> m_basisMap;
    BasisDataMap<TData> m_weightMap;
    BasisDataMap<TData> m_derivativeMap;
    BasisDataMap<TData> m_fac0;
    BasisDataMap<TData> m_fac1;

    std::vector<MemoryRegion<TData>> m_jac;
    std::vector<MemoryRegion<TData>> m_df;
    std::vector<MemoryRegion<TData>> m_wsp;

    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index0;
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index1;
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index2;
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<unsigned int>>
        m_index3;

    static constexpr size_t m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::vector_width<TData>::value
            : 1u;

    void SegBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);
    void TriBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);
    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock);
    void HexBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);
    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock);
    void PyrBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);
    void TetBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nq0 = m_expPtr->GetNumPoints(0);

        const auto nCoord = m_expPtr->GetCoordim();

        // Get Basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacptr  = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto dfptr   = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto diffptr = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // advVel pointers
        auto &advVelblock = m_advVel.GetBlocks()[m_blk];

        auto advVelptr =
            (advVelblock.GetInterleaveWidth() == m_implInterleaveWidth)
                ? advVelblock.template GetPtr<MemSpace, ReadOnly>()
                : advVelblock.template GetPtr<MemSpace, ReadWrite>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nCoord, nq0,
                                             0, 0, nm0, 0, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            if (nc == 0) // only need to interlace adv vel once
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr);
            }

            // LinAdvDiffReaction kernel.
            LinAdvDiffReaction1DKernel<ExecSpace, Implementation, DEFORMED>(
                nCoord, nm0, nq0, nElmtsPad, B0, D0, W0, dfptr, jacptr, diffptr,
                advVelptr, inptr, outptr, wspptr, this->m_lambda);

            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        advVelblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nCoord = m_expPtr->GetCoordim();

        // Get Basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacptr  = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto dfptr   = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto diffptr = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // advVel pointers
        auto &advVelblock = m_advVel.GetBlocks()[m_blk];

        auto advVelptr =
            (advVelblock.GetInterleaveWidth() == m_implInterleaveWidth)
                ? advVelblock.template GetPtr<MemSpace, ReadOnly>()
                : advVelblock.template GetPtr<MemSpace, ReadWrite>();

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nCoord, nq0,
                                             0, 0, nm0, 0, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            if (nc == 0) // only need to interlace adv vel once
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr);
            }

            // LinAdvDiffReaction kernel.
            LinAdvDiffReaction1DKernel<ExecSpace, Implementation, DEFORMED>(
                nCoord, nm0, nq0, nElmtsPad, B0, D0, W0, dfptr, jacptr, diffptr,
                advVelptr, inptr, outptr, wspptr, this->m_lambda);

            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        advVelblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        constexpr bool device_only = true;

        // Shape size.
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nm1 = m_expPtr->GetBasisNumModes(1);

        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);

        const auto nCoord = m_expPtr->GetCoordim();

        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto f1 = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacptr  = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto dfptr   = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto diffptr = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // advVel pointers
        auto &advVelblock = m_advVel.GetBlocks()[m_blk];

        auto advVelptr =
            (advVelblock.GetInterleaveWidth() == m_implInterleaveWidth)
                ? advVelblock.template GetPtr<MemSpace, ReadOnly>()
                : advVelblock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelSize = advVelblock.size();

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
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nCoord, nq0,
                                             nq1, 0, nm0, nm1, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            if (nc == 0) // only need to interlace adv vel once
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr);

                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr + advVelSize);
            }

            // LinAdvDiffReaction kernel.
            LinAdvDiffReaction2DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                       DEFORMED>(
                nCoord, nm0, nm1, nq0, nq1, nElmtsPad, isModified, index0, B0,
                B1, D0, D1, W0, W1, f0, f1, dfptr, jacptr, diffptr, advVelptr,
                advVelptr + advVelSize, inptr, outptr, wspptr, this->m_lambda);

            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        advVelblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        constexpr bool device_only = true;

        const auto nCoord = m_expPtr->GetCoordim();

        // Flag for collapsed coordinate correction.
        const bool isModified =
            m_expPtr->GetBasis(0)->GetBasisType() == LibUtilities::eModified_A;

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto f1 = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacptr  = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto dfptr   = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto diffptr = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // advVel pointers
        auto &advVelblock = m_advVel.GetBlocks()[m_blk];

        auto advVelptr =
            (advVelblock.GetInterleaveWidth() == m_implInterleaveWidth)
                ? advVelblock.template GetPtr<MemSpace, ReadOnly>()
                : advVelblock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelSize = advVelblock.size();

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
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, nCoord, nq0,
                                             nq1, 0, nm0, nm1, 0));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            if (nc == 0) // only need to interlace adv vel once
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr);

                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr + advVelSize);
            }

            // LinAdvDiffReaction kernel.
            LinAdvDiffReaction2DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                       DEFORMED>(
                nCoord, nm0, nm1, nq0, nq1, nElmtsPad, isModified, index0, B0,
                B1, D0, D1, W0, W1, f0, f1, dfptr, jacptr, diffptr, advVelptr,
                advVelptr + advVelSize, inptr, outptr, wspptr, this->m_lambda);

            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        advVelblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        constexpr bool device_only = true;

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
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto B2 =
            m_basisMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D2 =
            m_derivativeMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W2 =
            m_weightMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto f0  = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto f1  = m_fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f1m = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f2  = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacptr  = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto dfptr   = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto diffptr = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // advVel pointers
        auto &advVelblock = m_advVel.GetBlocks()[m_blk];

        auto advVelptr =
            (advVelblock.GetInterleaveWidth() == m_implInterleaveWidth)
                ? advVelblock.template GetPtr<MemSpace, ReadOnly>()
                : advVelblock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelSize = advVelblock.size();

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
                std::vector<unsigned int> index3(nm01);
                for (unsigned int p = 0, mode_pq = 0, mode_pqr = 0; p < nm0;
                     p++)
                {
                    for (unsigned int q = 0; q < nm1 - p; q++, mode_pq++)
                    {
                        index0[mode_pq] = p;
                        index3[mode_pq] = q;
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
                m_index3[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index3, ExecSpace::alignment, device_only);
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
        auto index3 =
            (indexingTet)
                ? m_index3[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, 3, nq0, nq1,
                                             nq2, nm0, nm1, nm2));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            if (nc == 0) // only need to interlace adv vel once
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr);

                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr + advVelSize);

                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(),
                    (TData *)advVelptr + 2 * advVelSize);
            }

            // LinAdvDiffReaction kernel.
            LinAdvDiffReaction3DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                       DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, isModified, index0,
                index1, index2, index3, B0, B1, B2, D0, D1, D2, W0, W1, W2, f0,
                f1, f1m, f2, dfptr, jacptr, diffptr, advVelptr,
                advVelptr + advVelSize, advVelptr + 2 * advVelSize, inptr,
                outptr, wspptr, this->m_lambda);

            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        advVelblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        constexpr bool device_only = true;

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
        auto B0 =
            m_basisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 =
            m_basisMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto B2 =
            m_basisMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D2 =
            m_derivativeMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W2 =
            m_weightMap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        auto f0  = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto f1  = m_fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f1m = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto f2  = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacptr  = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto dfptr   = m_df[m_blk].template GetPtr<MemSpace, ReadOnly>();
        auto diffptr = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const auto nElmtsPad = inblock.GetNumElementsWithPadding();

        // advVel pointers
        auto &advVelblock = m_advVel.GetBlocks()[m_blk];

        auto advVelptr =
            (advVelblock.GetInterleaveWidth() == m_implInterleaveWidth)
                ? advVelblock.template GetPtr<MemSpace, ReadOnly>()
                : advVelblock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelSize = advVelblock.size();

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
                std::vector<unsigned int> index3(nm01);
                for (unsigned int p = 0, mode_pq = 0, mode_pqr = 0; p < nm0;
                     p++)
                {
                    for (unsigned int q = 0; q < nm1 - p; q++, mode_pq++)
                    {
                        index0[mode_pq] = p;
                        index3[mode_pq] = q;
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
                m_index3[basisKeys] =
                    MemoryRegion<unsigned int>::template FromVector<MemSpace>(
                        index3, ExecSpace::alignment, device_only);
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
        auto index3 =
            (indexingTet)
                ? m_index3[basisKeys].template GetPtr<MemSpace, ReadOnly>()
                : nullptr;

        // Set workspace.
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            if (m_wsp.size() <= m_blk)
            {
                m_wsp.push_back(SetWorkspace(SHAPE_TYPE, nElmtsPad, 3, nq0, nq1,
                                             nq2, nm0, nm1, nm2));
            }
        }

        // Get workspace pointer.
        auto wspptr = std::is_same_v<Implementation, Operators::SumFac>
                          ? m_wsp[m_blk].template GetPtr<MemSpace, WriteOnly>()
                          : nullptr;

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inptr);

            if (nc == 0) // only need to interlace adv vel once
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr);

                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(), (TData *)advVelptr + advVelSize);

                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    advVelblock.GetInterleaveWidth(), nElmtsPad,
                    advVelblock.GetNumData(),
                    (TData *)advVelptr + 2 * advVelSize);
            }

            // LinAdvDiffReaction kernel.
            LinAdvDiffReaction3DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                       DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, nElmtsPad, isModified, index0,
                index1, index2, index3, B0, B1, B2, D0, D1, D2, W0, W1, W2, f0,
                f1, f1m, f2, dfptr, jacptr, diffptr, advVelptr,
                advVelptr + advVelSize, advVelptr + 2 * advVelSize, inptr,
                outptr, wspptr, this->m_lambda);

            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        advVelblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
