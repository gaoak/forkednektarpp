///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionSerialAVXSumFac.hpp
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

#include <LibUtilities/Foundations/Basis.h>

#include "Common/OperatorHelper.hpp"
#include "ElmtOps/OperatorLinAdvDiffReaction.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "ElmtOps/Helmholtz/HelmholtzSerialAVXSumFacKernels.hpp"
#include "ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Matrix-free implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorLinAdvDiffReactionImpl : public OperatorLinAdvDiffReaction<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

    Array<OneD, TData> Null1DArray;

public:
    OperatorLinAdvDiffReactionImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinAdvDiffReaction<TData>(expansionList)
    {
        // Initialise jacobian with paddings.
        m_physBlockAttributes = GetBlockAttributes<TData>(
            FieldState::Phys, expansionList, simd_t::width);
        m_jac = SetJacobian<MemSpace, TData>(
            expansionList, m_physBlockAttributes, ExecSpace::alignment);
        m_df = SetDerivativeFactor<MemSpace, TData>(
            expansionList, m_physBlockAttributes, ExecSpace::alignment);

        // Initialize the basis data.
        m_Bmap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eBasis, simd_t::alignment);
        m_weightMap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eWeights, simd_t::alignment);
        // Initialize the derivative matrix.
        m_derivativeMap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eDerivative, simd_t::alignment);

        // Initialize the BD data.
        m_dbasisMap = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eBasisDerivative, simd_t::alignment);

        // Initialize the collapsed to cartesian geometric factors.
        m_fac0 = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eHalfMultOnePlusZero, simd_t::alignment);
        m_fac1 = GetBasisData<MemSpace, NekDouble, simd_t>(
            expansionList, eTwoOverOneMinusZero, simd_t::alignment);

        auto nCoord = this->m_expansionList->GetCoordim(0);
        m_diffCoeff = std::vector<TData>(nCoord * (nCoord + 1) / 2, 0.0);

        // Set up temprary solution.
        m_diffCoeff[0] = 1.0; // D00
        if (nCoord >= 2)
        {
            m_diffCoeff[2] = 1.0; // D11
            if (nCoord == 3)
            {
                m_diffCoeff[5] = 1.0; // D22
            }
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Initialize index.
        size_t exp_idx = 0;

        m_nComps = in.GetNumComponents();
        ASSERTL1(m_nComps == out.GetNumComponents(),
                 "Number of input and output components differ");

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

    // className - for OperatorFactory
    static std::string className;

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
        // Check alignment.
        WARNINGL1(inblock.GetAlignment() == simd_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(outblock.GetAlignment() == simd_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

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
    unsigned int m_blk;
    size_t m_nComps;

    std::vector<BlockAttributes> m_physBlockAttributes;

    LocalRegions::ExpansionSharedPtr m_expPtr;

    Field<TData, FieldState::Phys> m_advVel;

    std::vector<MemoryRegion<TData>> m_jac;
    std::vector<MemoryRegion<TData>> m_df;

    BasisDataMap<simd_t> m_Bmap;
    BasisDataMap<simd_t> m_dbasisMap;
    BasisDataMap<simd_t> m_derivativeMap;
    BasisDataMap<simd_t> m_fac0;
    BasisDataMap<simd_t> m_fac1;
    BasisDataMap<simd_t> m_weightMap;

    std::vector<TData> m_diffCoeff;

    std::vector<TData> NullTDataVector;

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

        const auto nqTot = nq0;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0);

        constexpr auto ndf = 1;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto B0 = m_Bmap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto DB0 =
            m_dbasisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Allocate workspace.
        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        // advVel pointers
        auto &advVelBlock = m_advVel.GetBlocks()[m_blk];

        unsigned int advVel_interleave_width = advVelBlock.GetInterleaveWidth();
        advVelBlock.template SetInterleaveWidth<TData>(simd_t::width);

        auto advVel = (advVel_interleave_width == simd_t::width)
                          ? advVelBlock.template GetPtr<MemSpace, ReadOnly>()
                          : advVelBlock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();

        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);

        // loop over componnets
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr    = jacPtr_init;
            auto dfPtr     = dfPtr_init;
            auto advVelPtr = advVelPtr_init;

            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                    if (nc == 0) // only need to interlace adv vel once
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)advVelPtr);
                    }
                }

                // Step 1: BwdTrans.
                BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, B0, tmpIn, bwd);

                // Step 2: Take derivatives in collapsed coordinate space.
                PhysDerivTensor1DKernel(nq0, bwdvec, D0, deriv0);

                // Step 3: add Advect solution to  this->m_lambda * bwd
                AddAdvectionSegKernel<DEFORMED>(nq0, advVelPtr, dfPtr,
                                                deriv0vec, bwd, this->m_lambda);

                // Step 4: Inner product for mass matrix operation.
                IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                    nm0, nq0, bwdvec, B0, W0, jacPtr, tmpOut);

                // Step 5: Apply diffusion coefficiets.
                DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                    nq0, true, this->m_diffCoeff, false, NullTDataVector, dfPtr,
                    deriv0);

                // Step 6: Apply Laplacian metrics & inner product.
                IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                    nm0, nq0, deriv0vec, DB0, W0, jacPtr, tmpOut);

                // Increment pointers.
                dfPtr += dfSize * ndf;
                jacPtr += dfSize;
                advVelPtr += nqTot;
                tmpIn += nmTot;
                tmpOut += nmTot * simd_t::width;
            }
        }
        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nqTot = nq0;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0);

        constexpr auto ndf = 1;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey()};
        auto B0 = m_Bmap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto DB0 =
            m_dbasisMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Allocate workspace.
        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        // advVel pointers
        auto &advVelBlock                    = m_advVel.GetBlocks()[m_blk];
        unsigned int advVel_interleave_width = advVelBlock.GetInterleaveWidth();
        advVelBlock.template SetInterleaveWidth<TData>(simd_t::width);

        auto advVel = (advVel_interleave_width == simd_t::width)
                          ? advVelBlock.template GetPtr<MemSpace, ReadOnly>()
                          : advVelBlock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();

        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);

        // loop over componnets
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr    = jacPtr_init;
            auto dfPtr     = dfPtr_init;
            auto advVelPtr = advVelPtr_init;

            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                    if (nc == 0) // only need to interlace adv vel once
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)advVelPtr);
                    }
                }

                // Step 1: BwdTrans.
                BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, B0, tmpIn, bwd);

                // Step 2: Take derivatives in collapsed coordinate space.
                PhysDerivTensor1DKernel(nq0, bwdvec, D0, deriv0);

                // Step 3: add Advect solution to  this->m_lambda * bwd
                AddAdvectionSegKernel<DEFORMED>(nq0, advVelPtr, dfPtr,
                                                deriv0vec, bwd, this->m_lambda);

                // Step 4: Inner product for mass matrix operation.
                IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                    nm0, nq0, bwdvec, B0, W0, jacPtr, tmpOut);

                // Step 5: Apply diffusion coefficiets.
                DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                    nq0, true, this->m_diffCoeff, false, NullTDataVector, dfPtr,
                    deriv0);

                // Step 6: Apply Laplacian metrics & inner product.
                IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                    nm0, nq0, deriv0vec, DB0, W0, jacPtr, tmpOut);

                // Increment pointers.
                dfPtr += dfSize * ndf;
                jacPtr += dfSize;
                advVelPtr += nqTot;
                tmpIn += nmTot;
                tmpOut += nmTot * simd_t::width;
            }
        }
        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
    }

    // Non-size based Operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nm1 = m_expPtr->GetBasisNumModes(1);

        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);

        const auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

        constexpr auto ndf = 4;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 = m_Bmap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 = m_Bmap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        const simd_t *h0 = nullptr, *h1 = nullptr;

        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            h0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            h1 = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        auto diffderiv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv0);

        auto diffderiv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv1);

        // advVel pointers
        auto &advVelBlock = m_advVel.GetBlocks()[m_blk];

        unsigned int advVel_interleave_width = advVelBlock.GetInterleaveWidth();
        advVelBlock.template SetInterleaveWidth<TData>(simd_t::width);
        auto advVelOffset = advVelBlock.GetNumElmtGroups() * nqTot;
        auto advVel       = (advVel_interleave_width == simd_t::width)
                                ? advVelBlock.template GetPtr<MemSpace, ReadOnly>()
                                : advVelBlock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);

        // loop over componnets
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr    = jacPtr_init;
            auto dfPtr     = dfPtr_init;
            auto advVelPtr = advVelPtr_init;

            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                    // only need to interlace adv vel once but need to do all
                    // components
                    if (nc == 0)
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)advVelPtr);
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)(advVelPtr + advVelOffset));
                    }
                }

                // Step 1: BwdTrans.
                BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, isModified, B0,
                                             B1, wsp0, tmpIn, bwd);

                // Step 2 + 3: Get tensor derivative (deriv0, deriv1)  and apply
                // diffusion coeff (diffderiv0, diffderiv1)
                TensorDerivWithDiffuCoeff2DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, true, this->m_diffCoeff, false, NullTDataVector,
                    NullTDataVector, NullTDataVector, bwdvec, D0, D1, dfPtr, h0,
                    h1, diffderiv0, diffderiv1, deriv0, deriv1);

                // Step 4 evaluate advection term and add to bwd * lambda
                AddAdvection2DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, h0, h1, advVelPtr, advVelPtr + advVelOffset,
                    dfPtr, deriv0vec, deriv1vec, bwd, this->m_lambda);

                // Step 5: apply WJ, derivative and sum up
                SumDerivTensor2DKernel<DEFORMED, simd_t>(
                    nq0, nq1, diffderiv0vec, diffderiv1vec, W0, W1, jacPtr, D0,
                    D1, bwd, 1.0);
                // Step 6 : inner product without WJ
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, isModified, bwdvec, B0, B1, wsp0,
                    tmpOut);

                // Increment pointers.
                dfPtr += dfSize * ndf;
                jacPtr += dfSize;
                advVelPtr += nqTot;
                tmpIn += nmTot;
                tmpOut += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nq0, int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

        constexpr auto ndf = 4;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey()};
        auto B0 = m_Bmap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 = m_Bmap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto D0 =
            m_derivativeMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto D1 =
            m_derivativeMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto W0 =
            m_weightMap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto W1 =
            m_weightMap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        const simd_t *h0 = nullptr, *h1 = nullptr;

        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            h0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            h1 = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        auto diffderiv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv0);

        auto diffderiv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv1);

        // advVel pointers
        auto &advVelBlock = m_advVel.GetBlocks()[m_blk];

        unsigned int advVel_interleave_width = advVelBlock.GetInterleaveWidth();
        advVelBlock.template SetInterleaveWidth<TData>(simd_t::width);
        auto advVelOffset = advVelBlock.GetNumElmtGroups() * nqTot;

        auto advVel = (advVel_interleave_width == simd_t::width)
                          ? advVelBlock.template GetPtr<MemSpace, ReadOnly>()
                          : advVelBlock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);

        // loop over componnets
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr    = jacPtr_init;
            auto dfPtr     = dfPtr_init;
            auto advVelPtr = advVelPtr_init;

            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                    // only need to interlace adv vel once but need to do all
                    // components
                    if (nc == 0)
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)advVelPtr);
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)(advVelPtr + advVelOffset));
                    }
                }

                // Step 1: BwdTrans.
                BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, isModified, B0,
                                             B1, wsp0, tmpIn, bwd);

                // Step 2 + 3: Get tensor derivative (deriv0, deriv1)  and apply
                // diffusion coeff (diffderiv0, diffderiv1)
                TensorDerivWithDiffuCoeff2DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, true, this->m_diffCoeff, false, NullTDataVector,
                    NullTDataVector, NullTDataVector, bwdvec, D0, D1, dfPtr, h0,
                    h1, diffderiv0, diffderiv1, deriv0, deriv1);

                // Step 4 evaluate advection term and add to bwd * lambda
                AddAdvection2DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, h0, h1, advVelPtr, advVelPtr + advVelOffset,
                    dfPtr, deriv0vec, deriv1vec, bwd, this->m_lambda);

                // Step 5: apply WJ, derivative and sum up
                SumDerivTensor2DKernel<DEFORMED, simd_t>(
                    nq0, nq1, diffderiv0vec, diffderiv1vec, W0, W1, jacPtr, D0,
                    D1, bwd, 1.0);

                // Step 6 : inner product without WJ
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, isModified, bwdvec, B0, B1, wsp0,
                    tmpOut);

                // Increment pointers.
                dfPtr += dfSize * ndf;
                jacPtr += dfSize;
                advVelPtr += nqTot;
                tmpIn += nmTot;
                tmpOut += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_expPtr->GetBasisNumModes(0);
        const auto nm1 = m_expPtr->GetBasisNumModes(1);
        const auto nm2 = m_expPtr->GetBasisNumModes(2);

        const auto nq0 = m_expPtr->GetNumPoints(0);
        const auto nq1 = m_expPtr->GetNumPoints(1);
        const auto nq2 = m_expPtr->GetNumPoints(2);

        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        constexpr auto ndf = 9;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey(),
            m_expPtr->GetBasis(2)->GetBasisKey()};
        auto B0 = m_Bmap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 = m_Bmap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto B2 = m_Bmap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
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

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        // get geometric factors
        const simd_t *h0 = nullptr, *h1 = nullptr, *h2 = nullptr, *h3 = nullptr;
        if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
        {
            h0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            h1 = m_fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            h2 = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            h3 = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
        {
            h0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            h1 = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
        {
            h0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            h1 = m_fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            h2 = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);
        auto deriv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv2);

        auto diffderiv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv0);

        auto diffderiv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv1);

        auto diffderiv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv2);

        // advVel pointers
        auto &advVelBlock = m_advVel.GetBlocks()[m_blk];

        unsigned int advVel_interleave_width = advVelBlock.GetInterleaveWidth();
        advVelBlock.template SetInterleaveWidth<TData>(simd_t::width);
        auto advVelOffset = advVelBlock.GetNumElmtGroups() * nqTot;

        auto advVel = (advVel_interleave_width == simd_t::width)
                          ? advVelBlock.template GetPtr<MemSpace, ReadOnly>()
                          : advVelBlock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);

        // loop over componnets
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr    = jacPtr_init;
            auto dfPtr     = dfPtr_init;
            auto advVelPtr = advVelPtr_init;

            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                    if (nc == 0) // only need to interlace adv vel once
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)advVelPtr);
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)(advVelPtr + advVelOffset));
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)(advVelPtr + 2 * advVelOffset));
                    }
                }

                // Step 1: BwdTrans.
                BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2,
                                             isModified, B0, B1, B2, wsp0, wsp1,
                                             tmpIn, bwd);
                // Step 2 + 3 : Get tensor derivative and apply diffusion
                // coeff
                TensorDerivWithDiffuCoeff3DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, nq2, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector, bwdvec,
                    D0, D1, D2, dfPtr, h0, h1, h2, h3, diffderiv0, diffderiv1,
                    diffderiv2, deriv0, deriv1, deriv2);
                // Step 4 evaluate advection term and add to bwd * lambda
                AddAdvection3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, h0, h1, h2, h3, advVelPtr,
                    advVelPtr + advVelOffset, advVelPtr + 2 * advVelOffset,
                    dfPtr, deriv0vec, deriv1vec, deriv2vec, bwd,
                    this->m_lambda);
                // Step 5: apply WJ, derivative and sum up
                SumDerivTensor3DKernel<DEFORMED, simd_t>(
                    nq0, nq1, nq2, diffderiv0vec, diffderiv1vec, diffderiv2vec,
                    W0, W1, W2, jacPtr, D0, D1, D2, bwd, 1.0);
                // Step 6 : inner product without WJ
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, isModified, bwdvec, B0, B1,
                    B2, wsp0, wsp1, wsp2, tmpOut);

                // Increment pointers.
                dfPtr += dfSize * ndf;
                jacPtr += dfSize;
                advVelPtr += nqTot;
                tmpIn += nmTot;
                tmpOut += nmTot * simd_t::width;
            }
        }
        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv2, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv2, std::align_val_t(simd_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nm2, int nq0, int nq1, int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        constexpr auto ndf = 9;
        auto dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Flag for collapsed coordinate correction.
        const bool isModified =
            (m_expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Fetch basis and weight data.
        std::vector<LibUtilities::BasisKey> basisKeys{
            m_expPtr->GetBasis(0)->GetBasisKey(),
            m_expPtr->GetBasis(1)->GetBasisKey(),
            m_expPtr->GetBasis(2)->GetBasisKey()};
        auto B0 = m_Bmap[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        auto B1 = m_Bmap[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        auto B2 = m_Bmap[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
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

        // Fetch Jacobian and deriv factors.
        auto jacPtr_init = reinterpret_cast<const simd_t *>(
            m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>());
        auto dfPtr_init = reinterpret_cast<const simd_t *>(
            m_df[m_blk].template GetPtr<MemSpace, ReadOnly>());

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        // get geometric factors
        const simd_t *h0 = nullptr, *h1 = nullptr, *h2 = nullptr, *h3 = nullptr;
        if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron)
        {
            h0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            h1 = m_fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            h2 = m_fac1[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            h3 = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::ePrism)
        {
            h0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            h1 = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::ePyramid)
        {
            h0 = m_fac0[basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            h1 = m_fac0[basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            h2 = m_fac1[basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);
        auto deriv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv2);

        auto diffderiv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv0);

        auto diffderiv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv1);

        auto diffderiv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv2);

        // advVel pointers
        auto &advVelBlock = m_advVel.GetBlocks()[m_blk];

        unsigned int advVel_interleave_width = advVelBlock.GetInterleaveWidth();
        advVelBlock.template SetInterleaveWidth<TData>(simd_t::width);
        auto advVelOffset = advVelBlock.GetNumElmtGroups() * nqTot;
        auto advVel       = (advVel_interleave_width == simd_t::width)
                                ? advVelBlock.template GetPtr<MemSpace, ReadOnly>()
                                : advVelBlock.template GetPtr<MemSpace, ReadWrite>();

        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);

        // loop over componnets
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            auto jacPtr    = jacPtr_init;
            auto dfPtr     = dfPtr_init;
            auto advVelPtr = advVelPtr_init;

            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                    if (nc == 0) // only need to interlace adv vel once
                    {
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)advVelPtr);
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)(advVelPtr + advVelOffset));
                        ReshapeStorage<ExecSpace, simd_t::width>(
                            advVel_interleave_width, chunkSize, nqTot,
                            (TData *)(advVelPtr + 2 * advVelOffset));
                    }
                }

                // Step 1: BwdTrans.
                BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2,
                                             isModified, B0, B1, B2, wsp0, wsp1,
                                             tmpIn, bwd);
                // Step 2 + 3 : Get tensor derivative and apply diffusion
                // coeff
                TensorDerivWithDiffuCoeff3DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, nq2, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector, bwdvec,
                    D0, D1, D2, dfPtr, h0, h1, h2, h3, diffderiv0, diffderiv1,
                    diffderiv2, deriv0, deriv1, deriv2);
                // Step 4 evaluate advection term and add to bwd * lambda
                AddAdvection3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, h0, h1, h2, h3, advVelPtr,
                    advVelPtr + advVelOffset, advVelPtr + 2 * advVelOffset,
                    dfPtr, deriv0vec, deriv1vec, deriv2vec, bwd,
                    this->m_lambda);
                // Step 5: apply WJ, derivative and sum up
                SumDerivTensor3DKernel<DEFORMED, simd_t>(
                    nq0, nq1, nq2, diffderiv0vec, diffderiv1vec, diffderiv2vec,
                    W0, W1, W2, jacPtr, D0, D1, D2, bwd, 1.0);
                // Step 6 : inner product without WJ
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, isModified, bwdvec, B0, B1,
                    B2, wsp0, wsp1, wsp2, tmpOut);

                // Increment pointers.
                dfPtr += dfSize * ndf;
                jacPtr += dfSize;
                advVelPtr += nqTot;
                tmpIn += nmTot;
                tmpOut += nmTot * simd_t::width;
            }
        }
        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv2, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv2, std::align_val_t(simd_t::alignment));
    }
};

} // namespace Nektar::Operators::detail
