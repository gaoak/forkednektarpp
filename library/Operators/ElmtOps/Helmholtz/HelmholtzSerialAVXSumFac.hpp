///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzSerialAVXSumFac.hpp
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

#include "Common/OperatorHelper.hpp"
#include "ElmtOps/OperatorBwdTrans.hpp"
#include "ElmtOps/OperatorHelmholtz.hpp"
#include "ElmtOps/OperatorIProductWRTBase.hpp"
#include "ElmtOps/OperatorIProductWRTDerivBase.hpp"
#include "ElmtOps/OperatorPhysDeriv.hpp"
#include <LibUtilities/BasicUtils/NekInline.hpp>

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/Basis.h>

namespace Nektar::Operators::detail
{

// Include kernel functions after simd_t definition
#include "ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp"
#include "ElmtOps/Helmholtz/HelmholtzSerialAVXSumFacKernels.hpp"
#include "ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"
#include "ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

// Matrix-free implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::AVX>::value &&
               std::is_same<Implementation, Operators::SumFac>::value)>::type>
class OperatorHelmholtzImpl : public OperatorHelmholtz<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same<ExecSpace, NektarSpaces::AVX>::value,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtz<TData>(expansionList)
    {
        // Initialise jacobian with paddings
        auto locblocks =
            GetBlockAttributes(FieldState::Phys, expansionList, simd_t::width);

        size_t gFacSize =
            Operator<TData>::GetGeometricFactorSize(locblocks, simd_t::width);
        auto jac =
            Operator<TData>::SetJacobian(gFacSize, locblocks, simd_t::width);
        m_jac = MemoryRegion<TData>::template fromVector<MemSpace, TData>(
            *jac, ExecSpace::alignment);
        auto derivFac = Operator<TData>::SetDerivativeFactor(
            gFacSize, locblocks, simd_t::width);
        m_df = MemoryRegion<TData>::template fromVector<MemSpace, TData>(
            *derivFac, simd_t::alignment);

        // Initialize the basis data.
        m_Bmap = GetBasisData<MemSpace, TData, simd_t>(expansionList, eBasis,
                                                       simd_t::alignment);
        m_Wmap = GetBasisData<MemSpace, TData, simd_t>(expansionList, eWeights,
                                                       simd_t::alignment);
        // Initialize the derivative matrix.
        m_Dmap = GetBasisData<MemSpace, TData, simd_t>(
            expansionList, eDerivative, simd_t::alignment);

        // Initialize the BD data
        m_BDmap = GetBasisData<MemSpace, TData, simd_t>(
            expansionList, eBasisDerivative, simd_t::alignment);

        // Initialize the Z data.
        m_Zmap = GetBasisData<MemSpace, TData, simd_t>(expansionList, eZeros,
                                                       simd_t::alignment);

        auto nCoord = this->m_expansionList->GetCoordim(0);
        m_diffCoeff = Array<OneD, TData>(nCoord * (nCoord + 1) / 2, 0.0);

        // set up temprary solution
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
        // check alignment
        WARNINGL1(in.GetAlignment() == simd_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(out.GetAlignment() == simd_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

        // Reshape into simd_t::width. If the Field is already
        // interleaved, this method returns.
        in.template ReshapeStorage<ExecSpace, simd_t::width>();
        out.template ReshapeStorage<ExecSpace, simd_t::width>();

        const auto *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr      = out.template GetPtr<MemSpace, ReadWrite>();

        //---debug-----
        // std::cout << "Print Vectorized Jacobian:" << std::endl;
        // size_t jac_idx = 0;

        m_exp_idx = 0; // accumulates over blocks, used in operatorND()
        m_jac_idx = 0; // accumulates over blocks, accessed in operatorND()

        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            const auto &inblock  = in.GetBlocks()[block_idx];
            const auto &outblock = out.GetBlocks()[block_idx];
            const auto nElmts    = inblock.num_elements;

            // Determine shape and type of the element.
            const auto expPtr    = this->m_expansionList->GetExp(m_exp_idx);
            const auto nqTot     = expPtr->GetTotPoints();
            const auto shapeType = expPtr->DetShapeType();
            const auto dimension = expPtr->GetShapeDimension();
            const auto deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;

            m_nElmtGroup = inblock.GetNumElmtGroups(simd_t::width);

            // Fetch basis key for the current element type.
            m_basisKeys.clear();
            for (size_t d = 0; d < dimension; ++d)
            {
                m_basisKeys.push_back(expPtr->GetBasis(d)->GetBasisKey());
            }

            switch (shapeType)
            {
                    // Segment
                case LibUtilities::Seg:
                {
                    SegBlock(inPtr, outPtr);
                    break;
                }
                // Quads
                case LibUtilities::Quad:
                {
                    QuadBlock(inPtr, outPtr);
                    break;
                }
                // Triangles
                case LibUtilities::Tri:
                {
                    TriBlock(inPtr, outPtr);
                    break;
                }
                // Hexes
                case LibUtilities::Hex:
                {
                    HexBlock(inPtr, outPtr);
                    break;
                }
                // Tet
                case LibUtilities::Tet:
                {
                    TetBlock(inPtr, outPtr);
                    break;
                }
                // Pyr
                case LibUtilities::Pyr:
                {
                    PyrBlock(inPtr, outPtr);
                    break;
                }
                // Prism
                case LibUtilities::Prism:
                {
                    PrismBlock(inPtr, outPtr);
                    break;
                }
                default:
                    std::cout << "shapetype not implemented" << std::endl;
            }
            // #include "Operators/Common/SwitchLevel2Deformed.h"

            inPtr += inblock.block_size;
            outPtr += outblock.block_size;
            m_exp_idx += nElmts;

            if (deformed) // update m_jac_idx globally
            {
                m_jac_idx += nqTot * m_nElmtGroup * simd_t::width;
            }
            else
            {
                m_jac_idx += m_nElmtGroup * simd_t::width;
            }
        }
    }

    // className - for OperatorFactory
    static std::string className;

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorHelmholtzImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    int m_nElmtGroup, m_jac_idx, m_exp_idx;

    MemoryRegion<TData> m_jac;
    MemoryRegion<TData> m_df;

    BasisDataMap<simd_t> m_Bmap;
    BasisDataMap<simd_t> m_BDmap;
    BasisDataMap<simd_t> m_Dmap;
    BasisDataMap<simd_t> m_Zmap;
    BasisDataMap<simd_t> m_Wmap;

    std::vector<LibUtilities::BasisKey> m_basisKeys;
    Array<OneD, TData> m_diffCoeff;

    void SegBlock(const TData *inPtr, TData *outPtr);

    void TriBlock(const TData *inPtr, TData *outPtr);

    void QuadBlock(const TData *inPtr, TData *outPtr);

    void HexBlock(const TData *inPtr, TData *outPtr);

    void PrismBlock(const TData *inPtr, TData *outPtr);

    void PyrBlock(const TData *inPtr, TData *outPtr);

    void TetBlock(const TData *inPtr, TData *outPtr);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nq0 = expPtr->GetNumPoints(0);

        constexpr auto ndf = 1;

        const auto nqTot = nq0;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0);

        TData *bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *bwdvec =
            reinterpret_cast<typename simd_t::vectorType *>(bwd);
        TData *deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const simd_t *dfPtr = reinterpret_cast<const simd_t *>(
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]));
        const simd_t *jacPtr = reinterpret_cast<const simd_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));

        // Get basis data pointers
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto D0 =
            m_Dmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // Step 1: BwdTrans
            BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, B0, tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nq0, bwdvec, B0, W0, jacPtr, tmpOut, this->m_lambda);
            // Step 3: take derivatives in collapsed coordinate space
            PhysDerivTensor1DKernel(nq0, bwdvec, D0, deriv0);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                nq0, true, this->m_diffCoeff, false, NullNekDouble1DArray,
                dfPtr, deriv0);
            // Step 5: Apply Laplacian metrics & inner product
            IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nq0, deriv0vec, BD0, W0, jacPtr, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * simd_t::width;
        }

        // free aligned memory
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nq0>
    void Operator1D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        constexpr auto ndf = 1;

        const auto nqTot = nq0;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0);

        TData *bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *bwdvec =
            reinterpret_cast<typename simd_t::vectorType *>(bwd);
        TData *deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const simd_t *dfPtr = reinterpret_cast<const simd_t *>(
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]));
        const simd_t *jacPtr = reinterpret_cast<const simd_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));

        // Get basis data pointers
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto D0 =
            m_Dmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // Step 1: BwdTrans
            BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, B0, tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nq0, bwdvec, B0, W0, jacPtr, tmpOut, this->m_lambda);
            // Step 3: take derivatives in collapsed coordinate space
            PhysDerivTensor1DKernel(nq0, bwdvec, D0, deriv0);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                nq0, true, this->m_diffCoeff, false, NullNekDouble1DArray,
                dfPtr, deriv0);
            // Step 5: Apply Laplacian metrics & inner product
            IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nq0, deriv0vec, BD0, W0, jacPtr, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * simd_t::width;
        }

        // free aligned memory
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
    }

    // Non-size based Operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nm1 = expPtr->GetBasisNumModes(1);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);

        constexpr auto ndf = 4;

        const auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

        const bool isModified =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> m_h0, m_h1;

        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            const auto Z0 =
                m_Zmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            const auto Z1 =
                m_Zmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            GetHelmholtz2DHalfSpace<SHAPE_TYPE, simd_t>(nq0, nq1, Z0, Z1, m_h0,
                                                        m_h1);
        }

        // allocate workspace
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        TData *bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *bwdvec =
            reinterpret_cast<typename simd_t::vectorType *>(bwd);
        TData *deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        TData *deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const simd_t *dfPtr = reinterpret_cast<const simd_t *>(
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]));
        const simd_t *jacPtr = reinterpret_cast<const simd_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));

        // Get basis data pointers
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto B1 =
            m_Bmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto D0 =
            m_Dmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto D1 =
            m_Dmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD1 =
            m_BDmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W1 =
            m_Wmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // Step 1: BwdTrans
            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, isModified, B0, B1,
                                         wsp0, tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct2DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nm1, nq0, nq1, isModified, bwdvec, B0, B1, W0, W1, jacPtr,
                wsp0, tmpOut, this->m_lambda);
            // Step 3: take derivatives in collapsed coordinate space
            PhysDerivTensor2DKernel(nq0, nq1, bwdvec, D0, D1, deriv0, deriv1);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeff2DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                nq0, nq1, true, this->m_diffCoeff, false, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray, dfPtr, m_h0, m_h1,
                deriv0, deriv1);
            // Step 5: Apply Laplacian metrics & inner product
            IProduct2DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nq0, nq1, isModified, deriv0vec, BD0, B1, W0, W1,
                jacPtr, wsp0, tmpOut);
            IProduct2DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nq0, nq1, isModified, deriv1vec, B0, BD1, W0, W1,
                jacPtr, wsp0, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * simd_t::width;
        }

        // free aligned memory
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nq0, int nq1>
    void Operator2D(const TData *input, TData *output)
    {
        constexpr auto nqTot = nq0 * nq1;
        constexpr auto ndf   = 4;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);
        const bool isModified =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> m_h0, m_h1;

        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            const auto Z0 =
                m_Zmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            const auto Z1 =
                m_Zmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            GetHelmholtz2DHalfSpace<SHAPE_TYPE, simd_t>(nq0, nq1, Z0, Z1, m_h0,
                                                        m_h1);
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);
        alignas(simd_t::alignment) TData bwd[nqTot * simd_t::width];
        typename simd_t::vectorType *bwdvec =
            reinterpret_cast<typename simd_t::vectorType *>(bwd);
        alignas(simd_t::alignment) TData deriv0[nqTot * simd_t::width];
        typename simd_t::vectorType *deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        alignas(simd_t::alignment) TData deriv1[nqTot * simd_t::width];
        typename simd_t::vectorType *deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const simd_t *dfPtr = reinterpret_cast<const simd_t *>(
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]));
        const simd_t *jacPtr = reinterpret_cast<const simd_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));

        // Get basis data pointers
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto B1 =
            m_Bmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto D0 =
            m_Dmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto D1 =
            m_Dmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD1 =
            m_BDmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W1 =
            m_Wmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // load_interleave(inPtr, nqTot, tmpIn);
            // Step 1: BwdTrans
            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, isModified, B0, B1,
                                         wsp0, tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct2DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nm1, nq0, nq1, isModified, bwdvec, B0, B1, W0, W1, jacPtr,
                wsp0, tmpOut, this->m_lambda);
            // Step 3: take derivatives in collapsed coordinate space
            PhysDerivTensor2DKernel(nq0, nq1, bwdvec, D0, D1, deriv0, deriv1);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeff2DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                nq0, nq1, true, this->m_diffCoeff, false, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray, dfPtr, m_h0, m_h1,
                deriv0, deriv1);
            // Step 4: Apply Laplacian metrics & inner product
            IProduct2DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nq0, nq1, isModified, deriv0vec, BD0, B1, W0, W1,
                jacPtr, wsp0, tmpOut);
            IProduct2DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nq0, nq1, isModified, deriv1vec, B0, BD1, W0, W1,
                jacPtr, wsp0, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * simd_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nm1 = expPtr->GetBasisNumModes(1);
        const auto nm2 = expPtr->GetBasisNumModes(2);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);
        const auto nq2 = expPtr->GetNumPoints(2);

        constexpr auto ndf = 9;
        const auto nqTot   = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        const bool isModified =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> m_h0, m_h1, m_h2, m_h3;

        if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron ||
                      SHAPE_TYPE == LibUtilities::ePrism ||
                      SHAPE_TYPE == LibUtilities::ePyramid)
        {
            const auto Z0 =
                m_Zmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            const auto Z1 =
                m_Zmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            const auto Z2 =
                m_Zmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
            GetHelmholtz3DHalfSpace<SHAPE_TYPE, simd_t>(
                nq0, nq1, nq2, Z0, Z1, Z2, m_h0, m_h1, m_h2, m_h3);
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        TData *bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *bwdvec =
            reinterpret_cast<typename simd_t::vectorType *>(bwd);

        TData *deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        TData *deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        TData *deriv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        typename simd_t::vectorType *deriv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv2);

        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const simd_t *dfPtr = reinterpret_cast<const simd_t *>(
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]));
        const simd_t *jacPtr = reinterpret_cast<const simd_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));

        // Get basis data pointers
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto B1 =
            m_Bmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto B2 =
            m_Bmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto D0 =
            m_Dmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto D1 =
            m_Dmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto D2 =
            m_Dmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD1 =
            m_BDmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD2 =
            m_BDmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W1 =
            m_Wmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto W2 =
            m_Wmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // load_interleave(inPtr, nqTot, tmpIn);
            // Step 1: BwdTrans
            BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2,
                                         isModified, B0, B1, B2, wsp0, wsp1,
                                         tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct3DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, isModified, bwdvec, B0, B1, B2,
                W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut, this->m_lambda);
            // Step 3: take derivatives in standard space
            PhysDerivTensor3DKernel(nq0, nq1, nq2, bwdvec, D0, D1, D2, deriv0,
                                    deriv1, deriv2);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeff3DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                nq0, nq1, nq2, true, this->m_diffCoeff, false,
                NullNekDouble1DArray, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray, dfPtr, m_h0, m_h1,
                m_h2, m_h3, deriv0, deriv1, deriv2);
            // Step 5: Apply Laplacian metrics & inner product
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, isModified, deriv0vec, BD0, B1,
                B2, W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, isModified, deriv1vec, B0, BD1,
                B2, W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, isModified, deriv2vec, B0, B1,
                BD2, W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * simd_t::width;
        }

        // free aligned memory
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv2, std::align_val_t(simd_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nm2, int nq0, int nq1, int nq2>
    void Operator3D(const TData *input, TData *output)
    {
        constexpr auto ndf   = 9;
        constexpr auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const bool isModified =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> m_h0, m_h1, m_h2, m_h3;

        if constexpr (SHAPE_TYPE == LibUtilities::eTetrahedron ||
                      SHAPE_TYPE == LibUtilities::ePrism ||
                      SHAPE_TYPE == LibUtilities::ePyramid)
        {
            const auto Z0 =
                m_Zmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            const auto Z1 =
                m_Zmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            const auto Z2 =
                m_Zmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
            GetHelmholtz3DHalfSpace<SHAPE_TYPE, simd_t>(
                nq0, nq1, nq2, Z0, Z1, Z2, m_h0, m_h1, m_h2, m_h3);
        }

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        alignas(simd_t::alignment) TData bwd[nqTot * simd_t::width];
        typename simd_t::vectorType *bwdvec =
            reinterpret_cast<typename simd_t::vectorType *>(bwd);

        alignas(simd_t::alignment) TData deriv0[nqTot * simd_t::width];
        typename simd_t::vectorType *deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        alignas(simd_t::alignment) TData deriv1[nqTot * simd_t::width];
        typename simd_t::vectorType *deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        alignas(simd_t::alignment) TData deriv2[nqTot * simd_t::width];
        typename simd_t::vectorType *deriv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv2);

        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const simd_t *dfPtr = reinterpret_cast<const simd_t *>(
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]));
        const simd_t *jacPtr = reinterpret_cast<const simd_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));

        // Get basis data pointers
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto B1 =
            m_Bmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto B2 =
            m_Bmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto D0 =
            m_Dmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto D1 =
            m_Dmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto D2 =
            m_Dmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD1 =
            m_BDmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto BD2 =
            m_BDmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W1 =
            m_Wmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto W2 =
            m_Wmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // load_interleave(inPtr, nqTot, tmpIn);
            // Step 1: BwdTrans
            BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2,
                                         isModified, B0, B1, B2, wsp0, wsp1,
                                         tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct3DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, isModified, bwdvec, B0, B1, B2,
                W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut, this->m_lambda);
            // Step 3: take derivatives in standard space
            PhysDerivTensor3DKernel(nq0, nq1, nq2, bwdvec, D0, D1, D2, deriv0,
                                    deriv1, deriv2);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeff3DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                nq0, nq1, nq2, true, this->m_diffCoeff, false,
                NullNekDouble1DArray, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray, dfPtr, m_h0, m_h1,
                m_h2, m_h3, deriv0, deriv1, deriv2);
            // Step 5: Apply Laplacian metrics & inner product
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, isModified, deriv0vec, BD0, B1,
                B2, W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, isModified, deriv1vec, B0, BD1,
                B2, W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, isModified, deriv2vec, B0, B1,
                BD2, W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * simd_t::width;
        }
    }
};

} // namespace Nektar::Operators::detail
