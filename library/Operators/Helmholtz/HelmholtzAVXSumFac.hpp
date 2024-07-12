///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzAVXSumFac.hpp
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

#include "Operators/Helmholtz/HelmholtzImplBase.hpp"

#include "HelmholtzAVXSumFacKernels.hpp"
#include "Operators/BwdTrans/BwdTransAVXSumFacKernels.hpp"
#include "Operators/IProductWRTBase/IProductWRTBaseAVXSumFacKernels.hpp"
#include "Operators/PhysDeriv/PhysDerivAVXSumFacKernels.hpp"

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/Basis.h>
#include <LibUtilities/SimdLib/tinysimd.hpp>

namespace Nektar::Operators::detail
{

typedef std::vector<vec_t, tinysimd::allocator<vec_t>> VecVec_t;

// Matrix-free implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::AVX>::value &&
              std::is_same<Implementation, Operators::SumFac>::value>::type>
class OperatorHelmholtzImpl
    : public OperatorHelmholtzImplBase<ExecSpace, Implementation, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmholtzImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtzImplBase<ExecSpace, Implementation, TData>(
              expansionList)
    {
        // Initialise jacobian with paddings
        auto blocks =
            GetBlockAttributes(FieldState::Phys, expansionList, vec_t::width);

        size_t jacSize = Operator<TData>::GetGeometricFactorSize(blocks);
        auto jac       = Operator<TData>::SetJacobian(jacSize, blocks);
        m_jac = MemoryRegion<vec_t>::template fromVector<MemSpace, vec_t>(
            *jac, vec_t::alignment);
        auto derivFac = Operator<TData>::SetDerivativeFactor(jacSize, blocks);
        m_df = MemoryRegion<vec_t>::template fromVector<MemSpace, vec_t>(
            *derivFac, vec_t::alignment);

        // Initialize the basis data.
        m_Bmap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_BASIS_DATA, vec_t::alignment);
        m_Wmap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_WEIGHT_DATA, vec_t::alignment);
        // Initialize the derivative matrix.
        m_Dmap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_DERIVATIVE_DATA, vec_t::alignment);

        // Initialize the BD data
        m_BDmap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_BASIS_DERIVATIVE_DATA, vec_t::alignment);

        // Initialize the Z data.
        m_Zmap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_POINT_DATA, vec_t::alignment);

        auto nCoord = this->m_expansionList->GetCoordim(0);
        m_diffCoeff = Array<OneD, TData>(nCoord * (nCoord + 1) / 2, 0.0);
        if (nCoord >= 2)
        {
            m_diffCoeff[0] = 1.0; // D00
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
        if (in.GetAlignment() != vec_t::alignment)
        {
            NEKERROR(ErrorUtil::efatal,
                     "Input Field are not aligned to the required alignment "
                     "for the SIMD vector type.");
        }
        if (out.GetAlignment() != vec_t::alignment)
        {
            NEKERROR(ErrorUtil::efatal,
                     "Output Field are not aligned to the required alignment "
                     "for the SIMD vector type.");
        }
        // Reshape into vec_t::width. If the Field is already
        // interleaved, this method returns.
        in.template ReshapeStorage<vec_t::width>();
        out.template ReshapeStorage<vec_t::width>();

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
            auto const &inblock  = in.GetBlocks()[block_idx];
            auto const &outblock = out.GetBlocks()[block_idx];
            auto const nElmts    = inblock.num_elements;
            auto const nPadElmts = inblock.num_padding_elements;

            // Determine shape and type of the element.
            auto const expPtr    = this->m_expansionList->GetExp(m_exp_idx);
            auto const nqTot     = expPtr->GetTotPoints();
            auto const shapeType = expPtr->DetShapeType();
            auto const dimension = expPtr->GetShapeDimension();
            auto const deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;

            m_nElmtGroup = (nElmts + nPadElmts) / vec_t::width;

            // Fetch basis key for the current element type.
            m_basisKeys.clear();
            for (size_t d = 0; d < dimension; ++d)
            {
                m_basisKeys.push_back(expPtr->GetBasis(d)->GetBasisKey());
            }

#include "../Common/SwitchLevel2Deformed.h"

            inPtr += inblock.block_size;
            outPtr += outblock.block_size;
            m_exp_idx += nElmts;

            if (deformed) // update m_jac_idx globally
            {
                m_jac_idx += nqTot * m_nElmtGroup;
            }
            else
            {
                m_jac_idx += m_nElmtGroup;
            }
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorHelmholtzImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    int m_nElmtGroup, m_jac_idx, m_exp_idx;

    MemoryRegion<vec_t> m_jac;
    MemoryRegion<vec_t> m_df;

    BasisDataMap<vec_t> m_Bmap;
    BasisDataMap<vec_t> m_BDmap;
    BasisDataMap<vec_t> m_Dmap;
    BasisDataMap<vec_t> m_Zmap;
    BasisDataMap<vec_t> m_Wmap;

    std::vector<LibUtilities::BasisKey> m_basisKeys;

    // std::vector<LibUtilities::BasisSharedPtr> m_basis;
    // std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3> m_B;
    // std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3> m_DB;
    // std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3> m_D;
    // std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3> m_Z;

    // std::shared_ptr<std::vector<vec_t, tinysimd::allocator<vec_t>>> m_jac;
    // std::shared_ptr<std::vector<vec_t, tinysimd::allocator<vec_t>>> m_df;
    // std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3> m_w;
    // std::vector<vec_t, tinysimd::allocator<vec_t>> m_h0, m_h1, m_h2, m_h3;
    Array<OneD, TData> m_diffCoeff;

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator1D([[maybe_unused]] const NekDouble *input,
                    [[maybe_unused]] NekDouble *output)
    {
        ASSERTL0(false, "HelmholtzMatFree::operator1D: Not Impelented.");
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nq0>
    void operator1D([[maybe_unused]] const NekDouble *input,
                    [[maybe_unused]] NekDouble *output)
    {
        ASSERTL0(false, "HelmholtzMatFree::operator1D: Not Impelented.");
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator2D(const NekDouble *input, NekDouble *output)
    {
        auto const expPtr = this->m_expansionList->GetExp(m_exp_idx);

        auto const nm0 = expPtr->GetBasisNumModes(0);
        auto const nm1 = expPtr->GetBasisNumModes(1);

        auto const nq0 = expPtr->GetNumPoints(0);
        auto const nq1 = expPtr->GetNumPoints(1);

        constexpr auto ndf = 4;

        auto const nqTot = nq0 * nq1;
        auto const nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // auto const nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks = m_nmTot * vec_t::width;

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        std::vector<vec_t, allocator<vec_t>> m_h0, m_h1;

        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            const auto Z0 =
                m_Zmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            const auto Z1 =
                m_Zmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            GetHelmholtz2DHalfSpace<SHAPE_TYPE>(nq0, nq1, Z0, Z1, m_h0, m_h1);
        }

        // allocate workspace
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);

        NekDouble *bwd = static_cast<NekDouble *>(
            ::operator new[](nqTot *vec_t::width * sizeof(NekDouble),
                             std::align_val_t(vec_t::alignment)));
        vec_t::vectorType *bwdvec = reinterpret_cast<vec_t::vectorType *>(bwd);
        NekDouble *deriv0         = static_cast<NekDouble *>(
            ::operator new[](nqTot *vec_t::width * sizeof(NekDouble),
                             std::align_val_t(vec_t::alignment)));
        vec_t::vectorType *deriv0vec =
            reinterpret_cast<vec_t::vectorType *>(deriv0);
        NekDouble *deriv1 = static_cast<NekDouble *>(
            ::operator new[](nqTot *vec_t::width * sizeof(NekDouble),
                             std::align_val_t(vec_t::alignment)));
        vec_t::vectorType *deriv1vec =
            reinterpret_cast<vec_t::vectorType *>(deriv1);

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const vec_t *dfPtr =
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]);
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

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
            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, correct, B0, B1,
                                         wsp0, tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct2DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, bwdvec, B0, B1, W0, W1, jacPtr,
                wsp0, tmpOut, this->m_lambda);
            // Step 3: take derivatives in collapsed coordinate space
            PhysDerivTensor2DKernel(nq0, nq1, bwdvec, D0, D1, deriv0, deriv1);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeff2DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, true, this->m_diffCoeff, false, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray, dfPtr, m_h0, m_h1,
                deriv0, deriv1);
            // Step 5: Apply Laplacian metrics & inner product
            IProduct2DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, deriv0vec, BD0, B1, W0, W1, jacPtr,
                wsp0, tmpOut);
            IProduct2DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, deriv1vec, B0, BD1, W0, W1, jacPtr,
                wsp0, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * vec_t::width;
        }

        // free aligned memory
        ::operator delete[](bwd, std::align_val_t(vec_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(vec_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(vec_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nq0, int nq1>
    void operator2D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto nqTot = nq0 * nq1;
        constexpr auto ndf   = 4;
        auto const nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // auto const nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks = m_nmTot * vec_t::width;

        auto const expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        std::vector<vec_t, allocator<vec_t>> m_h0, m_h1;

        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            const auto Z0 =
                m_Zmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            const auto Z1 =
                m_Zmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            GetHelmholtz2DHalfSpace<SHAPE_TYPE>(nq0, nq1, Z0, Z1, m_h0, m_h1);
        }

        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);
        // std::vector<vec_t, allocator<vec_t>> bwd(nqTot), deriv0(nqTot),
        //     deriv1(nqTot);
        alignas(vec_t::alignment) NekDouble bwd[nqTot * vec_t::width];
        vec_t::vectorType *bwdvec = reinterpret_cast<vec_t::vectorType *>(bwd);
        alignas(vec_t::alignment) NekDouble deriv0[nqTot * vec_t::width];
        vec_t::vectorType *deriv0vec =
            reinterpret_cast<vec_t::vectorType *>(deriv0);
        alignas(vec_t::alignment) NekDouble deriv1[nqTot * vec_t::width];
        vec_t::vectorType *deriv1vec =
            reinterpret_cast<vec_t::vectorType *>(deriv1);

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const vec_t *dfPtr =
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]);
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

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
            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, correct, B0, B1,
                                         wsp0, tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct2DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, bwdvec, B0, B1, W0, W1, jacPtr,
                wsp0, tmpOut, this->m_lambda);
            // Step 3: take derivatives in collapsed coordinate space
            PhysDerivTensor2DKernel(nq0, nq1, bwdvec, D0, D1, deriv0, deriv1);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeff2DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, true, this->m_diffCoeff, false, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray, dfPtr, m_h0, m_h1,
                deriv0, deriv1);
            // Step 4: Apply Laplacian metrics & inner product
            IProduct2DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, deriv0vec, BD0, B1, W0, W1, jacPtr,
                wsp0, tmpOut);
            IProduct2DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, deriv1vec, B0, BD1, W0, W1, jacPtr,
                wsp0, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * vec_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator3D(const NekDouble *input, NekDouble *output)
    {
        auto const expPtr = this->m_expansionList->GetExp(m_exp_idx);

        auto const nm0 = expPtr->GetBasisNumModes(0);
        auto const nm1 = expPtr->GetBasisNumModes(1);
        auto const nm2 = expPtr->GetBasisNumModes(2);

        auto const nq0 = expPtr->GetNumPoints(0);
        auto const nq1 = expPtr->GetNumPoints(1);
        auto const nq2 = expPtr->GetNumPoints(2);

        constexpr auto ndf = 9;
        auto const nqTot   = nq0 * nq1 * nq2;
        auto const nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // auto const nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks = m_nmTot * vec_t::width;

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        std::vector<vec_t, allocator<vec_t>> m_h0, m_h1, m_h2, m_h3;

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
            GetHelmholtz3DHalfSpace<SHAPE_TYPE>(nq0, nq1, nq2, Z0, Z1, Z2, m_h0,
                                                m_h1, m_h2, m_h3);
        }

        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
            wsp2(wsp2Size);

        NekDouble *bwd = static_cast<NekDouble *>(
            ::operator new[](nqTot *vec_t::width * sizeof(NekDouble),
                             std::align_val_t(vec_t::alignment)));
        vec_t::vectorType *bwdvec = reinterpret_cast<vec_t::vectorType *>(bwd);

        NekDouble *deriv0 = static_cast<NekDouble *>(
            ::operator new[](nqTot *vec_t::width * sizeof(NekDouble),
                             std::align_val_t(vec_t::alignment)));
        vec_t::vectorType *deriv0vec =
            reinterpret_cast<vec_t::vectorType *>(deriv0);

        NekDouble *deriv1 = static_cast<NekDouble *>(
            ::operator new[](nqTot *vec_t::width * sizeof(NekDouble),
                             std::align_val_t(vec_t::alignment)));
        vec_t::vectorType *deriv1vec =
            reinterpret_cast<vec_t::vectorType *>(deriv1);

        NekDouble *deriv2 = static_cast<NekDouble *>(
            ::operator new[](nqTot *vec_t::width * sizeof(NekDouble),
                             std::align_val_t(vec_t::alignment)));
        vec_t::vectorType *deriv2vec =
            reinterpret_cast<vec_t::vectorType *>(deriv2);

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const vec_t *dfPtr =
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]);
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

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
            BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, correct,
                                         B0, B1, B2, wsp0, wsp1, tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct3DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, bwdvec, B0, B1, B2, W0,
                W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut, this->m_lambda);
            // Step 3: take derivatives in standard space
            PhysDerivTensor3DKernel(nq0, nq1, nq2, bwdvec, D0, D1, D2, deriv0,
                                    deriv1, deriv2);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeff3DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, true, this->m_diffCoeff, false,
                NullNekDouble1DArray, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray, dfPtr, m_h0, m_h1,
                m_h2, m_h3, deriv0, deriv1, deriv2);
            // Step 5: Apply Laplacian metrics & inner product
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, deriv0vec, BD0, B1, B2,
                W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, deriv1vec, B0, BD1, B2,
                W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, deriv2vec, B0, B1, BD2,
                W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * vec_t::width;
        }

        // free aligned memory
        ::operator delete[](bwd, std::align_val_t(vec_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(vec_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(vec_t::alignment));
        ::operator delete[](deriv2, std::align_val_t(vec_t::alignment));
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nm2, int nq0, int nq1, int nq2>
    void operator3D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto ndf   = 9;
        constexpr auto nqTot = nq0 * nq1 * nq2;
        auto const nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks     = m_nmTot * vec_t::width;

        auto const expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        std::vector<vec_t, allocator<vec_t>> m_h0, m_h1, m_h2, m_h3;

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
            GetHelmholtz3DHalfSpace<SHAPE_TYPE>(nq0, nq1, nq2, Z0, Z1, Z2, m_h0,
                                                m_h1, m_h2, m_h3);
        }

        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
            wsp2(wsp2Size);

        alignas(vec_t::alignment) NekDouble bwd[nqTot * vec_t::width];
        vec_t::vectorType *bwdvec = reinterpret_cast<vec_t::vectorType *>(bwd);

        alignas(vec_t::alignment) NekDouble deriv0[nqTot * vec_t::width];
        vec_t::vectorType *deriv0vec =
            reinterpret_cast<vec_t::vectorType *>(deriv0);

        alignas(vec_t::alignment) NekDouble deriv1[nqTot * vec_t::width];
        vec_t::vectorType *deriv1vec =
            reinterpret_cast<vec_t::vectorType *>(deriv1);

        alignas(vec_t::alignment) NekDouble deriv2[nqTot * vec_t::width];
        vec_t::vectorType *deriv2vec =
            reinterpret_cast<vec_t::vectorType *>(deriv2);

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }
        // m_jac_idx is an offset, accumulates over blocks
        const vec_t *dfPtr =
            &(m_df.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx * ndf]);
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

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
            BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, correct,
                                         B0, B1, B2, wsp0, wsp1, tmpIn, bwd);
            // Step 2: inner product for mass matrix operation
            IProduct3DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, bwdvec, B0, B1, B2, W0,
                W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut, this->m_lambda);
            // Step 3: take derivatives in standard space
            PhysDerivTensor3DKernel(nq0, nq1, nq2, bwdvec, D0, D1, D2, deriv0,
                                    deriv1, deriv2);
            // Step 4: apply diffusion coefficiets
            DiffusionCoeff3DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, true, this->m_diffCoeff, false,
                NullNekDouble1DArray, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray,
                NullNekDouble1DArray, NullNekDouble1DArray, dfPtr, m_h0, m_h1,
                m_h2, m_h3, deriv0, deriv1, deriv2);
            // Step 5: Apply Laplacian metrics & inner product
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, deriv0vec, BD0, B1, B2,
                W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, deriv1vec, B0, BD1, B2,
                W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            IProduct3DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, deriv2vec, B0, B1, BD2,
                W0, W1, W2, jacPtr, wsp0, wsp1, wsp2, tmpOut);
            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);
            // increment pointers:
            dfPtr += dfSize * ndf;
            jacPtr += dfSize;
            tmpIn += nmTot;
            tmpOut += nmTot * vec_t::width;
        }
    }
};

} // namespace Nektar::Operators::detail
