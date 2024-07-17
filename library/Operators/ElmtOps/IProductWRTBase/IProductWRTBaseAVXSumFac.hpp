///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseAVXSumFac.hpp
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
#include "ElmtOps/IProductWRTBase/IProductWRTBaseAVXSumFacKernels.hpp"
#include "ElmtOps/OperatorIProductWRTBase.hpp"

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
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise jacobian with paddings
        auto blocks =
            GetBlockAttributes(FieldState::Phys, expansionList, vec_t::width);

        // size_t jacSize         = Operator<TData>::GetGeometricFactorSize();
        // Array<OneD, TData> jac = Operator<TData>::SetJacobian(jacSize);

        // This jac is interleaved and ready to use.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize(blocks);
        std::shared_ptr<VecVec_t> jac =
            Operator<TData>::SetJacobian(jacSize, blocks);
        // but to integrate with kokkos and more, we need to transform it to
        // MemoryRegion<vec_t>. If TData and TDataIn are the same, the following
        // function will use memcpy to perform plain hard-copy.
        m_jac = MemoryRegion<vec_t>::template fromVector<MemSpace, vec_t>(
            *jac, vec_t::alignment);

        // Initialize the basis data.
        m_basisMap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_BASIS_DATA, vec_t::alignment);
        m_weightMap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_WEIGHT_DATA, vec_t::alignment);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               [[maybe_unused]] const TData lambda) override
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

        m_exp_idx = 0; // accumulates over blocks, accessed in operatorND()
        m_jac_idx = 0; // accumulates over blocks, accessed in operatorND()

        // Loop over the blocks.
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
                // m_basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
                m_basisKeys.push_back(expPtr->GetBasis(d)->GetBasisKey());
            }

#include "Operators/Common/SwitchLevel2Deformed.h"

            // Increment pointer and index for next element type.
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
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    static std::string className;

private:
    int m_nElmtGroup;
    int m_jac_idx, m_exp_idx;

    MemoryRegion<vec_t> m_jac;
    BasisDataMap<vec_t> m_basisMap;
    BasisDataMap<vec_t> m_weightMap;
    // std::array<LibUtilities::BasisKey, 3> m_basisKeys;
    std::vector<LibUtilities::BasisKey> m_basisKeys;

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator1D(const NekDouble *input, NekDouble *output)
    {
        auto const expPtr = this->m_expansionList->GetExp(m_exp_idx);

        auto const nm0 = expPtr->GetBasisNumModes(0);
        auto const nq0 = expPtr->GetNumPoints(0);

        auto const nmTot = nm0;
        auto const nqTot = nq0;
        // auto const nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks = m_nmTot * vec_t::width;

        // std::vector<vec_t, allocator<vec_t>> tmpIn(nqTot), tmpOut(m_nmTot);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

        const auto bPtr0 =
            m_basisMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr0 =
            m_weightMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // load_interleave(inPtr, nqTot, tmpIn);

            IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nq0, tmpIn, bPtr0, wPtr0, jacPtr, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
            jacPtr += jacSize;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nq0>
    void operator1D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto nmTot = nm0;
        constexpr auto nqTot = nq0;
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks     = m_nmTot * vec_t::width;

        // std::vector<vec_t, allocator<vec_t>> tmpIn(nqTot), tmpOut(m_nmTot);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

        const auto bPtr0 =
            m_basisMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr0 =
            m_weightMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // load_interleave(inPtr, nqTot, tmpIn);

            IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nq0, tmpIn, bPtr0, wPtr0, jacPtr, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
            jacPtr += jacSize;
        }
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

        auto const nqTot = nq0 * nq1;
        auto const nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // auto const nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks = m_nmTot * vec_t::width;

        // auto *inPtr  = &input[0];
        // auto *outPtr = &output[0];

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), tmpIn(nqTot),
        //     tmpOut(m_nmTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

        const auto bPtr0 =
            m_basisMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto bPtr1 =
            m_basisMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr0 =
            m_weightMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr1 =
            m_weightMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // load_interleave(inPtr, nqTot, tmpIn);

            IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, tmpIn, bPtr0, bPtr1, wPtr0, wPtr1,
                jacPtr, wsp0, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
            jacPtr += jacSize;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nq0, int nq1>
    void operator2D(const NekDouble *input, NekDouble *output)
    {
        auto const expPtr = this->m_expansionList->GetExp(m_exp_idx);

        constexpr auto nqTot = nq0 * nq1;
        auto const nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks     = m_nmTot * vec_t::width;

        // auto *inPtr  = &input[0];
        // auto *outPtr = &output[0];

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), tmpIn(nmTot),
        //     tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

        const auto bPtr0 =
            m_basisMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto bPtr1 =
            m_basisMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr0 =
            m_weightMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr1 =
            m_weightMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // load_interleave(inPtr, nqTot, tmpIn);

            IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, tmpIn, bPtr0, bPtr1, wPtr0, wPtr1,
                jacPtr, wsp0, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
            jacPtr += jacSize;
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

        auto const nqTot = nq0 * nq1 * nq2;
        auto const nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // auto const nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks = m_nmTot * vec_t::width;

        // auto *inPtr  = &input[0];
        // auto *outPtr = &output[0];

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
        //     wsp2(wsp2Size), tmpIn(nqTot), tmpOut(m_nmTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
            wsp2(wsp2Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

        const auto bPtr0 =
            m_basisMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto bPtr1 =
            m_basisMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto bPtr2 =
            m_basisMap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr0 =
            m_weightMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr1 =
            m_weightMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr2 =
            m_weightMap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // load_interleave(inPtr, nqTot, tmpIn);

            IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, tmpIn, bPtr0, bPtr1,
                bPtr2, wPtr0, wPtr1, wPtr2, jacPtr, wsp0, wsp1, wsp2, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
            jacPtr += jacSize;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nm2, int nq0, int nq1, int nq2>
    void operator3D(const NekDouble *input, NekDouble *output)
    {
        auto const expPtr = this->m_expansionList->GetExp(m_exp_idx);

        constexpr auto nqTot = nq0 * nq1 * nq2;
        auto const nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // auto const nmBlocks     = m_nmTot * vec_t::width;

        // auto *inPtr  = &input[0];
        // auto *outPtr = &output[0];

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
        //     wsp2(wsp2Size), tmpIn(nqTot), tmpOut(m_nmTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
            wsp2(wsp2Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        // Get jac and df pointers
        auto jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }
        const vec_t *jacPtr =
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]);

        const auto bPtr0 =
            m_basisMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto bPtr1 =
            m_basisMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto bPtr2 =
            m_basisMap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr0 =
            m_weightMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr1 =
            m_weightMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto wPtr2 =
            m_weightMap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Load and transpose data
            // load_interleave(inPtr, nqTot, tmpIn);

            IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, tmpIn, bPtr0, bPtr1,
                bPtr2, wPtr0, wPtr1, wPtr2, jacPtr, wsp0, wsp1, wsp2, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outPtr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
            jacPtr += jacSize;
        }
    }
};

} // namespace Nektar::Operators::detail
