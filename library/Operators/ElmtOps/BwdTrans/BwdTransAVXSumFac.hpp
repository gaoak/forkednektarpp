///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransAVXSumFac.hpp
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

#include "ElmtOps/BwdTrans/BwdTransAVXSumFacKernels.hpp"

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/Basis.h>

namespace Nektar::Operators::detail
{

// Matrix-free implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::AVX>::value &&
              std::is_same<Implementation, Operators::SumFac>::value>::type>
class OperatorBwdTransImpl : public OperatorBwdTrans<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
        // Initialize the basis data.
        m_basisMap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_BASIS_DATA, vec_t::alignment);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
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
        in.template ReshapeStorage<ExecSpace, vec_t::width>();
        out.template ReshapeStorage<ExecSpace, vec_t::width>();

        const auto *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();

        m_exp_idx = 0; // accumulated across each block.

        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            const auto &inblock  = in.GetBlocks()[block_idx];
            const auto &outblock = out.GetBlocks()[block_idx];
            const auto nElmts    = inblock.num_elements;

            // Determine shape and type of the element.
            const auto expPtr    = this->m_expansionList->GetExp(m_exp_idx);
            const auto shapeType = expPtr->DetShapeType();
            const auto dimension = expPtr->GetShapeDimension();

            m_nElmtGroup = inblock.num_elmt_groups;

            // Fetch basis key for the current element type.
            m_basisKeys.clear();
            for (size_t d = 0; d < dimension; ++d)
            {
                // m_basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
                m_basisKeys.push_back(expPtr->GetBasis(d)->GetBasisKey());
            }

#include "Common/SwitchLevel2NoDeformed.h"

            // Increment pointer and index for next element type.
            inPtr += inblock.block_size;
            outPtr += outblock.block_size;
            m_exp_idx += nElmts;
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorBwdTransImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    BasisDataMap<vec_t> m_basisMap;
    // std::array<LibUtilities::BasisKey, 3> m_basisKeys;
    std::vector<LibUtilities::BasisKey> m_basisKeys;

    int m_nElmtGroup, m_exp_idx;

    // templated operator(), which is instantiated by SwitchNodesPoints.h
    // and used in apply().
    // size based template version
    template <LibUtilities::ShapeType SHAPE_TYPE, int nm0, int nq0>
    void operator1D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto nqTot = nq0;
        constexpr auto nmTot = nm0;
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // constexpr auto nmBlocks = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmtGroup;
        // Workspace for kernels - also checks preconditions
        BwdTrans1DWorkspace<SHAPE_TYPE>(nm0, nq0);

        // std::vector<vec_t, allocator<vec_t>> tmpIn(nmTot), tmpOut(nqTot);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, bPtr0, tmpIn, tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void operator1D(const NekDouble *input, NekDouble *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nq0 = expPtr->GetNumPoints(0);

        const auto nqTot = nq0;
        const auto nmTot = nm0;
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        // Workspace for kernels - also checks preconditions
        BwdTrans1DWorkspace<SHAPE_TYPE>(nm0, nq0);

        // std::vector<vec_t, allocator<vec_t>> tmpIn(nmTot), tmpOut(nqTot);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, bPtr0, tmpIn, tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // size based template version
    template <LibUtilities::ShapeType SHAPE_TYPE, int nm0, int nm1, int nq0,
              int nq1>
    void operator2D(const NekDouble *input, NekDouble *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        constexpr auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks     = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), tmpIn(nmTot),
        //     tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr1 = m_basisMap[m_basisKeys[1]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, correct, bPtr0,
                                         bPtr1, wsp0, tmpIn, tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void operator2D(const NekDouble *input, NekDouble *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nm1 = expPtr->GetBasisNumModes(1);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);

        const auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), tmpIn(nmTot),
        //     tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr1 = m_basisMap[m_basisKeys[1]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, correct, bPtr0,
                                         bPtr1, wsp0, tmpIn, tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // size based template version
    template <LibUtilities::ShapeType SHAPE_TYPE, int nm0, int nm1, int nm2,
              int nq0, int nq1, int nq2>
    void operator3D(const NekDouble *input, NekDouble *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        constexpr auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks     = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
        //     tmpIn(nmTot), tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr1 = m_basisMap[m_basisKeys[1]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr2 = m_basisMap[m_basisKeys[2]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, correct,
                                         bPtr0, bPtr1, bPtr2, wsp0, wsp1, tmpIn,
                                         tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void operator3D(const NekDouble *input, NekDouble *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nm1 = expPtr->GetBasisNumModes(1);
        const auto nm2 = expPtr->GetBasisNumModes(2);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);
        const auto nq2 = expPtr->GetNumPoints(2);

        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
        //     tmpIn(nmTot), tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr1 = m_basisMap[m_basisKeys[1]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr2 = m_basisMap[m_basisKeys[2]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, correct,
                                         bPtr0, bPtr1, bPtr2, wsp0, wsp1, tmpIn,
                                         tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }
};

} // namespace Nektar::Operators::detail
