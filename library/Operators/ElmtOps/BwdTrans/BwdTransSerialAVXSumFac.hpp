///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransSerialAVXSumFac.hpp
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
#include <LibUtilities/BasicUtils/NekInline.hpp>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>

namespace Nektar::Operators::detail
{

#include "ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp"

// Matrix-free implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::AVX>::value &&
               std::is_same<Implementation, Operators::SumFac>::value)>::type>
class OperatorBwdTransImpl : public OperatorBwdTrans<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same<ExecSpace, NektarSpaces::AVX>::value,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
        // Initialize the basis data.
        m_basisMap = GetBasisData<MemSpace, TData, simd_t>(
            expansionList, eBasis, ExecSpace::alignment);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        const auto dimension = this->m_expansionList->GetShapeDimension();

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
        auto *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();

        m_exp_idx = 0; // accumulated across each block.

        // Initialize basiskey.
        m_basisKeys = std::vector<LibUtilities::BasisKey>(
            dimension, LibUtilities::NullBasisKey);

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

            m_nElmtGroup = inblock.GetNumElmtGroups();

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < dimension; ++d)
            {
                m_basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
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
    BasisDataMap<simd_t> m_basisMap;
    std::vector<LibUtilities::BasisKey> m_basisKeys;

    int m_nElmtGroup, m_exp_idx;

    void SegBlock(const TData *inPtr, TData *outPtr);
    void TriBlock(const TData *inPtr, TData *outPtr);
    void QuadBlock(const TData *inPtr, TData *outPtr);
    void HexBlock(const TData *inPtr, TData *outPtr);
    void PrismBlock(const TData *inPtr, TData *outPtr);
    void PyrBlock(const TData *inPtr, TData *outPtr);
    void TetBlock(const TData *inPtr, TData *outPtr);

    // templated operator(), which is instantiated by SwitchNodesPoints.h
    // and used in apply().
    // size based template version
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nq0>
    void Operator1D(const TData *input, TData *output)
    {
        constexpr auto nqTot = nq0;
        constexpr auto nmTot = nm0;
        // Workspace for kernels - also checks preconditions
        BwdTrans1DWorkspace<SHAPE_TYPE>(nm0, nq0);

        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, bPtr0, tmpIn, tmpOut);

            tmpIn += nmTot;
            tmpOut += nqTot * simd_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nq0 = expPtr->GetNumPoints(0);

        const auto nqTot = nq0;
        const auto nmTot = nm0;

        BwdTrans1DWorkspace<SHAPE_TYPE>(nm0, nq0);

        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, bPtr0, tmpIn, tmpOut);

            tmpIn += nmTot;
            tmpOut += nqTot * simd_t::width;
        }
    }

    // size based template version
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nq0, int nq1>
    void Operator2D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        constexpr auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);
        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr1 = m_basisMap[m_basisKeys[1]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, correct, bPtr0,
                                         bPtr1, wsp0, tmpIn, tmpOut);
            tmpIn += nmTot;
            tmpOut += nqTot * simd_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nm1 = expPtr->GetBasisNumModes(1);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);

        const auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);
        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr1 = m_basisMap[m_basisKeys[1]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, correct, bPtr0,
                                         bPtr1, wsp0, tmpIn, tmpOut);
            tmpIn += nmTot;
            tmpOut += nqTot * simd_t::width;
        }
    }

    // size based template version
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nm2, int nq0, int nq1, int nq2>
    void Operator3D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        constexpr auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);
        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr1 = m_basisMap[m_basisKeys[1]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr2 = m_basisMap[m_basisKeys[2]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, correct,
                                         bPtr0, bPtr1, bPtr2, wsp0, wsp1, tmpIn,
                                         tmpOut);
            tmpIn += nmTot;
            tmpOut += nqTot * simd_t::width;
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

        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);
        const typename simd_t::vectorType *tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        typename simd_t::scalarType *tmpOut =
            reinterpret_cast<typename simd_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            const auto bPtr0 = m_basisMap[m_basisKeys[0]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr1 = m_basisMap[m_basisKeys[1]]
                                   .template GetPtr<MemSpace, ReadOnly>();
            const auto bPtr2 = m_basisMap[m_basisKeys[2]]
                                   .template GetPtr<MemSpace, ReadOnly>();

            BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, correct,
                                         bPtr0, bPtr1, bPtr2, wsp0, wsp1, tmpIn,
                                         tmpOut);
            tmpIn += nmTot;
            tmpOut += nqTot * simd_t::width;
        }
    }
};

} // namespace Nektar::Operators::detail
