///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivAVXSumFac.hpp
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
#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "Common/OperatorHelper.hpp"
#include "ElmtOps/OperatorPhysDeriv.hpp"
#include <LibUtilities/BasicUtils/NekInline.hpp>

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/Basis.h>

namespace Nektar::Operators::detail
{
using namespace tinysimd;
#ifdef IS_SERIAL
using vec_t = scalarT<double>;
#else
using vec_t = simd<double>;
#endif

#include "ElmtOps/PhysDeriv/PhysDerivAVXSumFacKernels.hpp"

// Matrix-free implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::AVX>::value &&
               std::is_same<Implementation, Operators::SumFac>::value)>::type>
class OperatorPhysDerivImpl : public OperatorPhysDeriv<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
        // Initialise jacobian with paddings
        auto blocks =
            GetBlockAttributes(FieldState::Phys, expansionList, vec_t::width);
        auto dfSize =
            Operator<TData>::GetGeometricFactorSize(blocks, vec_t::width);
        auto derivFac =
            Operator<TData>::SetDerivativeFactor(dfSize, blocks, vec_t::width);

        m_derivFac = MemoryRegion<TData>::template fromVector<MemSpace, TData>(
            *derivFac, vec_t::alignment);

        // Initialize the zeros
        m_zeroMap = GetBasisData<MemSpace, TData, vec_t>(expansionList, eZeros,
                                                         vec_t::alignment);

        // Initialize the derivative matrix.
        m_derivativeMap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, eDerivative, vec_t::alignment);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // check alignment
        WARNINGL1(in.GetAlignment() == vec_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(out.GetAlignment() == vec_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

        // Reshape into vec_t::width. If the Field is already
        // interleaved, this method returns.
        in.template ReshapeStorage<ExecSpace, vec_t::width>();
        out.template ReshapeStorage<ExecSpace, vec_t::width>();

        const auto *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr      = out.template GetPtr<MemSpace, ReadWrite>();

#if 0
        const auto Coordim = this->m_expansionList->GetExp(0)->GetCoordim();
        ASSERTL0(Coordim <= out.GetNumComponents(),
                 "Output field has fewer components than the coordinate!");
#else
        m_coordDim = this->m_expansionList->GetExp(0)->GetCoordim();
        ASSERTL0(m_coordDim <= out.GetNumComponents(),
                 "Output field has fewer components than the coordinate!");
#endif
        m_outSize = out.GetFieldSize();

        m_exp_idx = 0; // accumulates over blocks, also used in operatorND()
        m_df_idx  = 0; // accumulates over blocks, accessed in operatorND()

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

            m_nElmtGroup = inblock.GetNumElmtGroups(vec_t::width);

            // Fetch basis key for the current element type.
            m_basisKeys.clear();
            for (size_t d = 0; d < dimension; ++d)
            {
                // m_basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
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
            // #include "Operators/Common/SwitchLevel1DeformedCoord.h"

            // Increment pointer and index for next element type.
            inPtr += inblock.block_size;
            outPtr += outblock.block_size;

            if (deformed)
            {
                m_df_idx += nqTot * m_nElmtGroup * vec_t::width;
            }
            else
            {
                m_df_idx += m_nElmtGroup * vec_t::width;
            }
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
            OperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    int m_nElmtGroup, m_df_idx, m_exp_idx;
    size_t m_outSize;
    size_t m_coordDim;

    MemoryRegion<TData> m_derivFac;
    BasisDataMap<vec_t> m_zeroMap;
    BasisDataMap<vec_t> m_derivativeMap;
    std::vector<LibUtilities::BasisKey> m_basisKeys;

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
        const auto nCoord = this->m_expansionList->GetCoordim(0);

        const auto nq0     = expPtr->GetNumPoints(0);
        const auto nqTot   = nq0;
        const auto nqBlock = nqTot * vec_t::width;

        const auto ndf = nCoord;
        int dfsize     = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Get derivative factor pointer
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(&(
            m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx * ndf]));

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        // const auto Z0 = m_zeroMap[m_basisKeys[0]].template GetPtr<MemSpace,
        // ReadOnly>();
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        for (int d = 0; d < nCoord; ++d)
        {
            tmpOut[d] =
                reinterpret_cast<vec_t::scalarType *>(output + d * m_outSize);
        }

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Get the basic derivative
            PhysDerivTensor1DKernel(nq0, tmpIn, D0, tmpOut[0]);

            // Calculate physical derivative
            PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(nq0, nCoord, dfPtr, tmpOut);

            // Increment pointers
            dfPtr += dfsize;
            tmpIn += nqTot;
            for (int d = 0; d < nCoord; ++d)
            {
                tmpOut[d] += nqBlock;
            }
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nCoord,
              int nq0>
    void Operator1D(const TData *input, TData *output)
    {
        constexpr auto nqTot   = nq0;
        constexpr auto nqBlock = nqTot * vec_t::width;

        // Get size of derivative factor block
        constexpr auto ndf = nCoord;
        int dfsize         = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Get derivative factor pointer
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(&(
            m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx * ndf]));

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        // const auto Z0 = m_zeroMap[m_basisKeys[0]].template GetPtr<MemSpace,
        // ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        for (int d = 0; d < nCoord; ++d)
        {
            tmpOut[d] =
                reinterpret_cast<vec_t::scalarType *>(output + d * m_outSize);
        }

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Get the basic derivative
            PhysDerivTensor1DKernel(nq0, tmpIn, D0, tmpOut[0]);

            // Calculate physical derivative
            PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(nq0, nCoord, dfPtr, tmpOut);

            // Increment pointers
            dfPtr += dfsize;
            tmpIn += nqTot;
            for (int d = 0; d < nCoord; ++d) // automatically unrolled
            {
                tmpOut[d] += nqBlock;
            }
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);
        const auto nCoord = this->m_expansionList->GetCoordim(0);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);

        const auto nqTot   = nq0 * nq1;
        const auto nqBlock = nqTot * vec_t::width;

        const auto ndf = 2 * nCoord;
        int dfsize     = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Get derivative factor pointer
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(&(
            m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx * ndf]));

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D1 = m_derivativeMap[m_basisKeys[1]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto Z0 =
            m_zeroMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z1 =
            m_zeroMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        for (int d = 0; d < nCoord; ++d)
        {
            tmpOut[d] =
                reinterpret_cast<vec_t::scalarType *>(output + d * m_outSize);
        }

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Results written to tmpOut0, tmpOut1
            PhysDerivTensor2DKernel(nq0, nq1, tmpIn, D0, D1, tmpOut[0],
                                    tmpOut[1]);
            // Calculate physical derivative
            PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(nq0, nq1, nCoord, Z0, Z1,
                                                    dfPtr, tmpOut);
            // Increment pointers
            dfPtr += dfsize;
            tmpIn += nqTot;
            for (int d = 0; d < nCoord; ++d) // automatically unrolled
            {
                tmpOut[d] += nqBlock;
            }
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nCoord,
              int nq0, int nq1>
    void Operator2D(const TData *input, TData *output)
    {
        constexpr auto nqTot   = nq0 * nq1;
        constexpr auto nqBlock = nqTot * vec_t::width;

        constexpr auto ndf = 2 * nCoord;
        int dfsize         = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Get derivative factor pointer
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(&(
            m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx * ndf]));

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D1 = m_derivativeMap[m_basisKeys[1]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto Z0 =
            m_zeroMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z1 =
            m_zeroMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        for (int d = 0; d < nCoord; ++d)
        {
            tmpOut[d] =
                reinterpret_cast<vec_t::scalarType *>(output + d * m_outSize);
        }

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Results written to tmpOut0, tmpOut1
            PhysDerivTensor2DKernel(nq0, nq1, tmpIn, D0, D1, tmpOut[0],
                                    tmpOut[1]);
            // Calculate physical derivative
            PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(nq0, nq1, nCoord, Z0, Z1,
                                                    dfPtr, tmpOut);
            // Increment pointers
            dfPtr += dfsize;
            tmpIn += nqTot;
            for (int d = 0; d < nCoord; ++d) // automatically unrolled
            {
                tmpOut[d] += nqBlock;
            }
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(const TData *input, TData *output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);
        const auto nq2 = expPtr->GetNumPoints(2);

        const auto nqTot    = nq0 * nq1 * nq2;
        const auto nqBlocks = nqTot * vec_t::width;

        constexpr auto ndf = 9u;
        int dfsize         = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0;
        PhysDeriv3DWorkspace<SHAPE_TYPE>(nq0, nq1, nq2, wsp0Size, wsp1Size);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size);

        // Get derivative factor pointer
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(&(
            m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx * ndf]));

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D1 = m_derivativeMap[m_basisKeys[1]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D2 = m_derivativeMap[m_basisKeys[2]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto Z0 =
            m_zeroMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z1 =
            m_zeroMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z2 =
            m_zeroMap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        tmpOut[0] = reinterpret_cast<vec_t::scalarType *>(output);
        tmpOut[1] = reinterpret_cast<vec_t::scalarType *>(output + m_outSize);
        tmpOut[2] =
            reinterpret_cast<vec_t::scalarType *>(output + 2 * m_outSize);

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            PhysDerivTensor3DKernel(nq0, nq1, nq2, tmpIn, D0, D1, D2, tmpOut[0],
                                    tmpOut[1], tmpOut[2]);
            // Calculate physical derivative
            PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, Z0, Z1, Z2, dfPtr, wsp0, wsp1, tmpOut[0],
                tmpOut[1], tmpOut[2]);

            // Increment pointers
            dfPtr += dfsize;
            tmpIn += nqTot;
            tmpOut[0] += nqBlocks;
            tmpOut[1] += nqBlocks;
            tmpOut[2] += nqBlocks;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nq0,
              int nq1, int nq2>
    void Operator3D(const TData *input, TData *output)
    {
        constexpr auto nqTot    = nq0 * nq1 * nq2;
        constexpr auto nqBlocks = nqTot * vec_t::width;

        constexpr auto ndf = 9u;
        int dfsize         = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0;
        PhysDeriv3DWorkspace<SHAPE_TYPE>(nq0, nq1, nq2, wsp0Size, wsp1Size);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size);

        // Get derivative factor pointer
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(&(
            m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx * ndf]));

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D1 = m_derivativeMap[m_basisKeys[1]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D2 = m_derivativeMap[m_basisKeys[2]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto Z0 =
            m_zeroMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z1 =
            m_zeroMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z2 =
            m_zeroMap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        tmpOut[0] = reinterpret_cast<vec_t::scalarType *>(output);
        tmpOut[1] = reinterpret_cast<vec_t::scalarType *>(output + m_outSize);
        tmpOut[2] =
            reinterpret_cast<vec_t::scalarType *>(output + 2 * m_outSize);

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            PhysDerivTensor3DKernel(nq0, nq1, nq2, tmpIn, D0, D1, D2, tmpOut[0],
                                    tmpOut[1], tmpOut[2]);
            // Calculate physical derivative
            PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, Z0, Z1, Z2, dfPtr, wsp0, wsp1, tmpOut[0],
                tmpOut[1], tmpOut[2]);

            // Increment pointers
            dfPtr += dfsize;
            tmpIn += nqTot;
            tmpOut[0] += nqBlocks;
            tmpOut[1] += nqBlocks;
            tmpOut[2] += nqBlocks;
        }
    }
};

} // namespace Nektar::Operators::detail
