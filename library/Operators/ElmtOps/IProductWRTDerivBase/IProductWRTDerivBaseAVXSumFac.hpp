///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseAVXSumFac.hpp
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
#include "ElmtOps/OperatorIProductWRTDerivBase.hpp"
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
#include "ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseAVXSumFacKernels.hpp"

// Sum-factorisation implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, NektarSpaces::AVX>::value &&
               std::is_same<Implementation, Operators::SumFac>::value)>::type>
class OperatorIProductWRTDerivBaseImpl
    : public OperatorIProductWRTDerivBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
        // Initialise jacobian with paddings if appropriate
        auto locblocks =
            GetBlockAttributes(FieldState::Phys, expansionList, vec_t::width);
        size_t gFacSize =
            Operator<TData>::GetGeometricFactorSize(locblocks, vec_t::width);
        auto jac =
            Operator<TData>::SetJacobian(gFacSize, locblocks, vec_t::width);
        auto derivFac = Operator<TData>::SetDerivativeFactor(
            gFacSize, locblocks, vec_t::width);

        m_jac = MemoryRegion<TData>::template fromVector<MemSpace, TData>(
            *jac, ExecSpace::alignment);

        m_derivFac = MemoryRegion<TData>::template fromVector<MemSpace, TData>(
            *derivFac, ExecSpace::alignment);

        // Initialize the basis data.
        m_Bmap = GetBasisData<MemSpace, TData, vec_t>(expansionList, eBasis,
                                                      vec_t::alignment);
        m_Wmap = GetBasisData<MemSpace, TData, vec_t>(expansionList, eWeights,
                                                      vec_t::alignment);
        // Initialize the derivative matrix.
        m_Dmap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, eDerivative, vec_t::alignment);
        // Initialize the BD data
        m_BDmap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, eBasisDerivative, vec_t::alignment);
        // Initialize the geometric factors
        m_Fac0 = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, eHalfMultOnePlusZero, vec_t::alignment);
        m_Fac1 = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, eTwoOverOneMinusZero, vec_t::alignment);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               [[maybe_unused]] bool APPEND = false) override
    {

        // check alignment
        WARNINGL1(in.GetAlignment() == vec_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(out.GetAlignment() == vec_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

        if (in.GetVecWidth() != vec_t::width)
        {
            // Reshape into vec_t::width. If the Field is already
            // interleaved, this method returns.
            in.template ReshapeStorage<ExecSpace, vec_t::width>();
            out.template ReshapeStorage<ExecSpace, vec_t::width>();
        }

        const auto *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr      = out.template GetPtr<MemSpace, ReadWrite>();

        m_exp_idx = 0;
        m_jac_idx = 0;
        m_df_idx  = 0;
        m_inSize  = in.GetFieldSize();

        // Loop over the blocks.
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

            // Increment pointer and index for next element type.
            inPtr += inblock.block_size;
            outPtr += outblock.block_size;
            m_exp_idx += nElmts;

            if (deformed) // update m_jac_idx globally
            {
                m_jac_idx += nqTot * m_nElmtGroup * vec_t::width;
            }
            else
            {
                m_jac_idx += m_nElmtGroup * vec_t::width;
            }
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    size_t m_jac_idx;
    size_t m_df_idx;
    size_t m_exp_idx;
    size_t m_nElmtGroup;
    size_t m_inSize;

    MemoryRegion<TData> m_jac;
    MemoryRegion<TData> m_derivFac;
    std::vector<LibUtilities::BasisKey> m_basisKeys;

    BasisDataMap<vec_t> m_Bmap;
    BasisDataMap<vec_t> m_BDmap;
    BasisDataMap<vec_t> m_Dmap;
    BasisDataMap<vec_t> m_Wmap;
    BasisDataMap<vec_t> m_Fac0;
    BasisDataMap<vec_t> m_Fac1;

    void SegBlock(const TData *inPtr, TData *outPtr);

    void TriBlock(const TData *inPtr, TData *outPtr);

    void QuadBlock(const TData *inPtr, TData *outPtr);

    void HexBlock(const TData *inPtr, TData *outPtr);

    void PrismBlock(const TData *inPtr, TData *outPtr);

    void PyrBlock(const TData *inPtr, TData *outPtr);

    void TetBlock(const TData *inPtr, TData *outPtr);

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(const TData *inPtr, TData *outPtr)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nmodes0 = expPtr->GetBasisNumModes(0);
        const auto nquad0  = expPtr->GetNumPoints(0);

        const auto ncoord = this->m_expansionList->GetCoordim(0);

        // Fetch basis key for the current element type.
        m_basisKeys.clear();
        m_basisKeys.push_back(expPtr->GetBasis(0)->GetBasisKey());

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(inPtr);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(outPtr);

        std::vector<vec_t, tinysimd::allocator<vec_t>> df_tmp(ncoord);
        std::vector<vec_t, tinysimd::allocator<vec_t>> tmp0(nquad0);

        vec_t::scalarType *tmpPtr =
            reinterpret_cast<vec_t::scalarType *>(tmp0.data());

        size_t ipt = 1;

        if (DEFORMED)
        {
            ipt *= nquad0;
        }
        const vec_t *jacPtr = reinterpret_cast<const vec_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(
            &(m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx]));
        const auto BD0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            StdAlignDerivBase1D<DEFORMED>(nquad0, ncoord, dfPtr, df_tmp,
                                          m_inSize, tmpIn, tmpPtr);
            IProductSegKernel<false, false, DEFORMED>(
                nmodes0, nquad0, (const vec_t::vectorType *)tmpPtr, BD0, W0,
                jacPtr, tmpOut);

            tmpIn += nquad0;
            tmpOut += nmodes0 * vec_t::width;
            jacPtr += ipt;
            dfPtr += ipt * ncoord;
        }
        m_df_idx += ipt * ncoord * m_nElmtGroup * vec_t::width;
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nmodes0,
              int nquad0>
    void Operator1D(const TData *inPtr, TData *outPtr)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto ncoord = this->m_expansionList->GetCoordim(0);

        // Fetch basis key for the current element type.
        m_basisKeys.clear();
        m_basisKeys.push_back(expPtr->GetBasis(0)->GetBasisKey());

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(inPtr);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(outPtr);

        std::vector<vec_t, tinysimd::allocator<vec_t>> df_tmp(ncoord);
        std::vector<vec_t, tinysimd::allocator<vec_t>> tmp0(nquad0);

        vec_t::scalarType *tmpPtr =
            reinterpret_cast<vec_t::scalarType *>(tmp0.data());

        size_t ipt = 1;
        if (DEFORMED)
        {
            ipt *= nquad0;
        }
        const vec_t *jacPtr = reinterpret_cast<const vec_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(
            &(m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx]));
        const auto BD0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            StdAlignDerivBase1D<DEFORMED>(nquad0, ncoord, dfPtr, df_tmp,
                                          m_inSize, tmpIn, tmpPtr);
            IProductSegKernel<false, false, DEFORMED>(
                nmodes0, nquad0, (const vec_t::vectorType *)tmpPtr, BD0, W0,
                jacPtr, tmpOut);

            tmpIn += nquad0;
            tmpOut += nmodes0 * vec_t::width;
            jacPtr += ipt;
            dfPtr += ipt * ncoord;
        }
        m_df_idx += ipt * ncoord * m_nElmtGroup * vec_t::width;
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(const TData *inPtr, TData *outPtr)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nmodes0 = expPtr->GetBasisNumModes(0);
        const auto nmodes1 = expPtr->GetBasisNumModes(1);

        const auto nquad0 = expPtr->GetNumPoints(0);
        const auto nquad1 = expPtr->GetNumPoints(1);

        const auto ncoord = this->m_expansionList->GetCoordim(0);

        auto const ndf       = 2u * ncoord;
        auto const totPoints = nquad0 * nquad1;
        auto const totModes  = expPtr->GetNcoeffs();

        // Get Basis and weight data
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto B1 =
            m_Bmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB1 =
            m_BDmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W1 =
            m_Wmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        std::vector<vec_t, tinysimd::allocator<vec_t>> df_tmp(ndf);

        std::vector<vec_t, tinysimd::allocator<vec_t>> wsp(nquad1),
            tmp0(totPoints), tmp1(totPoints);

        // provide pointer to temporary space for use in kernels
        vec_t::scalarType *tmpPtr[2];
        tmpPtr[0]  = reinterpret_cast<vec_t::scalarType *>(tmp0.data());
        tmpPtr[1]  = reinterpret_cast<vec_t::scalarType *>(tmp1.data());
        size_t ipt = 1;

        if (DEFORMED)
        {
            ipt *= totPoints;
        }
        const vec_t *jacPtr = reinterpret_cast<const vec_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(
            &(m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx]));
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(inPtr);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(outPtr);

        if constexpr (SHAPE_TYPE == LibUtilities::eQuadrilateral)
        {
            const bool colldir0 = expPtr->GetBasis(0)->Collocation();
            const bool colldir1 = expPtr->GetBasis(1)->Collocation();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, ncoord, dfPtr, df_tmp, m_inSize, tmpIn,
                    tmpPtr, nullptr, nullptr);
                IProductQuadKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nquad0, nquad1,
                    (const vec_t::vectorType *)tmpPtr[0], DB0, B1, W0, W1,
                    jacPtr, wsp, tmpOut, 1.0, false, colldir1);
                IProductQuadKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nquad0, nquad1,
                    (const vec_t::vectorType *)tmpPtr[1], B0, DB1, W0, W1,
                    jacPtr, wsp, tmpOut, 1.0, colldir0, false);

                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }

        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            const bool isModified =
                (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

            const vec_t *F0 =
                m_Fac0[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            const vec_t *F1 =
                m_Fac1[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, ncoord, dfPtr, df_tmp, m_inSize, tmpIn,
                    tmpPtr, F0, F1);
                IProductTriKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nquad0, nquad1, isModified,
                    (const vec_t::vectorType *)tmpPtr[0], DB0, B1, W0, W1,
                    jacPtr, wsp, tmpOut);
                IProductTriKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nquad0, nquad1, isModified,
                    (const vec_t::vectorType *)tmpPtr[1], B0, DB1, W0, W1,
                    jacPtr, wsp, tmpOut);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nmodes0,
              int nmodes1, int nquad0, int nquad1>
    void Operator2D(const TData *inPtr, TData *outPtr)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto ncoord = this->m_expansionList->GetCoordim(0);

        auto const ndf       = 2u * ncoord;
        auto const totPoints = nquad0 * nquad1;
        auto const totModes  = expPtr->GetNcoeffs();

        // Get Basis and weight data
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto B1 =
            m_Bmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB1 =
            m_BDmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W1 =
            m_Wmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        std::vector<vec_t, tinysimd::allocator<vec_t>> df_tmp(ndf);

        std::vector<vec_t, tinysimd::allocator<vec_t>> wsp(nquad1),
            tmp0(totPoints), tmp1(totPoints);

        // provide pointer to temporary space for use in kernels
        vec_t::scalarType *tmpPtr[2];
        tmpPtr[0] = reinterpret_cast<vec_t::scalarType *>(tmp0.data());
        tmpPtr[1] = reinterpret_cast<vec_t::scalarType *>(tmp1.data());

        size_t ipt = 1;
        if (DEFORMED)
        {
            ipt *= totPoints;
        }
        const vec_t *jacPtr = reinterpret_cast<const vec_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(
            &(m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx]));
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(inPtr);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(outPtr);

        if constexpr (SHAPE_TYPE == LibUtilities::eQuadrilateral)
        {
            const bool colldir0 = expPtr->GetBasis(0)->Collocation();
            const bool colldir1 = expPtr->GetBasis(1)->Collocation();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, ncoord, dfPtr, df_tmp, m_inSize, tmpIn,
                    tmpPtr, nullptr, nullptr);
                IProductQuadKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nquad0, nquad1,
                    (const vec_t::vectorType *)tmpPtr[0], DB0, B1, W0, W1,
                    jacPtr, wsp, tmpOut, 1.0, false, colldir1);
                IProductQuadKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nquad0, nquad1,
                    (const vec_t::vectorType *)tmpPtr[1], B0, DB1, W0, W1,
                    jacPtr, wsp, tmpOut, 1.0, colldir0, false);

                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }

        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle)
        {
            const bool isModified =
                (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

            const vec_t *F0 =
                m_Fac0[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            const vec_t *F1 =
                m_Fac1[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, ncoord, dfPtr, df_tmp, m_inSize, tmpIn,
                    tmpPtr, F0, F1);
                IProductTriKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nquad0, nquad1, isModified,
                    (const vec_t::vectorType *)tmpPtr[0], DB0, B1, W0, W1,
                    jacPtr, wsp, tmpOut);
                IProductTriKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nquad0, nquad1, isModified,
                    (const vec_t::vectorType *)tmpPtr[1], B0, DB1, W0, W1,
                    jacPtr, wsp, tmpOut);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(const TData *&inPtr, TData *&outPtr)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nmodes0 = expPtr->GetBasisNumModes(0);
        const auto nmodes1 = expPtr->GetBasisNumModes(1);
        const auto nmodes2 = expPtr->GetBasisNumModes(2);

        const auto nquad0 = expPtr->GetNumPoints(0);
        const auto nquad1 = expPtr->GetNumPoints(1);
        const auto nquad2 = expPtr->GetNumPoints(2);

        auto const ndf       = 9u;
        auto const totPoints = nquad0 * nquad1 * nquad2;
        auto const totModes  = expPtr->GetNcoeffs();

        // Get Basis and weight data
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto B1 =
            m_Bmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto B2 =
            m_Bmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB1 =
            m_BDmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB2 =
            m_BDmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W1 =
            m_Wmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto W2 =
            m_Wmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        std::vector<vec_t, tinysimd::allocator<vec_t>> df_tmp(ndf);
        std::vector<vec_t, tinysimd::allocator<vec_t>> wsp(nquad1 * nquad2),
            wsp0(nquad2), tmp0(totPoints), tmp1(totPoints), tmp2(totPoints);

        // provide pointer to temporary space for use in kernels
        vec_t::scalarType *tmpPtr[3];
        tmpPtr[0] = reinterpret_cast<vec_t::scalarType *>(tmp0.data());
        tmpPtr[1] = reinterpret_cast<vec_t::scalarType *>(tmp1.data());
        tmpPtr[2] = reinterpret_cast<vec_t::scalarType *>(tmp2.data());

        size_t ipt = 1;
        if (DEFORMED)
        {
            ipt *= totPoints;
        }
        const vec_t *jacPtr = reinterpret_cast<const vec_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(
            &(m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx]));
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(inPtr);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(outPtr);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            const bool colldir0 = expPtr->GetBasis(0)->Collocation();
            const bool colldir1 = expPtr->GetBasis(1)->Collocation();
            const bool colldir2 = expPtr->GetBasis(2)->Collocation();
            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBaseHex<DEFORMED>(nquad0, nquad1, nquad2, dfPtr,
                                               df_tmp, m_inSize, tmpIn, tmpPtr);
                IProductHexKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    (const vec_t::vectorType *)tmpPtr[0], DB0, B1, B2, W0, W1,
                    W2, jacPtr, wsp, wsp0, tmpOut, 1.0, false, colldir1,
                    colldir2);
                IProductHexKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    (const vec_t::vectorType *)tmpPtr[1], B0, DB1, B2, W0, W1,
                    W2, jacPtr, wsp, wsp0, tmpOut, 1.0, colldir0, false,
                    colldir2);
                IProductHexKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    (const vec_t::vectorType *)tmpPtr[2], B0, B1, DB2, W0, W1,
                    W2, jacPtr, wsp, wsp0, tmpOut, 1.0, colldir0, colldir1,
                    false);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const bool isModified =
                (expPtr->GetBasisType(0) == LibUtilities::eModified_A);
            const vec_t *F0, *F1, *F1a, *F2;

            F0  = m_Fac0[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            F1  = m_Fac0[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            F1a = m_Fac1[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            F2  = m_Fac1[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, nquad2, dfPtr, df_tmp, m_inSize, F0, F1,
                    F1a, F2, tmpIn, tmpPtr);
                IProductTetKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[0], DB0, B1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                IProductTetKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[1], B0, DB1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                IProductTetKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[2], B0, B1,
                    DB2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            const bool isModified =
                (expPtr->GetBasisType(0) == LibUtilities::eModified_A);
            const vec_t *F0, *F1, *F2;

            F0 = m_Fac0[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            F1 = m_Fac0[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            F2 = m_Fac1[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, nquad2, dfPtr, df_tmp, m_inSize, F0, F1,
                    nullptr, F2, tmpIn, tmpPtr);
                IProductPyrKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[0], DB0, B1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                IProductPyrKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[1], B0, DB1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                IProductPyrKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[2], B0, B1,
                    DB2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            std::vector<vec_t, tinysimd::allocator<vec_t>> wsp1(nmodes1);
            const bool isModified =
                (expPtr->GetBasisType(0) == LibUtilities::eModified_A);
            const vec_t *F0, *F2;

            F0 = m_Fac0[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            F2 = m_Fac1[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, nquad2, dfPtr, df_tmp, m_inSize, F0,
                    nullptr, nullptr, F2, tmpIn, tmpPtr);
                IProductPrismKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[0], DB0, B1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, wsp1, tmpOut);
                IProductPrismKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[1], B0, DB1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, wsp1, tmpOut);
                IProductPrismKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[2], B0, B1,
                    DB2, W0, W1, W2, jacPtr, wsp, wsp0, wsp1, tmpOut);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nmodes0,
              int nmodes1, int nmodes2, int nquad0, int nquad1, int nquad2>
    void Operator3D(const TData *&inPtr, TData *&outPtr)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        auto const ndf       = 9u;
        auto const totPoints = nquad0 * nquad1 * nquad2;
        auto const totModes  = expPtr->GetNcoeffs();

        // Get Basis and weight data
        const auto B0 =
            m_Bmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto B1 =
            m_Bmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto B2 =
            m_Bmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB0 =
            m_BDmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB1 =
            m_BDmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto DB2 =
            m_BDmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();
        const auto W0 =
            m_Wmap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto W1 =
            m_Wmap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto W2 =
            m_Wmap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        std::vector<vec_t, tinysimd::allocator<vec_t>> df_tmp(ndf);
        std::vector<vec_t, tinysimd::allocator<vec_t>> wsp(nquad1 * nquad2),
            wsp0(nquad2), tmp0(totPoints), tmp1(totPoints), tmp2(totPoints);

        // provide pointer to temporary space for use in kernels
        vec_t::scalarType *tmpPtr[3];
        tmpPtr[0] = reinterpret_cast<vec_t::scalarType *>(tmp0.data());
        tmpPtr[1] = reinterpret_cast<vec_t::scalarType *>(tmp1.data());
        tmpPtr[2] = reinterpret_cast<vec_t::scalarType *>(tmp2.data());

        size_t ipt = 1;
        if (DEFORMED)
        {
            ipt *= totPoints;
        }
        const vec_t *jacPtr = reinterpret_cast<const vec_t *>(
            &(m_jac.template GetPtr<MemSpace, ReadOnly>()[m_jac_idx]));
        const vec_t *dfPtr = reinterpret_cast<const vec_t *>(
            &(m_derivFac.template GetPtr<MemSpace, ReadOnly>()[m_df_idx]));
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(inPtr);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(outPtr);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            const bool colldir0 = expPtr->GetBasis(0)->Collocation();
            const bool colldir1 = expPtr->GetBasis(1)->Collocation();
            const bool colldir2 = expPtr->GetBasis(2)->Collocation();
            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBaseHex<DEFORMED>(nquad0, nquad1, nquad2, dfPtr,
                                               df_tmp, m_inSize, tmpIn, tmpPtr);
                IProductHexKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    (const vec_t::vectorType *)tmpPtr[0], DB0, B1, B2, W0, W1,
                    W2, jacPtr, wsp, wsp0, tmpOut, 1.0, false, colldir1,
                    colldir2);
                IProductHexKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    (const vec_t::vectorType *)tmpPtr[1], B0, DB1, B2, W0, W1,
                    W2, jacPtr, wsp, wsp0, tmpOut, 1.0, colldir0, false,
                    colldir2);
                IProductHexKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    (const vec_t::vectorType *)tmpPtr[2], B0, B1, DB2, W0, W1,
                    W2, jacPtr, wsp, wsp0, tmpOut, 1.0, colldir0, colldir1,
                    false);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const bool isModified =
                (expPtr->GetBasisType(0) == LibUtilities::eModified_A);
            const vec_t *F0, *F1, *F1a, *F2;

            F0  = m_Fac0[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            F1  = m_Fac0[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            F1a = m_Fac1[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            F2  = m_Fac1[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, nquad2, dfPtr, df_tmp, m_inSize, F0, F1,
                    F1a, F2, tmpIn, tmpPtr);
                IProductTetKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[0], DB0, B1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                IProductTetKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[1], B0, DB1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                IProductTetKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[2], B0, B1,
                    DB2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            const bool isModified =
                (expPtr->GetBasisType(0) == LibUtilities::eModified_A);
            const vec_t *F0, *F1, *F2;

            F0 = m_Fac0[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            F1 = m_Fac0[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
            F2 = m_Fac1[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, nquad2, dfPtr, df_tmp, m_inSize, F0, F1,
                    nullptr, F2, tmpIn, tmpPtr);
                IProductPyrKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[0], DB0, B1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                IProductPyrKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[1], B0, DB1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                IProductPyrKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[2], B0, B1,
                    DB2, W0, W1, W2, jacPtr, wsp, wsp0, tmpOut);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            std::vector<vec_t, tinysimd::allocator<vec_t>> wsp1(nmodes1);
            const bool isModified =
                (expPtr->GetBasisType(0) == LibUtilities::eModified_A);
            const vec_t *F0, *F2;

            F0 = m_Fac0[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
            F2 = m_Fac1[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

            for (int e = 0; e < m_nElmtGroup; ++e)
            {
                StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(
                    nquad0, nquad1, nquad2, dfPtr, df_tmp, m_inSize, F0,
                    nullptr, nullptr, F2, tmpIn, tmpPtr);
                IProductPrismKernel<false, false, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[0], DB0, B1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, wsp1, tmpOut);
                IProductPrismKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[1], B0, DB1,
                    B2, W0, W1, W2, jacPtr, wsp, wsp0, wsp1, tmpOut);
                IProductPrismKernel<false, true, DEFORMED>(
                    nmodes0, nmodes1, nmodes2, nquad0, nquad1, nquad2,
                    isModified, (const vec_t::vectorType *)tmpPtr[2], B0, B1,
                    DB2, W0, W1, W2, jacPtr, wsp, wsp0, wsp1, tmpOut);
                tmpIn += totPoints;
                tmpOut += totModes * vec_t::width;
                jacPtr += ipt;
                dfPtr += ipt * ndf;
            }
            m_df_idx += ipt * ndf * m_nElmtGroup * vec_t::width;
        }
    }
};

} // namespace Nektar::Operators::detail
