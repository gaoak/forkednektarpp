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

#include "Common/OperatorHelper.hpp"
#include "ElmtOps/OperatorPhysDeriv.hpp"

#include "ElmtOps/PhysDeriv/PhysDerivAVXSumFacKernels.hpp"

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

        // size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        // Array<OneD, Array<OneD, TData>> derivFac =
        //     Operator<TData>::SetDerivativeFactor(jacSize);

        size_t dfSize = Operator<TData>::GetGeometricFactorSize(blocks);
        std::shared_ptr<VecVec_t> derivFac =
            Operator<TData>::SetDerivativeFactor(dfSize, blocks);

        m_derivFac = MemoryRegion<vec_t>::template fromVector<MemSpace, vec_t>(
            *derivFac, vec_t::alignment);

        // Initialize the points.
        m_pointMap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_POINT_DATA, vec_t::alignment);

        // Initialize the derivative matrix.
        m_derivativeMap = GetBasisData<MemSpace, TData, vec_t>(
            expansionList, BASIS_DERIVATIVE_DATA, vec_t::alignment);
    }

    void apply(Field<TData, FieldState::Phys> &in,
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
        auto *outOrig     = out.template GetPtr<MemSpace, ReadWrite>();

        const auto Coordim = this->m_expansionList->GetExp(0)->GetCoordim();
        ASSERTL0(Coordim <= out.GetNumComponents(),
                 "Output field has fewer components than the coordinate!");
        // WARNINGL0(Coordim == out.GetNumComponents(),
        //           "Output field has more components than the coordinate!");

        std::vector<TData *> outPtr;
        outPtr.resize(Coordim);

        size_t scalar_field_size = out.size() / out.GetNumComponents();

        for (int d = 0; d < Coordim; ++d)
        {
            outPtr[d] = outOrig + d * scalar_field_size;
        }

        m_exp_idx = 0; // accumulates over blocks, also used in operatorND()
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

            m_nElmtGroup = inblock.num_elmt_groups;

            // Fetch basis key for the current element type.
            m_basisKeys.clear();
            for (size_t d = 0; d < dimension; ++d)
            {
                // m_basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
                m_basisKeys.push_back(expPtr->GetBasis(d)->GetBasisKey());
            }

#include "Operators/Common/SwitchLevel1DeformedCoord.h"

            // Increment pointer and index for next element type.
            inPtr += inblock.block_size;

            for (int d = 0; d < Coordim; ++d)
            {
                outPtr[d] += outblock.block_size;
            }

            if (deformed)
            {
                m_jac_idx += nqTot * m_nElmtGroup;
            }
            else
            {
                m_jac_idx += m_nElmtGroup;
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
    int m_nElmtGroup, m_jac_idx, m_exp_idx;

    MemoryRegion<vec_t> m_derivFac;
    BasisDataMap<vec_t> m_pointMap;
    BasisDataMap<vec_t> m_derivativeMap;
    // std::array<LibUtilities::BasisKey, 3> m_basisKeys;
    std::vector<LibUtilities::BasisKey> m_basisKeys;

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator1D(const NekDouble *input, std::vector<NekDouble *> output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nq0     = expPtr->GetNumPoints(0);
        const auto nqTot   = nq0;
        const auto nqBlock = nqTot * vec_t::width;

        const auto nCoord = output.size();
        const auto ndf    = nCoord;
        int dfsize        = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Get derivative factor pointer
        const vec_t *dfPtr  = m_derivFac.template GetPtr<MemSpace, ReadOnly>();
        const vec_t *df_ptr = &(dfPtr[m_jac_idx * ndf]);

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        // const auto Z0 = m_pointMap[m_basisKeys[0]].template GetPtr<MemSpace,
        // ReadOnly>();
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        for (int d = 0; d < nCoord; ++d)
        {
            tmpOut[d] = reinterpret_cast<vec_t::scalarType *>(output[d]);
        }

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Get the basic derivative
            PhysDerivTensor1DKernel(nq0, tmpIn, D0, tmpOut[0]);

            // Calculate physical derivative
            PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(nq0, nCoord, df_ptr,
                                                    tmpOut);

            // Increment pointers
            df_ptr += dfsize;
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
    void operator1D(const NekDouble *input, std::vector<NekDouble *> output)
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
        const vec_t *dfPtr  = m_derivFac.template GetPtr<MemSpace, ReadOnly>();
        const vec_t *df_ptr = &(dfPtr[m_jac_idx * ndf]);

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        // const auto Z0 = m_pointMap[m_basisKeys[0]].template GetPtr<MemSpace,
        // ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        for (int d = 0; d < nCoord; ++d)
        {
            tmpOut[d] = reinterpret_cast<vec_t::scalarType *>(output[d]);
        }

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Get the basic derivative
            PhysDerivTensor1DKernel(nq0, tmpIn, D0, tmpOut[0]);

            // Calculate physical derivative
            PhysDeriv1DKernel<SHAPE_TYPE, DEFORMED>(nq0, nCoord, df_ptr,
                                                    tmpOut);

            // Increment pointers
            df_ptr += dfsize;
            tmpIn += nqTot;
            for (int d = 0; d < nCoord; ++d) // automatically unrolled
            {
                tmpOut[d] += nqBlock;
            }
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator2D(const NekDouble *input, std::vector<NekDouble *> output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);

        const auto nqTot   = nq0 * nq1;
        const auto nqBlock = nqTot * vec_t::width;

        const auto nCoord = output.size();
        const auto ndf    = 2 * nCoord;
        int dfsize        = ndf;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Get derivative factor pointer
        const vec_t *dfPtr  = m_derivFac.template GetPtr<MemSpace, ReadOnly>();
        const vec_t *df_ptr = &(dfPtr[m_jac_idx * ndf]);

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D1 = m_derivativeMap[m_basisKeys[1]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto Z0 =
            m_pointMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z1 =
            m_pointMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        for (int d = 0; d < nCoord; ++d)
        {
            tmpOut[d] = reinterpret_cast<vec_t::scalarType *>(output[d]);
        }

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Results written to tmpOut0, tmpOut1
            PhysDerivTensor2DKernel(nq0, nq1, tmpIn, D0, D1, tmpOut[0],
                                    tmpOut[1]);
            // Calculate physical derivative
            PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(nq0, nq1, nCoord, Z0, Z1,
                                                    df_ptr, tmpOut);
            // Increment pointers
            df_ptr += dfsize;
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
    void operator2D(const NekDouble *input, std::vector<NekDouble *> output)
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
        const vec_t *dfPtr  = m_derivFac.template GetPtr<MemSpace, ReadOnly>();
        const vec_t *df_ptr = &(dfPtr[m_jac_idx * ndf]);

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D1 = m_derivativeMap[m_basisKeys[1]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto Z0 =
            m_pointMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z1 =
            m_pointMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        for (int d = 0; d < nCoord; ++d)
        {
            tmpOut[d] = reinterpret_cast<vec_t::scalarType *>(output[d]);
        }

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            // Results written to tmpOut0, tmpOut1
            PhysDerivTensor2DKernel(nq0, nq1, tmpIn, D0, D1, tmpOut[0],
                                    tmpOut[1]);
            // Calculate physical derivative
            PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(nq0, nq1, nCoord, Z0, Z1,
                                                    df_ptr, tmpOut);
            // Increment pointers
            df_ptr += dfsize;
            tmpIn += nqTot;
            for (int d = 0; d < nCoord; ++d) // automatically unrolled
            {
                tmpOut[d] += nqBlock;
            }
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator3D(const NekDouble *input, std::vector<NekDouble *> output)
    {
        const auto expPtr = this->m_expansionList->GetExp(m_exp_idx);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);
        const auto nq2 = expPtr->GetNumPoints(2);

        const auto nqTot    = nq0 * nq1 * nq2;
        const auto nqBlocks = nqTot * vec_t::width;

        constexpr auto ndf = 9;
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
        const vec_t *dfPtr  = m_derivFac.template GetPtr<MemSpace, ReadOnly>();
        const vec_t *df_ptr = &(dfPtr[m_jac_idx * ndf]);

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D1 = m_derivativeMap[m_basisKeys[1]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D2 = m_derivativeMap[m_basisKeys[2]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto Z0 =
            m_pointMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z1 =
            m_pointMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z2 =
            m_pointMap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        tmpOut[0] = reinterpret_cast<vec_t::scalarType *>(output[0]);
        tmpOut[1] = reinterpret_cast<vec_t::scalarType *>(output[1]);
        tmpOut[2] = reinterpret_cast<vec_t::scalarType *>(output[2]);

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            PhysDerivTensor3DKernel(nq0, nq1, nq2, tmpIn, D0, D1, D2, tmpOut[0],
                                    tmpOut[1], tmpOut[2]);
            // Calculate physical derivative
            PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, Z0, Z1, Z2, df_ptr, wsp0, wsp1, tmpOut[0],
                tmpOut[1], tmpOut[2]);

            // Increment pointers
            df_ptr += dfsize;
            tmpIn += nqTot;
            tmpOut[0] += nqBlocks;
            tmpOut[1] += nqBlocks;
            tmpOut[2] += nqBlocks;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nq0,
              int nq1, int nq2>
    void operator3D(const NekDouble *input, std::vector<NekDouble *> output)
    {
        constexpr auto nqTot    = nq0 * nq1 * nq2;
        constexpr auto nqBlocks = nqTot * vec_t::width;

        constexpr auto ndf = 9;
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
        const vec_t *dfPtr  = m_derivFac.template GetPtr<MemSpace, ReadOnly>();
        const vec_t *df_ptr = &(dfPtr[m_jac_idx * ndf]);

        const auto D0 = m_derivativeMap[m_basisKeys[0]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D1 = m_derivativeMap[m_basisKeys[1]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto D2 = m_derivativeMap[m_basisKeys[2]]
                            .template GetPtr<MemSpace, ReadOnly>();
        const auto Z0 =
            m_pointMap[m_basisKeys[0]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z1 =
            m_pointMap[m_basisKeys[1]].template GetPtr<MemSpace, ReadOnly>();
        const auto Z2 =
            m_pointMap[m_basisKeys[2]].template GetPtr<MemSpace, ReadOnly>();

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        tmpOut[0] = reinterpret_cast<vec_t::scalarType *>(output[0]);
        tmpOut[1] = reinterpret_cast<vec_t::scalarType *>(output[1]);
        tmpOut[2] = reinterpret_cast<vec_t::scalarType *>(output[2]);

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            PhysDerivTensor3DKernel(nq0, nq1, nq2, tmpIn, D0, D1, D2, tmpOut[0],
                                    tmpOut[1], tmpOut[2]);
            // Calculate physical derivative
            PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, Z0, Z1, Z2, df_ptr, wsp0, wsp1, tmpOut[0],
                tmpOut[1], tmpOut[2]);

            // Increment pointers
            df_ptr += dfsize;
            tmpIn += nqTot;
            tmpOut[0] += nqBlocks;
            tmpOut[1] += nqBlocks;
            tmpOut[2] += nqBlocks;
        }
    }
};

} // namespace Nektar::Operators::detail
