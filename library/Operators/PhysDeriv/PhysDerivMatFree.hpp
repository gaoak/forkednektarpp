///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivMatFree.hpp
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

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/Basis.h>
#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "Operators/OperatorPhysDeriv.hpp"
#include "PhysDerivMatFreeKernels.hpp"

namespace Nektar::Operators::detail
{

using vec_t = tinysimd::simd<double>;

// Matrix-free implementation
template <typename TData>
class OperatorPhysDerivImpl<TData, ImplMatFree>
    : public OperatorPhysDeriv<TData>
{
public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
        // Initialise jacobian with paddings
        auto blocks =
            GetBlockAttributes(FieldState::Phys, expansionList, vec_t::width);
        size_t jacSize = Operator<TData>::GetVectorizedGeomFactorSize(blocks);
        m_df = Operator<TData>::SetVectorizedDerivFactor(jacSize, blocks);
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
        in.template ReshapeStorage<vec_t::width>();
        out.template ReshapeStorage<vec_t::width>();

        TData *inptr       = in.GetStorage().GetCPUPtr();
        auto const Coordim = this->m_expansionList->GetExp(0)->GetCoordim();
        ASSERTL0(Coordim <= out.GetNumComponents(),
                 "Output field has fewer components than the coordinate!");
        // WARNINGL0(Coordim == out.GetNumComponents(),
        //           "Output field has more components than the coordinate!");
        // GetStorage() returns *m_storage, not shared_ptr
        size_t scalar_field_size =
            out.GetStorage().size() / out.GetNumComponents();
        std::vector<TData *> outptr;
        outptr.resize(Coordim);
        for (int d = 0; d < Coordim; ++d)
        {
            outptr[d] = out.GetStorage().GetCPUPtr() + d * scalar_field_size;
        }

        size_t exp_idx = 0; // accumulates over blocks, not used in operatorND()
        m_jac_idx      = 0; // accumulates over blocks, accessed in operatorND()

        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto const expPtr    = this->m_expansionList->GetExp(exp_idx);
            auto const &inblock  = in.GetBlocks()[block_idx];
            auto const &outblock = out.GetBlocks()[block_idx];
            auto const shapeType = expPtr->DetShapeType();
            auto const dimension = expPtr->GetShapeDimension();
            auto const deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            m_nElmtGroup =
                (inblock.num_elements + inblock.num_padding_elements) /
                vec_t::width;

            m_basis.resize(dimension);
            for (int i = 0; i < dimension; ++i)
            {
                m_basis[i] = expPtr->GetBasis(i);
                // prepare D data
                auto D = m_basis[i]->GetD()->GetPtr();
                m_D[i].resize(D.size());
                for (int j = 0; j < D.size(); ++j)
                {
                    m_D[i][j] = D[j];
                }
                // prepare Z data
                auto Z = m_basis[i]->GetZ();
                m_Z[i].resize(Z.size());
                for (int j = 0; j < Z.size(); ++j)
                {
                    m_Z[i][j] = Z[j];
                }
            }

#include "SwitchNodesPoints.h"

            inptr += inblock.block_size;
            for (int d = 0; d < Coordim; ++d)
            {
                outptr[d] += outblock.block_size;
            }
            exp_idx += inblock.num_elements;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorPhysDerivImpl<TData, ImplMatFree>>(
            expansionList);
    }

    static std::string className;

private:
    std::vector<LibUtilities::BasisSharedPtr> m_basis;
    int m_nElmtGroup, m_jac_idx;
    std::shared_ptr<std::vector<vec_t, tinysimd::allocator<vec_t>>>
        m_df; // Derivative factors
    std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3>
        m_D; // Derivative matrix
    std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3> m_Z; // Zeroes

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator1D(const NekDouble *input, std::vector<NekDouble *> output)
    {
        const auto nq0     = m_basis[0]->GetNumPoints();
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
        const vec_t *df_ptr = &((*this->m_df)[m_jac_idx * ndf]);

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
            PhysDerivTensor1DKernel(nq0, tmpIn, this->m_D[0], tmpOut[0]);

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
        if constexpr (DEFORMED)
        {
            m_jac_idx += nqTot * m_nElmtGroup;
        }
        else
        {
            m_jac_idx += m_nElmtGroup;
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
        const vec_t *df_ptr = &((*this->m_df)[m_jac_idx * ndf]);

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
            PhysDerivTensor1DKernel(nq0, tmpIn, this->m_D[0], tmpOut[0]);

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
        if constexpr (DEFORMED)
        {
            m_jac_idx += nqTot * m_nElmtGroup;
        }
        else
        {
            m_jac_idx += m_nElmtGroup;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator2D(const NekDouble *input, std::vector<NekDouble *> output)
    {
        const auto nq0 = m_basis[0]->GetNumPoints();
        const auto nq1 = m_basis[1]->GetNumPoints();

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
        const vec_t *df_ptr = &((*this->m_df)[m_jac_idx * ndf]);

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
            PhysDerivTensor2DKernel(nq0, nq1, tmpIn, this->m_D[0], this->m_D[1],
                                    tmpOut[0], tmpOut[1]);
            // Calculate physical derivative
            PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nCoord, this->m_Z[0], this->m_Z[1], df_ptr, tmpOut);
            // Increment pointers
            df_ptr += dfsize;
            tmpIn += nqTot;
            for (int d = 0; d < nCoord; ++d) // automatically unrolled
            {
                tmpOut[d] += nqBlock;
            }
        }
        if constexpr (DEFORMED)
        {
            m_jac_idx += nqTot * m_nElmtGroup;
        }
        else
        {
            m_jac_idx += m_nElmtGroup;
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
        const vec_t *df_ptr = &((*this->m_df)[m_jac_idx * ndf]);

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
            PhysDerivTensor2DKernel(nq0, nq1, tmpIn, this->m_D[0], this->m_D[1],
                                    tmpOut[0], tmpOut[1]);
            // Calculate physical derivative
            PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nCoord, this->m_Z[0], this->m_Z[1], df_ptr, tmpOut);
            // Increment pointers
            df_ptr += dfsize;
            tmpIn += nqTot;
            for (int d = 0; d < nCoord; ++d) // automatically unrolled
            {
                tmpOut[d] += nqBlock;
            }
        }
        if constexpr (DEFORMED)
        {
            m_jac_idx += nqTot * m_nElmtGroup;
        }
        else
        {
            m_jac_idx += m_nElmtGroup;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator3D(const NekDouble *input, std::vector<NekDouble *> output)
    {
        const auto nq0 = m_basis[0]->GetNumPoints();
        const auto nq1 = m_basis[1]->GetNumPoints();
        const auto nq2 = m_basis[2]->GetNumPoints();

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
        const vec_t *df_ptr = &((*this->m_df)[m_jac_idx * ndf]);

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        tmpOut[0] = reinterpret_cast<vec_t::scalarType *>(output[0]);
        tmpOut[1] = reinterpret_cast<vec_t::scalarType *>(output[1]);
        tmpOut[2] = reinterpret_cast<vec_t::scalarType *>(output[2]);

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            PhysDerivTensor3DKernel(nq0, nq1, nq2, tmpIn, this->m_D[0],
                                    this->m_D[1], this->m_D[2], tmpOut[0],
                                    tmpOut[1], tmpOut[2]);
            // Calculate physical derivative
            PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, this->m_Z[0], this->m_Z[1], this->m_Z[2], df_ptr,
                wsp0, wsp1, tmpOut[0], tmpOut[1], tmpOut[2]);

            // Increment pointers
            df_ptr += dfsize;
            tmpIn += nqTot;
            tmpOut[0] += nqBlocks;
            tmpOut[1] += nqBlocks;
            tmpOut[2] += nqBlocks;
        }
        if constexpr (DEFORMED)
        {
            m_jac_idx += nqTot * m_nElmtGroup;
        }
        else
        {
            m_jac_idx += m_nElmtGroup;
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
        const vec_t *df_ptr = &((*this->m_df)[m_jac_idx * ndf]);

        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);

        vec_t::scalarType *tmpOut[3];
        tmpOut[0] = reinterpret_cast<vec_t::scalarType *>(output[0]);
        tmpOut[1] = reinterpret_cast<vec_t::scalarType *>(output[1]);
        tmpOut[2] = reinterpret_cast<vec_t::scalarType *>(output[2]);

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            PhysDerivTensor3DKernel(nq0, nq1, nq2, tmpIn, this->m_D[0],
                                    this->m_D[1], this->m_D[2], tmpOut[0],
                                    tmpOut[1], tmpOut[2]);
            // Calculate physical derivative
            PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, this->m_Z[0], this->m_Z[1], this->m_Z[2], df_ptr,
                wsp0, wsp1, tmpOut[0], tmpOut[1], tmpOut[2]);

            // Increment pointers
            df_ptr += dfsize;
            tmpIn += nqTot;
            tmpOut[0] += nqBlocks;
            tmpOut[1] += nqBlocks;
            tmpOut[2] += nqBlocks;
        }
        if constexpr (DEFORMED)
        {
            m_jac_idx += nqTot * m_nElmtGroup;
        }
        else
        {
            m_jac_idx += m_nElmtGroup;
        }
    }
};

} // namespace Nektar::Operators::detail
