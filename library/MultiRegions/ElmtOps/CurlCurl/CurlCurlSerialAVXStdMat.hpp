///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlSerialAVXStdMat.hpp
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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp>

#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlSerialAVXStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class CurlCurlBlockOpImpl : public CurlCurlBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    CurlCurlBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : CurlCurlBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        ASSERTL1(m_coordDim == 2 || m_coordDim == 3,
                 "CurlCurl operator only defined for 2D and 3D.");

        ASSERTL1(m_dimension == m_coordDim,
                 "Shape dimension and coordinate dimension are not the same.");

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType, StdRegions::ePhysDerivStdMatTranspose));

        // Fetch derivative factor.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            CurlCurlBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_matptr;
    const TData *m_dfptr;

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        if (inblock.GetNumHomoModes() > 1)
        {
            ApplyCurl3DH1(inblock, outblock);
            return;
        }

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // omega = curl(u) is a scalar in 2D and a vector in 3D.
        const unsigned int nOmega = (m_dimension == 2u) ? 1u : 3u;

        // Get static workspace pointer. The standard derivatives of every
        // component are held at once so that the chain rule and the curl can
        // be applied by a single sweep.
        const auto derivoffset = m_nqTot * simd_t::width;
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                (m_dimension * m_coordDim + nOmega) * derivoffset);
        auto derivptr = wspptr;
        auto omegaptr = derivptr + m_dimension * m_coordDim * derivoffset;

        // Dispatch kernel. The standard derivative matrices of all the
        // directions are stored contiguously, so a multiply of
        // dimension * nqTot columns forms the standard derivatives of one
        // component in every direction at once.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_dimension * m_nqTot, m_nqTot, 1.0, 0.0);

        // Offsets between the components of a block.
        const auto inoffset  = inblock.CompSize();
        const auto outoffset = outblock.CompSize();

        auto dfptr = m_dfptr;

        // Loop over element groups.
        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {
            // Reshape, if necessary.
            if (e % width_ratio == 0)
            {
                for (unsigned int c = 0; c < m_coordDim; ++c)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        m_nqTot, (TData *)inptr + c * inoffset);
                }
            }

            // Step 1: Deriv of u
            // Perform matrix-matrix multiply.
            for (unsigned int c = 0; c < m_coordDim; ++c)
            {
                gemm_kernel(inptr + c * inoffset, m_matptr,
                            derivptr + c * m_dimension * derivoffset);
            }

            // Step 2: omega = curl(u)
            if (m_dimension == 2u)
            {
                // omega_z = dv/dx - du/dy
                if (m_isDeformed)
                {
                    Curl2DScalarStdMatKernel<ExecSpace, true>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(omegaptr));
                }
                else
                {
                    Curl2DScalarStdMatKernel<ExecSpace, false>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(omegaptr));
                }
            }
            else
            {
                if (m_isDeformed)
                {
                    Curl3DStdMatKernel<ExecSpace, true>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(omegaptr),
                        reinterpret_cast<simd_t *>(omegaptr + derivoffset),
                        reinterpret_cast<simd_t *>(omegaptr + 2 * derivoffset));
                }
                else
                {
                    Curl3DStdMatKernel<ExecSpace, false>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(omegaptr),
                        reinterpret_cast<simd_t *>(omegaptr + derivoffset),
                        reinterpret_cast<simd_t *>(omegaptr + 2 * derivoffset));
                }
            }

            // Step 3: Deriv of omega
            // Perform matrix-matrix multiply.
            for (unsigned int c = 0; c < nOmega; ++c)
            {
                gemm_kernel(omegaptr + c * derivoffset, m_matptr,
                            derivptr + c * m_dimension * derivoffset);
            }

            // Step 4: out = curl(omega)
            if (m_dimension == 2u)
            {
                // q = {d(omega_z)/dy, -d(omega_z)/dx}
                if (m_isDeformed)
                {
                    Curl2DVectorStdMatKernel<ExecSpace, true>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr + outoffset));
                }
                else
                {
                    Curl2DVectorStdMatKernel<ExecSpace, false>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr + outoffset));
                }
            }
            else
            {
                if (m_isDeformed)
                {
                    Curl3DStdMatKernel<ExecSpace, true>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr + outoffset),
                        reinterpret_cast<simd_t *>(outptr + 2 * outoffset));
                }
                else
                {
                    Curl3DStdMatKernel<ExecSpace, false>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr + outoffset),
                        reinterpret_cast<simd_t *>(outptr + 2 * outoffset));
                }
            }

            // Reshape back, if necessary.
            if (e % width_ratio == width_ratio - 1)
            {
                for (unsigned int c = 0; c < m_coordDim; ++c)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)inptr + c * inoffset -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        outptr + c * outoffset -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }
            }

            // Increment pointers.
            dfptr += (m_isDeformed)
                         ? m_coordDim * m_dimension * m_nqTot * simd_t::width
                         : m_coordDim * m_dimension * simd_t::width;
            inptr += m_nqTot * simd_t::width;
            outptr += m_nqTot * simd_t::width;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    /// \brief One curl's plane part on each plane of a 3DH1 block: three
    /// components in, three out.
    ///
    /// The curl of a 3DH1 field splits into a part that differentiates in the
    /// plane, \f$(\partial_y f_z, -\partial_x f_z,
    /// \partial_x f_y - \partial_y f_x)\f$, and a part that differentiates
    /// along z. This writes the first; the operator's z-op adds the second,
    /// and CurlCurlOpImpl takes the pair twice, the output carrying omega in
    /// between. Both halves of the plane part are already in the 2D kernels:
    /// the scalar one is the third component, taken from the derivatives of
    /// the first two, and the vector one is the first two components, taken
    /// from the derivatives of the third.
    void ApplyCurl3DH1(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        // A curl needs all three components whichever way round it is taken.
        constexpr unsigned int nComp = 3u;

        ASSERTL1(inblock.GetNumComponents() == nComp &&
                     outblock.GetNumComponents() == nComp,
                 "The homogeneous curl needs all three components");

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Offsets between the components of a block: one component spans all
        // of the planes.
        const auto nhomo     = inblock.GetNumHomoModes();
        const auto inoffset  = inblock.CompSize() * nhomo;
        const auto outoffset = outblock.CompSize() * nhomo;

        // Get static workspace pointer. The standard derivatives of every
        // component are held at once so that the chain rule and the curl can
        // be applied by a single sweep.
        const auto derivoffset = m_nqTot * simd_t::width;
        auto derivptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                nComp * m_dimension * derivoffset);

        // Dispatch kernel. The standard derivative matrices of all the
        // directions are stored contiguously, so a multiply of
        // dimension * nqTot columns forms the standard derivatives of one
        // component in every direction at once.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_dimension * m_nqTot, m_nqTot, 1.0, 0.0);

        // Loop over the planes. Each holds one xy curl, over the same
        // geometry, so the derivative factors restart with every plane while
        // the field pointers run on through the block.
        for (unsigned int p = 0; p < nhomo; ++p)
        {
            auto dfptr = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int c = 0; c < nComp; ++c)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, interleaveWidth, chunkSize,
                            m_nqTot, (TData *)inptr + c * inoffset);
                    }
                }

                // Step 1: standard derivatives of every component.
                for (unsigned int c = 0; c < nComp; ++c)
                {
                    gemm_kernel(inptr + c * inoffset, m_matptr,
                                derivptr + c * m_dimension * derivoffset);
                }

                // Step 2: the third component, df_y/dx - df_x/dy, from the
                // derivatives of the first two.
                if (m_isDeformed)
                {
                    Curl2DScalarStdMatKernel<ExecSpace, true>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(outptr + 2 * outoffset));
                }
                else
                {
                    Curl2DScalarStdMatKernel<ExecSpace, false>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(outptr + 2 * outoffset));
                }

                // Step 3: the first two components, {df_z/dy, -df_z/dx}, from
                // the derivatives of the third.
                if (m_isDeformed)
                {
                    Curl2DVectorStdMatKernel<ExecSpace, true>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(
                            derivptr + 2 * m_dimension * derivoffset),
                        reinterpret_cast<simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr + outoffset));
                }
                else
                {
                    Curl2DVectorStdMatKernel<ExecSpace, false>(
                        m_nqTot, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(
                            derivptr + 2 * m_dimension * derivoffset),
                        reinterpret_cast<simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr + outoffset));
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int c = 0; c < nComp; ++c)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nqTot,
                            (TData *)inptr + c * inoffset -
                                (width_ratio - 1) * m_nqTot * simd_t::width);
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nqTot,
                            outptr + c * outoffset -
                                (width_ratio - 1) * m_nqTot * simd_t::width);
                    }
                }

                // Increment pointers.
                dfptr +=
                    (m_isDeformed)
                        ? m_coordDim * m_dimension * m_nqTot * simd_t::width
                        : m_coordDim * m_dimension * simd_t::width;
                inptr += m_nqTot * simd_t::width;
                outptr += m_nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
