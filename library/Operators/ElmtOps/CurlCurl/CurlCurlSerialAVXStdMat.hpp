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

#include "Operators/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXStdMatKernels.hpp"

namespace Nektar::Operators::detail
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
                        NekDataWarehouseSharedPtr dataWarehouse)
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

        m_matptr = dataWarehouse->template GetData<MemSpace>(StdMatKey<TData>(
            basisKeys, m_shapeType, ePhysDerivStdMatTranspose));

        // Fetch derivative factor.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            DerivFactorKey<TData>(block_idx, m_implInterleaveWidth, false));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
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

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                6 * m_dimension * simd_t::width * m_nqTot);

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nqTot, 1.0, 0.0);

        // Loop over components.
        const auto inoffset  = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();

        auto dfptr = m_dfptr;

        auto derivRef = wspptr;
        auto grad0    = reinterpret_cast<simd_t *>(
            derivRef + m_dimension * m_nqTot * simd_t::width);
        auto grad1     = grad0 + m_coordDim * m_nqTot;
        auto grad2     = grad1 + m_coordDim * m_nqTot;
        auto omega     = grad2 + m_coordDim * m_nqTot;
        auto gradOmega = omega + m_coordDim * m_nqTot;

        auto computePhysDeriv = [&](const TData *fieldptr, simd_t *physOut) {
            for (unsigned int d = 0; d < m_dimension; ++d)
            {
                gemm_kernel(fieldptr, m_matptr + d * m_nqTot * m_nqTot,
                            derivRef + d * m_nqTot * simd_t::width);
            }

            if (m_isDeformed)
            {
                MultiplyByDerivFactorKernel<ExecSpace, true>(
                    m_nqTot, m_coordDim, m_dimension, 1, m_nqTot, m_nqTot,
                    reinterpret_cast<const simd_t *>(dfptr),
                    reinterpret_cast<const simd_t *>(derivRef), physOut);
            }
            else
            {
                MultiplyByDerivFactorKernel<ExecSpace, false>(
                    m_nqTot, m_coordDim, m_dimension, 1, m_nqTot, m_nqTot,
                    reinterpret_cast<const simd_t *>(dfptr),
                    reinterpret_cast<const simd_t *>(derivRef), physOut);
            }
        };

        // Loop over element groups.
        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {
            auto inptr0  = inptr;
            auto inptr1  = inptr0 + inoffset;
            auto inptr2  = inptr1 + inoffset;
            auto outptr0 = outblock.template GetPtr<MemSpace, WriteOnly>() +
                           e * m_nqTot * simd_t::width;
            auto outptr1 = outptr0 + outoffset;
            auto outptr2 = outptr1 + outoffset;

            // Reshape, if necessary.
            if (e % width_ratio == 0)
            {
                ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                          interleaveWidth, chunkSize, m_nqTot,
                                          (TData *)inptr0);
                ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                          interleaveWidth, chunkSize, m_nqTot,
                                          (TData *)inptr1);
                if (m_dimension == 3)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize,
                                              m_nqTot, (TData *)inptr2);
                }
            }

            computePhysDeriv(inptr0, grad0);
            computePhysDeriv(inptr1, grad1);

            if (m_dimension == 2)
            {
                auto outsimd0 = reinterpret_cast<simd_t *>(outptr0);
                auto outsimd1 = reinterpret_cast<simd_t *>(outptr1);

                for (unsigned int i = 0; i < m_nqTot; ++i)
                {
                    omega[i] = grad1[i] - grad0[m_nqTot + i];
                }

                computePhysDeriv(reinterpret_cast<const TData *>(omega),
                                 gradOmega);

                for (unsigned int i = 0; i < m_nqTot; ++i)
                {
                    outsimd0[i] = gradOmega[m_nqTot + i];
                    outsimd1[i] = -gradOmega[i];
                }
            }
            else
            {
                auto outsimd0 = reinterpret_cast<simd_t *>(outptr0);
                auto outsimd1 = reinterpret_cast<simd_t *>(outptr1);
                auto outsimd2 = reinterpret_cast<simd_t *>(outptr2);

                computePhysDeriv(inptr2, grad2);

                for (unsigned int i = 0; i < m_nqTot; ++i)
                {
                    omega[i] = grad2[m_nqTot + i] - grad1[2 * m_nqTot + i];
                    omega[m_nqTot + i]     = grad0[2 * m_nqTot + i] - grad2[i];
                    omega[2 * m_nqTot + i] = grad1[i] - grad0[m_nqTot + i];
                }

                computePhysDeriv(reinterpret_cast<const TData *>(omega), grad0);
                computePhysDeriv(
                    reinterpret_cast<const TData *>(omega + m_nqTot), grad1);
                computePhysDeriv(
                    reinterpret_cast<const TData *>(omega + 2 * m_nqTot),
                    grad2);

                for (unsigned int i = 0; i < m_nqTot; ++i)
                {
                    outsimd0[i] = grad2[m_nqTot + i] - grad1[2 * m_nqTot + i];
                    outsimd1[i] = grad0[2 * m_nqTot + i] - grad2[i];
                    outsimd2[i] = grad1[i] - grad0[m_nqTot + i];
                }
            }

            // Reshape back, if necessary.
            if (e % width_ratio == width_ratio - 1)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, m_nqTot,
                    (TData *)inptr0 -
                        (width_ratio - 1) * m_nqTot * simd_t::width);
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, m_nqTot,
                    (TData *)inptr1 -
                        (width_ratio - 1) * m_nqTot * simd_t::width);
                if (m_dimension == 3)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)inptr2 -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }

                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, m_nqTot,
                    (TData *)outptr0 -
                        (width_ratio - 1) * m_nqTot * simd_t::width);
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, m_nqTot,
                    (TData *)outptr1 -
                        (width_ratio - 1) * m_nqTot * simd_t::width);
                if (m_dimension == 3)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)outptr2 -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }
            }

            // Increment pointer.
            dfptr += (m_isDeformed)
                         ? m_coordDim * m_dimension * m_nqTot * simd_t::width
                         : m_coordDim * m_dimension * simd_t::width;
            inptr += m_nqTot * simd_t::width;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
