///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialAVXStdMat.hpp
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

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseSerialAVXStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTDerivBaseBlockOpImpl(const unsigned int block_idx,
                                    const LocalRegions::ExpansionSharedPtr &exp,
                                    NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTDerivBaseBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        LibUtilities::PointsType nodalType =
            (exp->IsNodalNonTensorialExp())
                ? exp->GetNodalPointsKey().GetPointsType()
                : LibUtilities::eNoPointsType;

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdMatKey<TData>(basisKeys, m_shapeType,
                             eIProductWRTDerivBaseStdMatTranspose, nodalType));
        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            DerivFactorKey<TData>(block_idx, m_implInterleaveWidth, false));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            IProductWRTDerivBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
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
    const TData *m_jacptr;
    const TData *m_dfptr;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Coeff> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = this->m_append
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Get workspace.
        auto wspptr = this->template GetWorkSpace<MemSpace>(
            m_dimension * simd_t::width * m_nqTot);

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 1.0);

        // Loop over components.
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto inoffset_vec =
            inblock.CompSize() * inblock.GetNumHomoModes() / simd_t::width;
        const auto wspsize = m_nqTot;
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, interleaveWidth, chunkSize,
                            m_nqTot, (TData *)inptr + k * inoffset);
                    }
                }

                // Multiply by derivative factor and Jacobian.
                if (m_isDeformed)
                {
                    MultiplyByJacobianAndDerivFactorKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, 1, inoffset_vec,
                        wspsize, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(wspptr));
                    jacptr += m_nqTot * simd_t::width;
                    dfptr += m_coordDim * m_dimension * m_nqTot * simd_t::width;
                }
                else
                {
                    MultiplyByJacobianAndDerivFactorKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, 1, inoffset_vec,
                        wspsize, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(wspptr));
                    jacptr += simd_t::width;
                    dfptr += m_coordDim * m_dimension * simd_t::width;
                }

                // Perform matrix-matrix multiply.
                if (!this->m_append)
                {
                    for (unsigned int q = 0; q < m_nmTot; ++q)
                    {
                        reinterpret_cast<simd_t *>(outptr)[q] = 0.0;
                    }
                }

                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    gemm_kernel(wspptr + d * simd_t::width * m_nqTot,
                                m_matptr + d * m_nqTot * m_nmTot, outptr);
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nqTot,
                            (TData *)inptr + k * inoffset -
                                (width_ratio - 1) * m_nqTot * simd_t::width);
                    }
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nmTot * simd_t::width);
                }

                // Increment pointers.
                inptr += m_nqTot * simd_t::width;
                outptr += m_nmTot * simd_t::width;
            }

            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (m_coordDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
