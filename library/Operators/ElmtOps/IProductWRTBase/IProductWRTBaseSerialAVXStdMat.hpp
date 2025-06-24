///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSerialAVXStdMat.hpp
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

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTBaseBlockOpImpl : public IProductWRTBaseBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTBaseBlockOpImpl(const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTBaseBlockOp<TData>(exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
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

        m_matptr = dataWarehouse->template GetData<ExecSpace>(
            StdMatKey<TData>(basisKeys, m_shapeType,
                             eIProductWRTBaseStdMatTranspose, nodalType));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            IProductWRTBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
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
    MemoryRegion<TData> m_wsp;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Get interleave parameter.
        const auto interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio      = (interleave_width == 1)
                                          ? 1
                                          : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Allocate storage.
        if (m_wsp.size() == 0)
        {
            m_wsp = MemoryRegion<TData>::Create(simd_t::width * m_nqTot,
                                                ExecSpace::alignment);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 0.0);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr = jacptr_init;

            for (size_t e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                        interleave_width, chunkSize, m_nqTot, (TData *)inptr);
                }

                // Multiply by jacobian.
                if (m_isDeformed)
                {
                    MultiplyByJacobianKernel<ExecSpace, true>(
                        m_nqTot, 1, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(wspptr), this->m_scale);
                    jacptr += m_nqTot * simd_t::width;
                }
                else
                {
                    MultiplyByJacobianKernel<ExecSpace, false>(
                        m_nqTot, 1, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(wspptr), this->m_scale);
                    jacptr += simd_t::width;
                }

                // Perform matrix-matrix multiply.
                gemm_kernel(wspptr, m_matptr, outptr);

                // Increment pointers.
                inptr += m_nqTot * simd_t::width;
                outptr += m_nmTot * simd_t::width;
            }
        }
    }
};

} // namespace Nektar::Operators::detail
