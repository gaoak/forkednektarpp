///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSerialStdMatXSMM.hpp
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

#include "Operators/ElmtOps/IProductWRTBase/OperatorIProductWRTBase.hpp"
#include "Operators/Utils/UtilsKernels.hpp"
#include <StdRegions/StdExpansion.h>

#include "Operators/Utils/LibXSMMDispatchWrapper.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorIProductWRTBaseImpl
    : public BlockOperatorIProductWRTBase<TData>
{
    using simd_t   = typename simd_type_if<true, TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorIProductWRTBaseImpl(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorIProductWRTBase<TData>(exp, dataWarehouse)
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

        LibUtilities::PointsType nodalType = LibUtilities::eNoPointsType;
        if (exp->IsNodalNonTensorialExp())
        {
            nodalType = exp->GetNodalPointsKey().GetPointsType();
        }

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
            BlockOperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
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
        // const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto input  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto inptr  = reinterpret_cast<const simd_t *>(input);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio              = (interleave_width == 1)
                                                  ? 1
                                                  : interleave_width / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

        // Fetch Jacobian.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Allocate workspace : just 1 element
        // m_wsp.resize(m_implInterleaveWidth * m_nqTot);
        // Get workspace pointer.
        // auto wspptr = m_wsp.data();
        m_wsp = MemoryRegion<TData>::Create(
            "IProd wsp", m_implInterleaveWidth * m_nqTot, ExecSpace::alignment);
        auto wspptr     = m_wsp.template GetPtr<MemSpace, ReadWrite>();
        auto wspsimdptr = reinterpret_cast<simd_t *>(wspptr);

        // libxsmm parameters
        TData alpha  = this->m_scale;
        TData beta   = 0.0;
        int flags    = 0;
        int prefetch = LIBXSMM_PREFETCH_NONE;

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            static_cast<int>(m_implInterleaveWidth), static_cast<int>(m_nmTot),
            static_cast<int>(m_nqTot), alpha, beta, flags, prefetch);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr = reinterpret_cast<const simd_t *>(jacptr_init);
            if (m_isDeformed)
            {
                for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, inblock.GetNumData(),
                            (TData *)inptr);
                    }

                    // Multiply by Jacobian for this element
                    // and launch gemm kernel
                    for (unsigned int q = 0; q < m_nqTot; ++q)
                    {
                        wspsimdptr[q] = jacptr[q] * inptr[q];
                    }

                    // Perform matrix-matrix multiply.
                    gemm_kernel(wspptr, m_matptr, outptr);

                    inptr += m_nqTot;
                    outptr += m_nmTot * m_implInterleaveWidth;
                    jacptr += m_nqTot;
                }
            }
            else // regular
            {
                for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, inblock.GetNumData(),
                            (TData *)inptr);
                    }
                    // Multiply by Jacobian for this element
                    // and launch gemm kernel
                    for (unsigned int q = 0; q < m_nqTot; ++q)
                    {
                        wspsimdptr[q] = jacptr[0] * inptr[q];
                    }
                    // Perform matrix-matrix multiply.
                    gemm_kernel(wspptr, m_matptr, outptr);

                    inptr += m_nqTot;
                    outptr += m_nmTot * m_implInterleaveWidth;
                    ++jacptr;
                }
            }
        }
    }
};

} // namespace Nektar::Operators::detail
