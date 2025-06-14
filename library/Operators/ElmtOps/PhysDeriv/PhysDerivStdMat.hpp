///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivStdMat.hpp
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

#include "Operators/ElmtOps/PhysDeriv/OperatorPhysDeriv.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorPhysDerivImpl : public BlockOperatorPhysDeriv<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorPhysDerivImpl(const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorPhysDeriv<TData>(exp, dataWarehouse)
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

        // Specialization for AVX/libXSMM
        if constexpr (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
        {
            m_matptr =
                dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
                    basisKeys, m_shapeType, ePhysDerivStdMatTranspose));
        }
        else
        {
            m_matptr = dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType, ePhysDerivStdMat));
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BlockOperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
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

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        auto handle = NekHandle<ExecSpace>::GetInstance();

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch derivative factor.
        constexpr bool transpose =
            std::is_same_v<ExecSpace, NektarSpaces::Device>;
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), transpose));

        // Specialization for AVX/libXSMM
        if constexpr (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
        {
            // Get interleave parameter.
            const auto interleave_width = inblock.GetInterleaveWidth();
            const auto width_ratio =
                (interleave_width == 1)
                    ? 1
                    : interleave_width / m_implInterleaveWidth;
            const auto chunkSize =
                std::max(m_implInterleaveWidth, interleave_width);

            const auto outsize = m_nqTot * (nelmt / simd_t::width);
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                auto dfptr = dfptr_init;

                const TData alpha  = 1.0;
                const TData beta   = 0.0;
                const int flags    = 0;
                const int prefetch = LIBXSMM_PREFETCH_NONE;

                // Dispatch kernel.
                auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
                    static_cast<int>(simd_t::width), static_cast<int>(m_nqTot),
                    static_cast<int>(m_nqTot), alpha, beta, flags, prefetch);

                for (size_t e = 0; e < nelmt / simd_t::width; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, m_nqTot,
                            (TData *)inptr);
                    }

                    // Perform matrix-matrix multiply.
                    for (unsigned int d = 0; d < m_dimension; d++)
                    {
                        gemm_kernel(inptr, m_matptr + d * m_nqTot * m_nqTot,
                                    outptr + d * m_nqTot * nelmt);
                    }

                    // Multiply by derivative factor.
                    if (m_isDeformed)
                    {
                        MultiplyByDerivFactorKernel<ExecSpace, true>(
                            m_nqTot, m_coordDim, m_dimension, 1, outsize,
                            outsize, reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(outptr),
                            reinterpret_cast<simd_t *>(outptr));
                        dfptr +=
                            m_coordDim * m_dimension * m_nqTot * simd_t::width;
                    }
                    else
                    {
                        MultiplyByDerivFactorKernel<ExecSpace, false>(
                            m_nqTot, m_coordDim, m_dimension, 1, outsize,
                            outsize, reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(outptr),
                            reinterpret_cast<simd_t *>(outptr));
                        dfptr += m_coordDim * m_dimension * simd_t::width;
                    }

                    // Increment pointer.
                    inptr += m_nqTot * simd_t::width;
                    outptr += m_nqTot * simd_t::width;
                }
                outptr += (m_coordDim - 1) * outblock.size();
            }
        }
        else
        {
            const auto outsize = m_nqTot * nelmt;
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                auto dfptr = dfptr_init;

                const TData alpha = 1.0;
                const TData beta  = 0.0;

                // Reshape, if necessary.
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(),
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    (TData *)inptr);

                // Perform matrix-matrix multiply.
                NekGemmStridedBatched(
                    handle, "N", "N", m_nqTot, nelmt, m_nqTot, alpha, m_matptr,
                    m_nqTot, m_nqTot * m_nqTot, inptr, m_nqTot, 0, beta, outptr,
                    m_nqTot, m_nqTot * nelmt, m_dimension);

                // Multiply by derivative factor.
                if (m_isDeformed)
                {
                    MultiplyByDerivFactorKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, outsize,
                        outsize, dfptr, outptr, outptr);
                }
                else
                {
                    MultiplyByDerivFactorKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, outsize,
                        outsize, dfptr, outptr, outptr);
                }

                // Increment pointer.
                inptr += inblock.size();
                outptr += m_coordDim * outblock.size();
            }
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
