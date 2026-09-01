///////////////////////////////////////////////////////////////////////////////
//
// File: MaxStdVelocitySerialAVX.hpp
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
// Description: MaxStdVelocity, Serial and AVX implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp"
#include "Operators/CFL/MaxStdVelocity/MaxStdVelocityBlockOp.hpp"
#include "Operators/CFL/MaxStdVelocity/MaxStdVelocitySerialAVXKernels.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Serial and AVX implementation of MaxStdVelocityBlockOp.
 *
 * Works in the vector type, one simd lane per element, as the Norm
 * operators do: each element group is reshaped to the implementation's
 * width if the field is stored at another, padded lanes are zeroed inline,
 * and the kernel's per-lane maxima are horizontally reduced at the end.
 */
template <typename ExecSpace, typename TData>
class MaxStdVelocityBlockOpImpl : public MaxStdVelocityBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    MaxStdVelocityBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : MaxStdVelocityBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_nqTot     = exp->GetTotPoints();
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // The velocity is contracted with the derivative factors in the
        // untransposed layout, which stores them point by point, at this
        // implementation's width.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));
    }

    static std::string className;

    static std::unique_ptr<MaxStdVelocityBlockOp<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<MaxStdVelocityBlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    bool m_isDeformed;
    unsigned int m_nqTot;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    const TData *m_dfptr;

#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI
    // system is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::MemoryRegion<TData> &data) override
    {
        WARNINGL1(m_warnOnce ||
                      (inblock.GetAlignment() % simd_t::alignment == 0),
                  "Input Field is not aligned to the required alignment "
                  "for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif

        if (m_isDeformed)
        {
            ApplyImpl<true>(inblock, data);
        }
        else
        {
            ApplyImpl<false>(inblock, data);
        }
    }

    template <bool DEFORMED>
    void ApplyImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::MemoryRegion<TData> &data)
    {
        const auto nelmt    = inblock.GetNumElements();
        const auto nqTot    = inblock.GetNumData();
        const auto compSize = inblock.CompSize();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Initialize pointers.
        auto inptr   = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto dataptr = data.template GetPtr<MemSpace, ReadWrite>();

        const TData soundSpeedFactor = this->m_soundSpeedFactor;
        // The components the kernel reads: the velocity and, when weighted,
        // the wave speed after them.
        const unsigned int nComp =
            m_coordDim + ((soundSpeedFactor != TData(0)) ? 1u : 0u);
        const size_t compVecStride = compSize / simd_t::width;

        const size_t dfSize =
            (DEFORMED ? size_t(nqTot) * m_coordDim * m_dimension
                      : size_t(m_coordDim) * m_dimension);

        auto dfptr = m_dfptr;

        simd_t acc = 0.0;
        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {
            // Reshape the components the kernel reads, if necessary.
            if (e % width_ratio == 0)
            {
                for (unsigned int c = 0; c < nComp; ++c)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        nqTot, (TData *)inptr + c * compSize);
                }
            }

            // Zeroing padding elements.
            for (unsigned int c = 0; c < nComp; ++c)
            {
                for (unsigned int q = 0; q < nqTot; ++q)
                {
                    const auto offset =
                        c * compSize + q * m_implInterleaveWidth;
                    for (unsigned int i = 0; i < m_implInterleaveWidth; ++i)
                    {
                        if (e * m_implInterleaveWidth + i >= nelmt)
                        {
                            ((TData *)inptr)[offset + i] = 0.0;
                        }
                    }
                }
            }

            acc = max(acc, MaxStdVelocityKernel<DEFORMED>(
                               nqTot, m_dimension, m_coordDim, compVecStride,
                               soundSpeedFactor,
                               reinterpret_cast<const simd_t *>(dfptr),
                               reinterpret_cast<const simd_t *>(inptr)));

            // Reshape back, if necessary.
            if (e % width_ratio == width_ratio - 1)
            {
                for (unsigned int c = 0; c < nComp; ++c)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)inptr + c * compSize -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }
            }

            // Increment pointers for the next elmt group.
            inptr += nqTot * simd_t::width;
            dfptr += dfSize * simd_t::width;
        }

        // Accumulate over vector width.
        for (unsigned int i = 0; i < simd_t::width; ++i)
        {
            dataptr[this->m_block_idx] =
                std::max(dataptr[this->m_block_idx], acc[i]);
        }
    }
};

} // namespace Nektar::Operators::detail
