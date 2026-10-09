///////////////////////////////////////////////////////////////////////////////
//
// File: MultiFieldHelper.hpp
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
// Description: Products over the columns of a MultiField.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/Field/MultiField.hpp>
#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp>

#include "SolverCore/GlobalLinSysOps/MultiFieldHelper/MultiFieldKernels.hpp"
#include "SolverCore/GlobalLinSysOps/MultiFieldHelper/MultiFieldWorkSpace.hpp"

#include <algorithm>
#include <cstdint>

namespace Nektar::SolverCore
{

namespace detail
{

/**
 * @brief BLAS handle for @p streamID: the Serial one on host, which also
 * provides the transposed Gemv.
 */
template <typename ExecSpace>
auto GetMultiFieldBlasHandle(const unsigned int streamID)
{
    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
        return NekBlas::Handle<ExecSpace>::GetInstance(streamID);
    }
    else
    {
        return NekBlas::Handle<NektarSpaces::Serial>::GetInstance(streamID);
    }
}

/**
 * @brief Call @p f(js, je) on each run [js, je) of [j0, j1) within one segment
 * of @p numFieldPerSegment fields.
 */
template <typename F>
void ForEachSegment(const unsigned int numFieldPerSegment,
                    const unsigned int j0, const unsigned int j1, F &&f)
{
    for (unsigned int js = j0; js < j1;)
    {
        const unsigned int je =
            std::min(j1, (js / numFieldPerSegment + 1) * numFieldPerSegment);
        f(js, je);
        js = je;
    }
}

/**
 * @brief Pointer to block @p blk of column @p js of @p X, after making block
 * @p blk of columns [js, je), which lie in one segment, current in
 * @p MemSpace.
 */
template <typename MemSpace, typename TData, FieldState TState>
const TData *GetColumnsPtr(LibUtilities::MultiField<TData, TState> &X,
                           const unsigned int blk, const unsigned int js,
                           const unsigned int je, const unsigned int streamID)
{
    const TData *first = nullptr;
    for (unsigned int j = js; j < je; ++j)
    {
        auto &block = X[j].GetBlocks()[blk];
        auto ptr    = block.template GetPtr<MemSpace, ReadOnly>(streamID);
        if (j == js)
        {
            first = ptr;
        }
        ASSERTL1(ptr == first + (j - js) * X.GetFieldSize(),
                 "MultiField - Columns of a segment are not contiguous.");
        ASSERTL1(block.GetInterleaveWidth() ==
                     X[js].GetBlocks()[blk].GetInterleaveWidth(),
                 "MultiField - Inconsistent interleave format between "
                 "columns.");
    }

    return first;
}

/**
 * @brief out[j - j0] = (column j of @p X, field) for j0 <= j < j1, over the
 * entries @p mask selects if given, otherwise padding excluded.
 */
template <typename ExecSpace, typename TData, FieldState TState>
void MultiDotImpl(LibUtilities::Field<std::uint8_t, TState> *mask,
                  LibUtilities::MultiField<TData, TState> &X,
                  const unsigned int j0, const unsigned int j1,
                  LibUtilities::Field<TData, TState> &field, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    ASSERTL1(j0 < j1 && j1 <= X.GetNumField(),
             "MultiField::MultiDot - Invalid column range.");

    const unsigned int ncols   = j1 - j0;
    const unsigned int nblocks = field.GetBlocks().size();
    TData *blockDots           = nullptr;
    if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        blockDots = MultiFieldWorkSpace<TData>::template Get<MemSpace>(
            nblocks * ncols, 0);
    }

    for (unsigned int blk = 0; blk < nblocks; ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto &block = field.GetBlocks()[blk];
        ASSERTL1(block.GetInterleaveWidth() ==
                     X[j0].GetBlocks()[blk].GetInterleaveWidth(),
                 "MultiField::MultiDot - Inconsistent interleave format.");

        const size_t compSize = block.CompSize();
        const unsigned int ncomp =
            block.GetNumComponents() * block.GetNumHomoModes();
        const size_t nrows = compSize * ncomp;

        // Masked copy of block.
        auto ptr = block.template GetPtr<MemSpace, ReadOnly>(streamID);
        auto maskedptr =
            MultiFieldWorkSpace<TData>::template Get<MemSpace>(nrows, streamID);
        if (mask)
        {
            auto maskptr =
                mask->GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID);
            CopyMasked<ExecSpace>(nrows, maskptr, ptr, maskedptr, streamID);
        }
        else
        {
            auto maskptr =
                Math::internalMathKernelMask<MemSpace>::GetInstance(block);
            CopyWithoutPadding<ExecSpace>(compSize, ncomp, maskptr, ptr,
                                          maskedptr, streamID);
        }

        auto handle = GetMultiFieldBlasHandle<ExecSpace>(streamID);

        ForEachSegment(
            X.GetNumFieldPerSegment(), j0, j1,
            [&](const unsigned int js, const unsigned int je) {
                auto aptr = GetColumnsPtr<MemSpace>(X, blk, js, je, streamID);

                // On device, each block writes its own dot products, summed
                // afterwards on stream 0; on host, blocks accumulate in turn.
                TData *y     = out + (js - j0);
                TData scaleY = (blk == 0) ? 0.0 : 1.0;
                if constexpr (std::is_same_v<MemSpace,
                                             NektarSpaces::DeviceSpace>)
                {
                    y      = blockDots + blk * ncols + (js - j0);
                    scaleY = 0.0;
                }

                NekBlas::Gemv(handle, "T", nrows, je - js, (TData)1.0, aptr,
                              X.GetFieldSize(), maskedptr, 1, scaleY, y, 1);
            });
    }

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        SumBlockDots<ExecSpace>(nblocks, ncols, ncols, blockDots, out);
    }
}

} // namespace detail

/**
 * @brief z = beta z + alpha sum of coeffs[j - j0] column j of @p X, for
 * j0 <= j < j1.
 *
 * Block by block, each block on its own stream. @p coeffs is in the memory
 * space of @p ExecSpace. With @p beta zero, @p z is not read.
 */
template <typename ExecSpace, typename TData, FieldState TState>
void MultiAxpy(
    const typename LibUtilities::MultiField<TData, TState>::value_type alpha,
    LibUtilities::MultiField<TData, TState> &X, const unsigned int j0,
    const unsigned int j1, const TData *coeffs,
    const typename LibUtilities::MultiField<TData, TState>::value_type beta,
    LibUtilities::Field<TData, TState> &z)
{
    using MemSpace = typename ExecSpace::memory_space;

    ASSERTL1(j0 < j1 && j1 <= X.GetNumField(),
             "MultiField::MultiAxpy - Invalid column range.");

    for (unsigned int blk = 0; blk < z.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto &zblk               = z.GetBlocks()[blk];
        const unsigned int width = X[j0].GetBlocks()[blk].GetInterleaveWidth();
        const size_t nrows =
            zblk.CompSize() * zblk.GetNumComponents() * zblk.GetNumHomoModes();

        TData *zptr = (beta == 0.0)
                          ? zblk.template GetPtr<MemSpace, WriteOnly>(streamID)
                          : zblk.template GetPtr<MemSpace, ReadWrite>(streamID);
        ASSERTL1(beta == 0.0 || zblk.GetInterleaveWidth() == width,
                 "MultiField::MultiAxpy - Inconsistent interleave format.");

        auto handle  = detail::GetMultiFieldBlasHandle<ExecSpace>(streamID);
        TData scaleZ = beta;
        detail::ForEachSegment(
            X.GetNumFieldPerSegment(), j0, j1,
            [&](const unsigned int js, const unsigned int je) {
                auto aptr =
                    detail::GetColumnsPtr<MemSpace>(X, blk, js, je, streamID);
                NekBlas::Gemv(handle, "N", nrows, je - js, alpha, aptr,
                              X.GetFieldSize(), coeffs + (js - j0), 1, scaleZ,
                              zptr, 1);
                scaleZ = 1.0;
            });

        zblk.template SetInterleaveWidth<TData>(width);
    }
}

/**
 * @brief out[j - j0] = (column j of @p X, field) for j0 <= j < j1, padding
 * excluded.
 *
 * @p out is in the memory space of @p ExecSpace.
 */
template <typename ExecSpace, typename TData, FieldState TState>
void MultiDot(LibUtilities::MultiField<TData, TState> &X, const unsigned int j0,
              const unsigned int j1, LibUtilities::Field<TData, TState> &field,
              TData *out)
{
    LibUtilities::Field<std::uint8_t, TState> *mask = nullptr;
    detail::MultiDotImpl<ExecSpace>(mask, X, j0, j1, field, out);
}

/**
 * @brief out[j - j0] = (column j of @p X, field) for j0 <= j < j1, over the
 * entries @p mask selects. @p mask must exclude padding, as for Math::ddot.
 *
 * @p out is in the memory space of @p ExecSpace.
 */
template <typename ExecSpace, typename TData, FieldState TState>
void MultiDot(LibUtilities::Field<std::uint8_t, TState> &mask,
              LibUtilities::MultiField<TData, TState> &X, const unsigned int j0,
              const unsigned int j1, LibUtilities::Field<TData, TState> &field,
              TData *out)
{
    detail::MultiDotImpl<ExecSpace>(&mask, X, j0, j1, field, out);
}

/**
 * @brief out = Gram matrix of columns j0 <= j < j1 of @p X, padding excluded:
 * out[a + b n] = (column j0 + a, column j0 + b), n = j1 - j0.
 *
 * The columns must lie in one segment. @p out is in the memory space of
 * @p ExecSpace.
 */
template <typename ExecSpace, typename TData, FieldState TState>
void Gram(LibUtilities::MultiField<TData, TState> &X, const unsigned int j0,
          const unsigned int j1, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    ASSERTL1(j0 < j1 && j1 <= X.GetNumField(),
             "MultiField::Gram - Invalid column range.");
    ASSERTL1(j0 / X.GetNumFieldPerSegment() ==
                 (j1 - 1) / X.GetNumFieldPerSegment(),
             "MultiField::Gram - Columns span several segments.");

    const unsigned int ncols   = j1 - j0;
    const unsigned int nblocks = X[j0].GetBlocks().size();
    TData *blockDots           = nullptr;
    if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        blockDots = detail::MultiFieldWorkSpace<TData>::template Get<MemSpace>(
            nblocks * ncols * ncols, 0);
    }

    for (unsigned int blk = 0; blk < nblocks; ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto &block           = X[j0].GetBlocks()[blk];
        const size_t compSize = block.CompSize();
        const unsigned int ncomp =
            block.GetNumComponents() * block.GetNumHomoModes();
        const size_t nrows = compSize * ncomp;

        // Copies of the columns without padding, one after the other.
        auto aptr = detail::GetColumnsPtr<MemSpace>(X, blk, j0, j1, streamID);
        auto maskedptr =
            detail::MultiFieldWorkSpace<TData>::template Get<MemSpace>(
                ncols * nrows, streamID);
        auto mask = Math::internalMathKernelMask<MemSpace>::GetInstance(block);
        for (unsigned int c = 0; c < ncols; ++c)
        {
            detail::CopyWithoutPadding<ExecSpace>(
                compSize, ncomp, mask, aptr + c * X.GetFieldSize(),
                maskedptr + c * nrows, streamID);
        }

        // On device, each block writes its own products, summed afterwards on
        // stream 0; on host, blocks accumulate in turn.
        TData *c     = out;
        TData scaleC = (blk == 0) ? 0.0 : 1.0;
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            c      = blockDots + blk * ncols * ncols;
            scaleC = 0.0;
        }

        auto handle = detail::GetMultiFieldBlasHandle<ExecSpace>(streamID);
        NekBlas::Gemm(handle, "T", "N", ncols, ncols, nrows, (TData)1.0,
                      maskedptr, nrows, maskedptr, nrows, scaleC, c, ncols);
    }

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        detail::SumBlockDots<ExecSpace>(nblocks, ncols * ncols, ncols * ncols,
                                        blockDots, out);
    }
}

} // namespace Nektar::SolverCore
