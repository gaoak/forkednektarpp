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
 * @brief Assert that the padding of columns [j0, j1) of @p X, and of
 * @p field if given, is zero, as the products assume.
 *
 * Debug builds only, since it reads the data on host.
 */
template <typename TData, FieldState TState>
void AssertZeroPadding(LibUtilities::MultiField<TData, TState> &X,
                       const unsigned int j0, const unsigned int j1,
                       LibUtilities::Field<TData, TState> *field = nullptr)
{
    for (unsigned int j = j0; j < j1; ++j)
    {
        Math::internalAssertZeroPadding(X[j]);
    }
    if (field)
    {
        Math::internalAssertZeroPadding(*field);
    }
}

/**
 * @brief out[j - j0] = (column j of @p X, field) for j0 <= j < j1, over the
 * entries @p mask selects if given, otherwise padding excluded, which is
 * zero.
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

    AssertZeroPadding(X, j0, j1, mask ? nullptr : &field);

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

        const size_t nrows = block.CompSize() * block.GetNumComponents() *
                             block.GetNumHomoModes();

        // Masked copy of block, or block itself, whose padding is zero.
        auto ptr = block.template GetPtr<MemSpace, ReadOnly>(streamID);
        if (mask)
        {
            auto maskedptr = MultiFieldWorkSpace<TData>::template Get<MemSpace>(
                nrows, streamID);
            auto maskptr =
                mask->GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID);
            CopyMasked<ExecSpace>(nrows, maskptr, ptr, maskedptr, streamID);
            ptr = maskedptr;
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
                              X.GetFieldSize(), ptr, 1, scaleY, y, 1);
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
 * @brief Column j + l of @p Y = beta column j + l of @p Y + alpha sum of
 * coeffs[l n + i - j0] column i of @p X, for j0 <= i < j1, l = 0, 1,
 * n = j1 - j0.
 *
 * Block by block, each block on its own stream. When columns j and j + 1 of
 * @p Y lie in one segment, both are updated by one matrix product per segment
 * of @p X, reading the columns of @p X once; otherwise each is updated by
 * MultiAxpy. The columns of @p X must not include them. @p coeffs is in the
 * memory space of @p ExecSpace. With @p beta zero, @p Y is not read.
 */
template <typename ExecSpace, typename TData, FieldState TState>
void MultiAxpy2(
    const typename LibUtilities::MultiField<TData, TState>::value_type alpha,
    LibUtilities::MultiField<TData, TState> &X, const unsigned int j0,
    const unsigned int j1, const TData *coeffs,
    const typename LibUtilities::MultiField<TData, TState>::value_type beta,
    LibUtilities::MultiField<TData, TState> &Y, const unsigned int j)
{
    using MemSpace = typename ExecSpace::memory_space;

    ASSERTL1(j0 < j1 && j1 <= X.GetNumField(),
             "MultiField::MultiAxpy2 - Invalid column range.");
    ASSERTL1(j + 1 < Y.GetNumField(),
             "MultiField::MultiAxpy2 - Invalid output columns.");
    ASSERTL1(&X != &Y || j1 <= j || j + 2 <= j0,
             "MultiField::MultiAxpy2 - Output columns in the column range.");

    const unsigned int ncols = j1 - j0;

    if (j / Y.GetNumFieldPerSegment() != (j + 1) / Y.GetNumFieldPerSegment())
    {
        MultiAxpy<ExecSpace>(alpha, X, j0, j1, coeffs, beta, Y[j]);
        MultiAxpy<ExecSpace>(alpha, X, j0, j1, coeffs + ncols, beta, Y[j + 1]);
        return;
    }

    for (unsigned int blk = 0; blk < Y[j].GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto &yblk               = Y[j].GetBlocks()[blk];
        const unsigned int width = X[j0].GetBlocks()[blk].GetInterleaveWidth();
        const size_t nrows =
            yblk.CompSize() * yblk.GetNumComponents() * yblk.GetNumHomoModes();

        TData *yptr = nullptr;
        for (unsigned int l = 0; l < 2; ++l)
        {
            auto &block = Y[j + l].GetBlocks()[blk];
            TData *ptr =
                (beta == 0.0)
                    ? block.template GetPtr<MemSpace, WriteOnly>(streamID)
                    : block.template GetPtr<MemSpace, ReadWrite>(streamID);
            if (l == 0)
            {
                yptr = ptr;
            }
            ASSERTL1(ptr == yptr + l * Y.GetFieldSize(),
                     "MultiField::MultiAxpy2 - Output columns are not "
                     "contiguous.");
            ASSERTL1(beta == 0.0 || block.GetInterleaveWidth() == width,
                     "MultiField::MultiAxpy2 - Inconsistent interleave "
                     "format.");
        }

        auto handle  = detail::GetMultiFieldBlasHandle<ExecSpace>(streamID);
        TData scaleY = beta;
        detail::ForEachSegment(
            X.GetNumFieldPerSegment(), j0, j1,
            [&](const unsigned int js, const unsigned int je) {
                auto aptr =
                    detail::GetColumnsPtr<MemSpace>(X, blk, js, je, streamID);
                NekBlas::Gemm(handle, "N", "N", nrows, 2, je - js, alpha, aptr,
                              X.GetFieldSize(), coeffs + (js - j0), ncols,
                              scaleY, yptr, Y.GetFieldSize());
                scaleY = 1.0;
            });

        for (unsigned int l = 0; l < 2; ++l)
        {
            Y[j + l].GetBlocks()[blk].template SetInterleaveWidth<TData>(width);
        }
    }
}

/**
 * @brief out[j - j0] = (column j of @p X, field) for j0 <= j < j1, padding
 * excluded.
 *
 * The padding of the columns and of @p field must be zero. @p out is in the
 * memory space of @p ExecSpace.
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
 * The padding of the columns must be zero. @p out is in the memory space of
 * @p ExecSpace.
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
 * @brief out[j - j0] = (column j of @p X, field0) and
 * out[n + j - j0] = (column j of @p X, field1) for j0 <= j < j1,
 * n = j1 - j0, over the entries @p mask selects, reading the columns once.
 * @p mask must exclude padding, as for Math::ddot.
 *
 * The padding of the columns must be zero. Either field may be a column of
 * @p X. @p out is in the memory space of @p ExecSpace.
 */
template <typename ExecSpace, typename TData, FieldState TState>
void MultiDot2(LibUtilities::Field<std::uint8_t, TState> &mask,
               LibUtilities::MultiField<TData, TState> &X,
               const unsigned int j0, const unsigned int j1,
               LibUtilities::Field<TData, TState> &field0,
               LibUtilities::Field<TData, TState> &field1, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    ASSERTL1(j0 < j1 && j1 <= X.GetNumField(),
             "MultiField::MultiDot2 - Invalid column range.");

    detail::AssertZeroPadding(X, j0, j1);

    const unsigned int ncols   = j1 - j0;
    const unsigned int nblocks = field0.GetBlocks().size();
    TData *blockDots           = nullptr;
    if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        blockDots = detail::MultiFieldWorkSpace<TData>::template Get<MemSpace>(
            nblocks * 2 * ncols, 0);
    }

    for (unsigned int blk = 0; blk < nblocks; ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto &block0 = field0.GetBlocks()[blk];
        auto &block1 = field1.GetBlocks()[blk];
        ASSERTL1(block0.GetInterleaveWidth() ==
                         X[j0].GetBlocks()[blk].GetInterleaveWidth() &&
                     block1.GetInterleaveWidth() ==
                         X[j0].GetBlocks()[blk].GetInterleaveWidth(),
                 "MultiField::MultiDot2 - Inconsistent interleave format.");

        const size_t nrows = block0.CompSize() * block0.GetNumComponents() *
                             block0.GetNumHomoModes();

        // Masked copies of both blocks, side by side.
        auto maskedptr =
            detail::MultiFieldWorkSpace<TData>::template Get<MemSpace>(
                2 * nrows, streamID);
        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        detail::CopyMasked<ExecSpace>(
            nrows, maskptr,
            block0.template GetPtr<MemSpace, ReadOnly>(streamID), maskedptr,
            streamID);
        detail::CopyMasked<ExecSpace>(
            nrows, maskptr,
            block1.template GetPtr<MemSpace, ReadOnly>(streamID),
            maskedptr + nrows, streamID);

        auto handle = detail::GetMultiFieldBlasHandle<ExecSpace>(streamID);

        detail::ForEachSegment(
            X.GetNumFieldPerSegment(), j0, j1,
            [&](const unsigned int js, const unsigned int je) {
                auto aptr =
                    detail::GetColumnsPtr<MemSpace>(X, blk, js, je, streamID);

                // On device, each block writes its own dot products, summed
                // afterwards on stream 0; on host, blocks accumulate in turn.
                TData *y     = out + (js - j0);
                TData scaleY = (blk == 0) ? 0.0 : 1.0;
                if constexpr (std::is_same_v<MemSpace,
                                             NektarSpaces::DeviceSpace>)
                {
                    y      = blockDots + blk * 2 * ncols + (js - j0);
                    scaleY = 0.0;
                }

                NekBlas::Gemm(handle, "T", "N", je - js, 2, nrows, (TData)1.0,
                              aptr, X.GetFieldSize(), maskedptr, nrows, scaleY,
                              y, ncols);
            });
    }

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        detail::SumBlockDots<ExecSpace>(nblocks, 2 * ncols, 2 * ncols,
                                        blockDots, out);
    }
}

/**
 * @brief out = Gram matrix of columns j0 <= j < j1 of @p X, padding excluded:
 * out[a + b n] = (column j0 + a, column j0 + b), n = j1 - j0.
 *
 * The columns must lie in one segment and their padding must be zero. @p out
 * is in the memory space of @p ExecSpace.
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

    detail::AssertZeroPadding(X, j0, j1);

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

        auto &block        = X[j0].GetBlocks()[blk];
        const size_t nrows = block.CompSize() * block.GetNumComponents() *
                             block.GetNumHomoModes();

        auto aptr = detail::GetColumnsPtr<MemSpace>(X, blk, j0, j1, streamID);

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
        NekBlas::Gemm(handle, "T", "N", ncols, ncols, nrows, (TData)1.0, aptr,
                      X.GetFieldSize(), aptr, X.GetFieldSize(), scaleC, c,
                      ncols);
    }

    if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        detail::SumBlockDots<ExecSpace>(nblocks, ncols * ncols, ncols * ncols,
                                        blockDots, out);
    }
}

} // namespace Nektar::SolverCore
