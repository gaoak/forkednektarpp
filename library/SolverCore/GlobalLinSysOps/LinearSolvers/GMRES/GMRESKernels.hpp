///////////////////////////////////////////////////////////////////////////////
//
// File: GMRESKernels.hpp
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

#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"
#include "LibUtilities/LoopExecution/LoopExecution.hpp"
#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>

#include <cstdint>
#include <type_traits>

namespace Nektar::SolverCore::detail
{

// Modified Gram-Schmidt step on nsize entries in one pass:
// w = wScale * w - h * v, then the masked product of the updated w with x, or
// with itself if x is w, added to *out. x is read after w is written, so
// neither may be declared restrict.
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
NEK_FORCE_INLINE static void UpdateGramSchmidtDotKernel(
    const size_t nsize, const TData h, const TData *v, const TData wScale,
    TData *w, const std::uint8_t *mask, const TData *x, TData *out,
    [[maybe_unused]] const unsigned int streamID)
{
    TData sum = 0.0;
    for (size_t i = 0; i < nsize; ++i)
    {
        const TData w0 = -h * v[i] + wScale * w[i];
        w[i]           = w0;
        sum += mask[i] * w0 * x[i];
    }
    *out += sum;
}

// One SIMD chunk of UpdateGramSchmidtDotKernel at entry offset: updates w and
// accumulates the masked product in sumChunk. tmp is scratch for the mask.
template <typename TData>
NEK_FORCE_INLINE static void UpdateGramSchmidtDotChunk(
    const size_t offset, const tinysimd::simd<TData> &hChunk,
    const tinysimd::simd<TData> &scaleChunk, const TData *v, TData *w,
    const std::uint8_t *mask, const TData *x,
    typename tinysimd::simd<TData>::scalarArray &tmp,
    tinysimd::simd<TData> &sumChunk)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    // w = wScale * w - h * v
    simd_t vChunk, wChunk;
    vChunk.load(v + offset, is_aligned);
    wChunk.load(w + offset, is_aligned);
    simd_t w0Chunk = scaleChunk * wChunk;
    w0Chunk.fma(vChunk, hChunk);
    w0Chunk.store(w + offset, is_aligned);

    // The mask has no SIMD type: widen it through tmp.
    for (unsigned int i = 0; i < simd_t::width; ++i)
    {
        tmp[i] = mask[offset + i];
    }
    simd_t maskChunk;
    maskChunk.load(tmp, is_aligned);

    // x may be w: loaded after w is stored, it holds the updated w.
    simd_t xChunk;
    xChunk.load(x + offset, is_aligned);
    sumChunk.fma(maskChunk * w0Chunk, xChunk);
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
NEK_FORCE_INLINE static void UpdateGramSchmidtDotKernel(
    const size_t nsize, const TData h, const TData *v, const TData wScale,
    TData *w, const std::uint8_t *mask, const TData *x, TData *out,
    [[maybe_unused]] const unsigned int streamID)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    simd_t hChunk, scaleChunk;
    hChunk.broadcast(-h);
    scaleChunk.broadcast(wScale);
    simd_t sumChunk0 = 0, sumChunk1 = 0, sumChunk2 = 0, sumChunk3 = 0;
    alignas(simd_t::alignment) typename simd_t::scalarArray tmp;

    size_t i = 0;

    // Vectorized loop unroll 4x
    for (; i + 4 * simd_t::width <= nsize; i += 4 * simd_t::width)
    {
        UpdateGramSchmidtDotChunk(i, hChunk, scaleChunk, v, w, mask, x, tmp,
                                  sumChunk0);
        UpdateGramSchmidtDotChunk(i + simd_t::width, hChunk, scaleChunk, v, w,
                                  mask, x, tmp, sumChunk1);
        UpdateGramSchmidtDotChunk(i + 2 * simd_t::width, hChunk, scaleChunk, v,
                                  w, mask, x, tmp, sumChunk2);
        UpdateGramSchmidtDotChunk(i + 3 * simd_t::width, hChunk, scaleChunk, v,
                                  w, mask, x, tmp, sumChunk3);
    }

    // Vectorized loop unroll 2x
    for (; i + 2 * simd_t::width <= nsize; i += 2 * simd_t::width)
    {
        UpdateGramSchmidtDotChunk(i, hChunk, scaleChunk, v, w, mask, x, tmp,
                                  sumChunk0);
        UpdateGramSchmidtDotChunk(i + simd_t::width, hChunk, scaleChunk, v, w,
                                  mask, x, tmp, sumChunk1);
    }

    // Vectorized loop
    for (; i + simd_t::width <= nsize; i += simd_t::width)
    {
        UpdateGramSchmidtDotChunk(i, hChunk, scaleChunk, v, w, mask, x, tmp,
                                  sumChunk0);
    }

    sumChunk0 = (sumChunk0 + sumChunk1) + (sumChunk2 + sumChunk3);
    sumChunk0.store(tmp, is_aligned);
    TData sum = 0.0;
    for (unsigned int j = 0; j < simd_t::width; ++j)
    {
        sum += tmp[j];
    }

    // Spillover loop.
    for (; i < nsize; ++i)
    {
        const TData w0 = -h * v[i] + wScale * w[i];
        w[i]           = w0;
        sum += mask[i] * w0 * x[i];
    }
    *out += sum;
}

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void UpdateGramSchmidtDotKernel(
    const size_t nsize, const TData h, const TData *v, const TData wScale,
    TData *w, const std::uint8_t *mask, const TData *x, TData *out,
    const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_reduce<ExecSpace, false, Nektar::ReduceSum<TData>>(
        0, nsize,
        NEKTAR_LAMBDA(const size_t idx) {
            const TData w0 = -h * v[idx] + wScale * w[idx];
            w[idx]         = w0;
            return mask[idx] * w0 * x[idx];
        },
        out);
    Nektar::LoopExecutionSetStreamID(0);
}
#endif

// Modified Gram-Schmidt step in one pass: w = wScale * w - h * v, then the
// masked product of the updated w with x, or with itself if x is w, in
// *out, in the memory space of ExecSpace.
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void UpdateGramSchmidtDot(
    const TData h, LibUtilities::Field<TData, FieldState::Coeff> &v,
    const TData wScale, LibUtilities::Field<TData, FieldState::Coeff> &w,
    LibUtilities::Field<std::uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<TData, FieldState::Coeff> &x, TData *out)
{
    using MemSpace = typename ExecSpace::memory_space;

    Math::zeroKernel<ExecSpace>(1, out);

    for (unsigned int blk = 0; blk < w.GetBlocks().size(); ++blk)
    {
        const unsigned int streamID = blk + 1;

        auto size = w.GetBlocks()[blk].CompSize() * w.GetNumComponents() *
                    w.GetNumHomoModes();
        auto vptr =
            v.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto maskptr =
            mask.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
        auto wptr =
            w.GetBlocks()[blk].template GetPtr<MemSpace, ReadWrite>(streamID);
        auto xptr =
            x.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>(streamID);

        // The padding is zero, so the whole block is reduced.
        UpdateGramSchmidtDotKernel<ExecSpace>(size, h, vptr, wScale, wptr,
                                              maskptr, xptr, out, streamID);
    }
}

} // namespace Nektar::SolverCore::detail
