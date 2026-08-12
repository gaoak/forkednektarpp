///////////////////////////////////////////////////////////////////////////////
//
// File: MathAVXKernels.hpp
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

#include "Operators/Common/Spaces.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <numeric>
#include <type_traits>

namespace Nektar::Math
{

// NOTE: Those AVX Math kernels assumed aligned memory. Using non-aligned memory
// would result in memory fault.

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void zeroKernel(const size_t nsize, TData *x,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    std::memset(x, 0, nsize * sizeof(TData));
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void fillKernel(const size_t nsize, const TData &val, TData *x,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    std::fill(x, x + nsize, val);
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void copyKernel(const size_t nsize, const TData *x, TData *y,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    std::memcpy(y, x, nsize * sizeof(TData));
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void absKernel(const size_t nsize, const TData *x, TData *y,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt = nsize;

    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;

        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        simd_t yChunk0 = abs(xChunk0);
        simd_t yChunk1 = abs(xChunk1);
        simd_t yChunk2 = abs(xChunk2);
        simd_t yChunk3 = abs(xChunk3);

        yChunk0.store(y, is_aligned);
        yChunk1.store(y + simd_t::width, is_aligned);
        yChunk2.store(y + 2 * simd_t::width, is_aligned);
        yChunk3.store(y + 3 * simd_t::width, is_aligned);

        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }

    // Unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        simd_t xChunk0, xChunk1;

        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        simd_t yChunk0 = abs(xChunk0);
        simd_t yChunk1 = abs(xChunk1);

        yChunk0.store(y, is_aligned);
        yChunk1.store(y + simd_t::width, is_aligned);

        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        simd_t xChunk;
        xChunk.load(x, is_aligned);
        simd_t yChunk = abs(xChunk);
        yChunk.store(y, is_aligned);
        x += simd_t::width;
        y += simd_t::width;
        cnt -= simd_t::width;
    }

    // Spillover loop
    while (cnt)
    {
        *y = std::abs(*x);
        ++x;
        ++y;
        --cnt;
    }
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void negKernel(const size_t nsize, const TData *x, TData *y,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt = nsize;
    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        // y = -x
        simd_t yChunk0 = -xChunk0;
        simd_t yChunk1 = -xChunk1;
        simd_t yChunk2 = -xChunk2;
        simd_t yChunk3 = -xChunk3;

        // store
        yChunk0.store(y, is_aligned);
        yChunk1.store(y + simd_t::width, is_aligned);
        yChunk2.store(y + 2 * simd_t::width, is_aligned);
        yChunk3.store(y + 3 * simd_t::width, is_aligned);

        // update pointers
        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        // y = -x
        simd_t yChunk0 = -xChunk0;
        simd_t yChunk1 = -xChunk1;

        // store
        yChunk0.store(y, is_aligned);
        yChunk1.store(y + simd_t::width, is_aligned);

        // update pointers
        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk;
        xChunk.load(x, is_aligned);

        // z = -x
        simd_t yChunk = -xChunk;

        // store
        yChunk.store(y, is_aligned);

        // update pointers
        x += simd_t::width;
        y += simd_t::width;
        cnt -= simd_t::width;
    }

    // spillover loop
    while (cnt)
    {
        // y = -x;
        *y = -(*x);
        // update pointers
        ++x;
        ++y;
        --cnt;
    }
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void sqrtKernel(const size_t nsize, const TData *x, TData *y,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;
    size_t cnt   = nsize;

    while (cnt >= 4 * simd_t::width)
    {
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        simd_t yChunk0 = sqrt(xChunk0), yChunk1 = sqrt(xChunk1),
               yChunk2 = sqrt(xChunk2), yChunk3 = sqrt(xChunk3);

        yChunk0.store(y, is_aligned);
        yChunk1.store(y + simd_t::width, is_aligned);
        yChunk2.store(y + 2 * simd_t::width, is_aligned);
        yChunk3.store(y + 3 * simd_t::width, is_aligned);

        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }
    while (cnt >= simd_t::width)
    {
        simd_t xvChunk;
        xvChunk.load(x, is_aligned);
        simd_t yvChunk = sqrt(xvChunk);
        yvChunk.store(y, is_aligned);
        x += simd_t::width;
        y += simd_t::width;
        cnt -= simd_t::width;
    }
    while (cnt)
    {
        *y = std::sqrt(*x);
        ++x;
        ++y;
        --cnt;
    }
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void addKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt = nsize;
    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1, yChunk2, yChunk3;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);
        yChunk2.load(y + 2 * simd_t::width, is_aligned);
        yChunk3.load(y + 3 * simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        // z = x + y
        simd_t zChunk0 = xChunk0 + yChunk0;
        simd_t zChunk1 = xChunk1 + yChunk1;
        simd_t zChunk2 = xChunk2 + yChunk2;
        simd_t zChunk3 = xChunk3 + yChunk3;

        // store
        zChunk0.store(z, is_aligned);
        zChunk1.store(z + simd_t::width, is_aligned);
        zChunk2.store(z + 2 * simd_t::width, is_aligned);
        zChunk3.store(z + 3 * simd_t::width, is_aligned);

        // update pointers
        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        z += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        // z = x + y
        simd_t zChunk0 = xChunk0 + yChunk0;
        simd_t zChunk1 = xChunk1 + yChunk1;

        // store
        zChunk0.store(z, is_aligned);
        zChunk1.store(z + simd_t::width, is_aligned);

        // update pointers
        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        z += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t yChunk;
        yChunk.load(y, is_aligned);
        simd_t xChunk;
        xChunk.load(x, is_aligned);

        // z = x + y
        simd_t zChunk = xChunk + yChunk;

        // store
        zChunk.store(z, is_aligned);

        // update pointers
        x += simd_t::width;
        y += simd_t::width;
        z += simd_t::width;
        cnt -= simd_t::width;
    }

    // spillover loop
    while (cnt)
    {
        // z = x + y;
        *z = (*x) + (*y);
        // update pointers
        ++x;
        ++y;
        ++z;
        --cnt;
    }
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void subKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt = nsize;
    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1, yChunk2, yChunk3;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);
        yChunk2.load(y + 2 * simd_t::width, is_aligned);
        yChunk3.load(y + 3 * simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        // z = x - y
        simd_t zChunk0 = xChunk0 - yChunk0;
        simd_t zChunk1 = xChunk1 - yChunk1;
        simd_t zChunk2 = xChunk2 - yChunk2;
        simd_t zChunk3 = xChunk3 - yChunk3;

        // store
        zChunk0.store(z, is_aligned);
        zChunk1.store(z + simd_t::width, is_aligned);
        zChunk2.store(z + 2 * simd_t::width, is_aligned);
        zChunk3.store(z + 3 * simd_t::width, is_aligned);

        // update pointers
        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        z += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        // z = x - y
        simd_t zChunk0 = xChunk0 - yChunk0;
        simd_t zChunk1 = xChunk1 - yChunk1;

        // store
        zChunk0.store(z, is_aligned);
        zChunk1.store(z + simd_t::width, is_aligned);

        // update pointers
        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        z += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t yChunk;
        yChunk.load(y, is_aligned);
        simd_t xChunk;
        xChunk.load(x, is_aligned);

        // z = x - y
        simd_t zChunk = xChunk - yChunk;

        // store
        zChunk.store(z, is_aligned);

        // update pointers
        x += simd_t::width;
        y += simd_t::width;
        z += simd_t::width;
        cnt -= simd_t::width;
    }

    // spillover loop
    while (cnt)
    {
        // z = x - y;
        *z = (*x) - (*y);
        // update pointers
        ++x;
        ++y;
        ++z;
        --cnt;
    }
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void mulKernel(const size_t nsize, const TData alpha, const TData *x,
                      TData *y,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    simd_t aChunk;
    aChunk.broadcast(alpha);

    size_t cnt = nsize;
    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        // y = alpha * x
        simd_t yChunk0 = aChunk * xChunk0;
        simd_t yChunk1 = aChunk * xChunk1;
        simd_t yChunk2 = aChunk * xChunk2;
        simd_t yChunk3 = aChunk * xChunk3;

        // store
        yChunk0.store(y, is_aligned);
        yChunk1.store(y + simd_t::width, is_aligned);
        yChunk2.store(y + 2 * simd_t::width, is_aligned);
        yChunk3.store(y + 3 * simd_t::width, is_aligned);

        // update pointers
        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        // y = alpha * x
        simd_t yChunk0 = aChunk * xChunk0;
        simd_t yChunk1 = aChunk * xChunk1;

        // store
        yChunk0.store(y, is_aligned);
        yChunk1.store(y + simd_t::width, is_aligned);

        // update pointers
        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk;
        xChunk.load(x, is_aligned);

        // y = alpha * x
        simd_t yChunk = aChunk * xChunk;

        // store
        yChunk.store(y, is_aligned);

        // update pointers
        x += simd_t::width;
        y += simd_t::width;
        cnt -= simd_t::width;
    }

    // spillover loop
    while (cnt)
    {
        // y = alpha * x;
        *y = alpha * (*x);
        // update pointers
        ++x;
        ++y;
        --cnt;
    }
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void mulKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt = nsize;
    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1, yChunk2, yChunk3;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);
        yChunk2.load(y + 2 * simd_t::width, is_aligned);
        yChunk3.load(y + 3 * simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        // z = x * y
        simd_t zChunk0 = xChunk0 * yChunk0;
        simd_t zChunk1 = xChunk1 * yChunk1;
        simd_t zChunk2 = xChunk2 * yChunk2;
        simd_t zChunk3 = xChunk3 * yChunk3;

        // store
        zChunk0.store(z, is_aligned);
        zChunk1.store(z + simd_t::width, is_aligned);
        zChunk2.store(z + 2 * simd_t::width, is_aligned);
        zChunk3.store(z + 3 * simd_t::width, is_aligned);

        // update pointers
        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        z += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        // z = x * y
        simd_t zChunk0 = xChunk0 * yChunk0;
        simd_t zChunk1 = xChunk1 * yChunk1;

        // store
        zChunk0.store(z, is_aligned);
        zChunk1.store(z + simd_t::width, is_aligned);

        // update pointers
        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        z += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t yChunk;
        yChunk.load(y, is_aligned);
        simd_t xChunk;
        xChunk.load(x, is_aligned);

        // z = x * y
        simd_t zChunk = xChunk * yChunk;

        // store
        zChunk.store(z, is_aligned);

        // update pointers
        x += simd_t::width;
        y += simd_t::width;
        z += simd_t::width;
        cnt -= simd_t::width;
    }

    // spillover loop
    while (cnt)
    {
        // z = x * y;
        *z = (*x) * (*y);
        // update pointers
        ++x;
        ++y;
        ++z;
        --cnt;
    }
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void divKernel(const size_t nsize, const TData alpha, const TData *x,
                      TData *y,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    simd_t aChunk;
    aChunk.broadcast(alpha);

    size_t cnt = nsize;
    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        // y = alpha / x
        simd_t yChunk0 = aChunk / xChunk0;
        simd_t yChunk1 = aChunk / xChunk1;
        simd_t yChunk2 = aChunk / xChunk2;
        simd_t yChunk3 = aChunk / xChunk3;

        // store
        yChunk0.store(y, is_aligned);
        yChunk1.store(y + simd_t::width, is_aligned);
        yChunk2.store(y + 2 * simd_t::width, is_aligned);
        yChunk3.store(y + 3 * simd_t::width, is_aligned);

        // update pointers
        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        // y = alpha / x
        simd_t yChunk0 = aChunk / xChunk0;
        simd_t yChunk1 = aChunk / xChunk1;

        // store
        yChunk0.store(y, is_aligned);
        yChunk1.store(y + simd_t::width, is_aligned);

        // update pointers
        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk;
        xChunk.load(x, is_aligned);

        // y = alpha / x
        simd_t yChunk = aChunk / xChunk;

        // store
        yChunk.store(y, is_aligned);

        // update pointers
        x += simd_t::width;
        y += simd_t::width;
        cnt -= simd_t::width;
    }

    // spillover loop
    while (cnt)
    {
        // y = alpha / x;
        *y = alpha / (*x);
        // update pointers
        ++x;
        ++y;
        --cnt;
    }
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void divKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt = nsize;
    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1, yChunk2, yChunk3;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);
        yChunk2.load(y + 2 * simd_t::width, is_aligned);
        yChunk3.load(y + 3 * simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        // z = x / y
        simd_t zChunk0 = xChunk0 / yChunk0;
        simd_t zChunk1 = xChunk1 / yChunk1;
        simd_t zChunk2 = xChunk2 / yChunk2;
        simd_t zChunk3 = xChunk3 / yChunk3;

        // store
        zChunk0.store(z, is_aligned);
        zChunk1.store(z + simd_t::width, is_aligned);
        zChunk2.store(z + 2 * simd_t::width, is_aligned);
        zChunk3.store(z + 3 * simd_t::width, is_aligned);

        // update pointers
        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        z += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        // z = x / y
        simd_t zChunk0 = xChunk0 / yChunk0;
        simd_t zChunk1 = xChunk1 / yChunk1;

        // store
        zChunk0.store(z, is_aligned);
        zChunk1.store(z + simd_t::width, is_aligned);

        // update pointers
        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        z += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t yChunk;
        yChunk.load(y, is_aligned);
        simd_t xChunk;
        xChunk.load(x, is_aligned);

        // z = x / y
        simd_t zChunk = xChunk / yChunk;

        // store
        zChunk.store(z, is_aligned);

        // update pointers
        x += simd_t::width;
        y += simd_t::width;
        z += simd_t::width;
        cnt -= simd_t::width;
    }

    // spillover loop
    while (cnt)
    {
        // z = x / y;
        *z = (*x) / (*y);
        // update pointers
        ++x;
        ++y;
        ++z;
        --cnt;
    }
}

template <typename ExecSpace, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void daxpyKernel(const size_t nsize, const TData alpha, const TData *x,
                        const TData *y, TData *z,
                        [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt = nsize;
    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1, yChunk2, yChunk3;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);
        yChunk2.load(y + 2 * simd_t::width, is_aligned);
        yChunk3.load(y + 3 * simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        // z = alpha * x + y
        yChunk0.fma(simd_t(xChunk0), alpha);
        yChunk1.fma(simd_t(xChunk1), alpha);
        yChunk2.fma(simd_t(xChunk2), alpha);
        yChunk3.fma(simd_t(xChunk3), alpha);

        // store
        yChunk0.store(z, is_aligned);
        yChunk1.store(z + simd_t::width, is_aligned);
        yChunk2.store(z + 2 * simd_t::width, is_aligned);
        yChunk3.store(z + 3 * simd_t::width, is_aligned);

        // update pointers
        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        z += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t yChunk0, yChunk1;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);

        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        // z = alpha * x + y
        yChunk0.fma(simd_t(xChunk0), alpha);
        yChunk1.fma(simd_t(xChunk1), alpha);

        // store
        yChunk0.store(z, is_aligned);
        yChunk1.store(z + simd_t::width, is_aligned);

        // update pointers
        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        z += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t yChunk;
        yChunk.load(y, is_aligned);
        simd_t xChunk;
        xChunk.load(x, is_aligned);

        // z = alpha * x + y
        yChunk.fma(simd_t(xChunk), alpha);

        // store
        yChunk.store(z, is_aligned);

        // update pointers
        x += simd_t::width;
        y += simd_t::width;
        z += simd_t::width;
        cnt -= simd_t::width;
    }

    // spillover loop
    while (cnt)
    {
        // z = alpha * x + y;
        *z = alpha * (*x) + (*y);
        // update pointers
        ++x;
        ++y;
        ++z;
        --cnt;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void reduceSumKernel(const size_t nsize, const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt     = nsize;
    simd_t yChunk0 = 0, yChunk1 = 0, yChunk2 = 0, yChunk3 = 0;
    alignas(simd_t::alignment) typename simd_t::scalarArray tmp;

    if constexpr (init)
    {
        *out = 0.0;
    }

    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        yChunk0 += xChunk0;
        yChunk1 += xChunk1;
        yChunk2 += xChunk2;
        yChunk3 += xChunk3;

        // update pointers
        x += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }
    yChunk3.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }
    yChunk2.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        yChunk0 += xChunk0;
        yChunk1 += xChunk1;

        // update pointers
        x += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }
    yChunk1.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk0;
        xChunk0.load(x, is_aligned);

        yChunk0 += xChunk0;

        // update pointers
        x += simd_t::width;
        cnt -= simd_t::width;
    }
    yChunk0.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // spillover loop
    while (cnt)
    {
        *out += *x;
        // update pointers
        ++x;
        --cnt;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void reduceSumKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    // TODO: SIMD/AVX
    TData initializer = init ? 0.0 : *out;
    *out              = std::inner_product(mask, mask + nsize, x, initializer);
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void reduceMaxKernel(const size_t nsize, const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt     = nsize;
    simd_t yChunk0 = 0, yChunk1 = 0, yChunk2 = 0, yChunk3 = 0;
    alignas(simd_t::alignment) typename simd_t::scalarArray tmp;

    if constexpr (init)
    {
        *out = std::numeric_limits<TData>::lowest();
    }

    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        yChunk0 = max(xChunk0, yChunk0);
        yChunk1 = max(xChunk1, yChunk1);
        yChunk2 = max(xChunk2, yChunk2);
        yChunk3 = max(xChunk3, yChunk3);

        // update pointers
        x += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }
    yChunk3.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::max(tmp[i], *out);
    }
    yChunk2.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::max(tmp[i], *out);
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        yChunk0 = max(xChunk0, yChunk0);
        yChunk1 = max(xChunk1, yChunk1);

        // update pointers
        x += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }
    yChunk1.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::max(tmp[i], *out);
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk0;
        xChunk0.load(x, is_aligned);

        yChunk0 = max(xChunk0, yChunk0);

        // update pointers
        x += simd_t::width;
        cnt -= simd_t::width;
    }
    yChunk0.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::max(tmp[i], *out);
    }

    // spillover loop
    while (cnt)
    {
        *out = std::max(*x, *out);
        // update pointers
        ++x;
        --cnt;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void reduceMaxKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    // TODO: SIMD/AVX
    if (init)
    {
        *out = std::numeric_limits<TData>::lowest();
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out = mask[i] ? std::max(*out, x[i]) : *out;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void reduceMinKernel(const size_t nsize, const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt     = nsize;
    simd_t yChunk0 = 0, yChunk1 = 0, yChunk2 = 0, yChunk3 = 0;
    alignas(simd_t::alignment) typename simd_t::scalarArray tmp;

    if constexpr (init)
    {
        *out = std::numeric_limits<TData>::max();
    }

    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        yChunk0 = min(xChunk0, yChunk0);
        yChunk1 = min(xChunk1, yChunk1);
        yChunk2 = min(xChunk2, yChunk2);
        yChunk3 = min(xChunk3, yChunk3);

        // update pointers
        x += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }
    yChunk3.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::min(tmp[i], *out);
    }
    yChunk2.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::min(tmp[i], *out);
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        yChunk0 = min(xChunk0, yChunk0);
        yChunk1 = min(xChunk1, yChunk1);

        // update pointers
        x += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }
    yChunk1.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::min(tmp[i], *out);
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk0;
        xChunk0.load(x, is_aligned);

        yChunk0 = min(xChunk0, yChunk0);

        // update pointers
        x += simd_t::width;
        cnt -= simd_t::width;
    }
    yChunk0.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::min(tmp[i], *out);
    }

    // spillover loop
    while (cnt)
    {
        *out = std::min(*x, *out);
        // update pointers
        ++x;
        --cnt;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void reduceMinKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    // TODO: SIMD/AVX
    if (init)
    {
        *out = std::numeric_limits<TData>::max();
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out = mask[i] ? std::min(*out, x[i]) : *out;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void ddotKernel(const size_t nsize, const TData *x, const TData *y,
                       TData *out,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt     = nsize;
    simd_t zChunk0 = 0, zChunk1 = 0, zChunk2 = 0, zChunk3 = 0;
    alignas(simd_t::alignment) typename simd_t::scalarArray tmp;

    if constexpr (init)
    {
        *out = 0.0;
    }

    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        simd_t yChunk0, yChunk1, yChunk2, yChunk3;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);
        yChunk2.load(y + 2 * simd_t::width, is_aligned);
        yChunk3.load(y + 3 * simd_t::width, is_aligned);

        zChunk0.fma(simd_t(xChunk0), simd_t(yChunk0));
        zChunk1.fma(simd_t(xChunk1), simd_t(yChunk1));
        zChunk2.fma(simd_t(xChunk2), simd_t(yChunk2));
        zChunk3.fma(simd_t(xChunk3), simd_t(yChunk3));

        // update pointers
        x += 4 * simd_t::width;
        y += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }
    zChunk3.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }
    zChunk2.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        simd_t yChunk0, yChunk1;
        yChunk0.load(y, is_aligned);
        yChunk1.load(y + simd_t::width, is_aligned);

        zChunk0.fma(simd_t(xChunk0), simd_t(yChunk0));
        zChunk1.fma(simd_t(xChunk1), simd_t(yChunk1));

        // update pointers
        x += 2 * simd_t::width;
        y += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }
    zChunk1.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk0;
        xChunk0.load(x, is_aligned);

        simd_t yChunk0;
        yChunk0.load(y, is_aligned);

        zChunk0.fma(simd_t(xChunk0), simd_t(yChunk0));

        // update pointers
        x += simd_t::width;
        y += simd_t::width;
        cnt -= simd_t::width;
    }
    zChunk0.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // spillover loop
    while (cnt)
    {
        *out += (*x) * (*y);
        // update pointers
        ++x;
        ++y;
        --cnt;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void ddotKernel(const size_t nsize, const uint8_t *mask, const TData *x,
                       const TData *y, TData *out,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    // TODO: SIMD/AVX
    if (init)
    {
        *out = 0.0;
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out += mask[i] * x[i] * y[i];
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void l1normKernel(const size_t nsize, const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt     = nsize;
    simd_t yChunk0 = 0, yChunk1 = 0, yChunk2 = 0, yChunk3 = 0;
    alignas(simd_t::alignment) typename simd_t::scalarArray tmp;

    if constexpr (init)
    {
        *out = 0.0;
    }

    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        yChunk0 += abs(xChunk0);
        yChunk1 += abs(xChunk1);
        yChunk2 += abs(xChunk2);
        yChunk3 += abs(xChunk3);

        // update pointers
        x += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }
    yChunk3.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }
    yChunk2.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        yChunk0 += abs(xChunk0);
        yChunk1 += abs(xChunk1);

        // update pointers
        x += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }
    yChunk1.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk0;
        xChunk0.load(x, is_aligned);

        yChunk0 += abs(xChunk0);

        // update pointers
        x += simd_t::width;
        cnt -= simd_t::width;
    }
    yChunk0.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // spillover loop
    while (cnt)
    {
        *out += std::abs(*x);
        // update pointers
        ++x;
        --cnt;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void l1normKernel(const size_t nsize, const uint8_t *mask,
                         const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    // TODO: SIMD/AVX
    if (init)
    {
        *out = 0.0;
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out += mask[i] * std::abs(x[i]);
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void l2normKernel(const size_t nsize, const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt     = nsize;
    simd_t yChunk0 = 0, yChunk1 = 0, yChunk2 = 0, yChunk3 = 0;
    alignas(simd_t::alignment) typename simd_t::scalarArray tmp;

    if constexpr (init)
    {
        *out = 0.0;
    }

    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        yChunk0.fma(simd_t(xChunk0), simd_t(xChunk0));
        yChunk1.fma(simd_t(xChunk1), simd_t(xChunk1));
        yChunk2.fma(simd_t(xChunk2), simd_t(xChunk2));
        yChunk3.fma(simd_t(xChunk3), simd_t(xChunk3));

        // update pointers
        x += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }
    yChunk3.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }
    yChunk2.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        yChunk0.fma(simd_t(xChunk0), simd_t(xChunk0));
        yChunk1.fma(simd_t(xChunk1), simd_t(xChunk1));

        // update pointers
        x += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }
    yChunk1.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk0;
        xChunk0.load(x, is_aligned);

        yChunk0.fma(simd_t(xChunk0), simd_t(xChunk0));

        // update pointers
        x += simd_t::width;
        cnt -= simd_t::width;
    }
    yChunk0.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out += tmp[i];
    }

    // spillover loop
    while (cnt)
    {
        *out += (*x) * (*x);
        // update pointers
        ++x;
        --cnt;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void l2normKernel(const size_t nsize, const uint8_t *mask,
                         const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    // TODO: SIMD/AVX
    if (init)
    {
        *out = 0.0;
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out += mask[i] * x[i] * x[i];
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void lpnormKernel(const size_t nsize, const unsigned int p,
                         const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    // TODO: SIMD/AVX
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(x, x + nsize, initializer,
                                        [&](const TData &acc, const TData &val) {
                               return acc + std::pow(std::abs(val), p);
                           });
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void lpnormKernel(const size_t nsize, const unsigned int p,
                         const uint8_t *mask, const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    // TODO: SIMD/AVX
    if (init)
    {
        *out = 0.0;
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out += mask[i] * std::pow(std::abs(x[i]), p);
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void linfnormKernel(const size_t nsize, const TData *x, TData *out,
                           [[maybe_unused]] const unsigned int streamID = 0)
{
    using namespace tinysimd;
    using simd_t = simd<TData>;

    size_t cnt     = nsize;
    simd_t yChunk0 = 0, yChunk1 = 0, yChunk2 = 0, yChunk3 = 0;
    alignas(simd_t::alignment) typename simd_t::scalarArray tmp;

    if constexpr (init)
    {
        *out = 0;
    }

    // Vectorized loop unroll 4x
    while (cnt >= 4 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1, xChunk2, xChunk3;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);
        xChunk2.load(x + 2 * simd_t::width, is_aligned);
        xChunk3.load(x + 3 * simd_t::width, is_aligned);

        yChunk0 = max(abs(xChunk0), yChunk0);
        yChunk1 = max(abs(xChunk1), yChunk1);
        yChunk2 = max(abs(xChunk2), yChunk2);
        yChunk3 = max(abs(xChunk3), yChunk3);

        // update pointers
        x += 4 * simd_t::width;
        cnt -= 4 * simd_t::width;
    }
    yChunk3.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::max(std::abs(tmp[i]), *out);
    }
    yChunk2.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::max(std::abs(tmp[i]), *out);
    }

    // Vectorized loop unroll 2x
    while (cnt >= 2 * simd_t::width)
    {
        // load
        simd_t xChunk0, xChunk1;
        xChunk0.load(x, is_aligned);
        xChunk1.load(x + simd_t::width, is_aligned);

        yChunk0 = max(abs(xChunk0), yChunk0);
        yChunk1 = max(abs(xChunk1), yChunk1);

        // update pointers
        x += 2 * simd_t::width;
        cnt -= 2 * simd_t::width;
    }
    yChunk1.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::max(std::abs(tmp[i]), *out);
    }

    // Vectorized loop
    while (cnt >= simd_t::width)
    {
        // load
        simd_t xChunk0;
        xChunk0.load(x, is_aligned);

        yChunk0 = max(abs(xChunk0), yChunk0);

        // update pointers
        x += simd_t::width;
        cnt -= simd_t::width;
    }
    yChunk0.store(tmp, is_aligned);
    for (unsigned int i = 0; i < simd_t::width; i++)
    {
        *out = std::max(std::abs(tmp[i]), *out);
    }

    // spillover loop
    while (cnt)
    {
        *out = std::max(std::abs(*x), *out);
        // update pointers
        ++x;
        --cnt;
    }
}

template <typename ExecSpace, bool init, typename TData,
          std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::AVX>, bool>
              Enable = true>
inline void linfnormKernel(const size_t nsize, const uint8_t *mask,
                           const TData *x, TData *out,
                           [[maybe_unused]] const unsigned int streamID = 0)
{
    // TODO: SIMD/AVX
    if (init)
    {
        *out = 0.0;
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out = mask[i] ? std::max(*out, std::abs(x[i])) : *out;
    }
}

} // namespace Nektar::Math
