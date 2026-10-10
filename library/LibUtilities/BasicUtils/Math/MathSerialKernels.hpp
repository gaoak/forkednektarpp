///////////////////////////////////////////////////////////////////////////////
//
// File: MathSerialKernels.hpp
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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <numeric>
#include <type_traits>

namespace Nektar::Math
{

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void zeroKernel(const size_t nsize, TData *x,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    std::memset(x, 0, nsize * sizeof(TData));
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void fillKernel(const size_t nsize, const TData &val, TData *x,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    std::fill(x, x + nsize, val);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void copyKernel(const size_t nsize, const TData *x, TData *y,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    std::memcpy(y, x, nsize * sizeof(TData));
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void absKernel(const size_t nsize, const TData *x, TData *y,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y,
                   [](const TData &xi) { return std::abs(xi); });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void negKernel(const size_t nsize, const TData *x, TData *y,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, std::negate<TData>());
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void sqrtKernel(const size_t nsize, const TData *x, TData *y,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y,
                   [](const TData &xi) { return std::sqrt(xi); });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void addKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, z,
                   [](const TData &xi, const TData &yi) { return xi + yi; });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void addScalarKernel(const size_t nsize, const TData alpha,
                            const TData *x, TData *y,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, [&alpha](TData xi) { return xi + alpha; });
}

// addScalarKernel over the ncomp components of compSize values of a block,
// stored as element groups of width lanes by ndata points of which the first
// nelmt elements are not padding: y = x + alpha, and zero in the padding.
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void addScalarKernel(const unsigned int ncomp, const size_t compSize,
                            const size_t nelmt, const unsigned int ndata,
                            const unsigned int width, const TData alpha,
                            const TData *x, TData *y,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    const size_t prefix      = (nelmt / width) * width * ndata;
    const unsigned int lane0 = nelmt % width;
    for (unsigned int n = 0; n < ncomp; ++n, x += compSize, y += compSize)
    {
        addScalarKernel<ExecSpace>(prefix, alpha, x, y);

        // The element group holding the first padded element, then the
        // element groups after it, all padding.
        size_t i = prefix;
        for (unsigned int pt = 0; lane0 != 0 && pt < ndata; ++pt)
        {
            for (unsigned int l = 0; l < width; ++l, ++i)
            {
                y[i] = (l < lane0) ? x[i] + alpha : TData(0);
            }
        }
        std::fill(y + i, y + compSize, TData(0));
    }
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void subKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, z,
                   [](const TData &xi, const TData &yi) { return xi - yi; });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void mulKernel(const size_t nsize, const TData alpha, const TData *x,
                      TData *y,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, [&alpha](TData xi) { return alpha * xi; });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void mulKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, z,
                   [](TData xi, TData yi) { return xi * yi; });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void divKernel(const size_t nsize, const TData alpha, const TData *x,
                      TData *y,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, [&alpha](TData xi) {
        return (xi == TData(0)) ? TData(0) : alpha / xi;
    });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void divKernel(const size_t nsize, const TData *x, const TData *y,
                      TData *z,
                      [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, z, [](TData xi, TData yi) {
        return (yi == TData(0)) ? TData(0) : xi / yi;
    });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void daxpyKernel(const size_t nsize, const TData alpha, const TData *x,
                        const TData *y, TData *z,
                        [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, z, [&](const TData &xi, const TData &yi) {
        return alpha * xi + yi;
    });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void daxpbyKernel(const size_t nsize, const TData alpha, const TData *x,
                         const TData beta, const TData *y, TData *z,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    std::transform(x, x + nsize, y, z, [&](const TData &xi, const TData &yi) {
        return alpha * xi + beta * yi;
    });
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void daxpbypzKernel(const size_t nsize, const TData alpha,
                           const TData *x, const TData beta, const TData *y,
                           const TData *z, TData *w,
                           [[maybe_unused]] const unsigned int streamID = 0)
{
    for (size_t i = 0; i < nsize; ++i)
    {
        w[i] = alpha * x[i] + beta * y[i] + z[i];
    }
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void daxpbypczKernel(const size_t nsize, const TData alpha,
                            const TData *x, const TData beta, const TData *y,
                            const TData gamma, const TData *z, TData *w,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    for (size_t i = 0; i < nsize; ++i)
    {
        w[i] = alpha * x[i] + beta * y[i] + gamma * z[i];
    }
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void reduceSumKernel(const size_t nsize, const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(x, x + nsize, initializer);
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void reduceSumKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::inner_product(mask, mask + nsize, x, initializer);
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void reduceMaxKernel(const size_t nsize, const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    TData initializer = init ? std::numeric_limits<TData>::lowest() : *out;
    *out = std::max(initializer, *(std::max_element(x, x + nsize)));
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void reduceMaxKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    if (init)
    {
        *out = std::numeric_limits<TData>::lowest();
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out = mask[i] ? std::max(*out, x[i]) : *out;
    }
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void reduceMinKernel(const size_t nsize, const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    TData initializer = init ? std::numeric_limits<TData>::max() : *out;
    *out = std::min(initializer, *(std::min_element(x, x + nsize)));
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void reduceMinKernel(const size_t nsize, const uint8_t *mask,
                            const TData *x, TData *out,
                            [[maybe_unused]] const unsigned int streamID = 0)
{
    if (init)
    {
        *out = std::numeric_limits<TData>::max();
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out = mask[i] ? std::min(*out, x[i]) : *out;
    }
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void ddotKernel(const size_t nsize, const TData *x, const TData *y,
                       TData *out,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::inner_product(x, x + nsize, y, initializer);
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void ddotKernel(const size_t nsize, const uint8_t *mask, const TData *x,
                       const TData *y, TData *out,
                       [[maybe_unused]] const unsigned int streamID = 0)
{
    if (init)
    {
        *out = 0.0;
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out += mask[i] * x[i] * y[i];
    }
}

// ddot2Kernel computes, in one pass over x, out[0] (+)= (x, y) and out[1] (+)=
// (x, z): two dot products sharing x, over the entries mask selects unless it
// is null.
template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void ddot2Kernel(const size_t nsize, const uint8_t *mask, const TData *x,
                        const TData *y, const TData *z, TData *out,
                        [[maybe_unused]] const unsigned int streamID = 0)
{
    TData xy = init ? 0.0 : out[0];
    TData xz = init ? 0.0 : out[1];
    for (size_t i = 0; i < nsize; i++)
    {
        const TData xi = mask ? mask[i] * x[i] : x[i];
        xy += xi * y[i];
        xz += xi * z[i];
    }
    out[0] = xy;
    out[1] = xz;
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void l1normKernel(const size_t nsize, const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(
        x, x + nsize, initializer,
        [](const TData &acc, const TData &val) { return acc + std::abs(val); });
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void l1normKernel(const size_t nsize, const uint8_t *mask,
                         const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    if (init)
    {
        *out = 0.0;
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out += mask[i] * std::abs(x[i]);
    }
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void l2normKernel(const size_t nsize, const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(
        x, x + nsize, initializer,
        [](const TData &acc, const TData &val) { return acc + val * val; });
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void l2normKernel(const size_t nsize, const uint8_t *mask,
                         const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    if (init)
    {
        *out = 0.0;
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out += mask[i] * x[i] * x[i];
    }
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void lpnormKernel(const size_t nsize, const unsigned int p,
                         const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(x, x + nsize, initializer,
                                        [&](const TData &acc, const TData &val) {
                               return acc + std::pow(std::abs(val), p);
                           });
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void lpnormKernel(const size_t nsize, const unsigned int p,
                         const uint8_t *mask, const TData *x, TData *out,
                         [[maybe_unused]] const unsigned int streamID = 0)
{
    if (init)
    {
        *out = 0.0;
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out += mask[i] * std::pow(std::abs(x[i]), p);
    }
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void linfnormKernel(const size_t nsize, const TData *x, TData *out,
                           [[maybe_unused]] const unsigned int streamID = 0)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(x, x + nsize, initializer,
                                        [](const TData &acc, const TData &val) {
                               return std::max(acc, std::abs(val));
                           });
}

template <
    typename ExecSpace, bool init, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial>, bool>
        Enable = true>
inline void linfnormKernel(const size_t nsize, const uint8_t *mask,
                           const TData *x, TData *out,
                           [[maybe_unused]] const unsigned int streamID = 0)
{
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
