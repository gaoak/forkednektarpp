///////////////////////////////////////////////////////////////////////////////
//
// File: MathDeviceOnHostKernels.hpp
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

#if defined(NEKTAR_ENABLE_DEVICEONHOST)

#include "Operators/LoopExecution/LoopExecution.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace Nektar
{

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
absKernel(const size_t nsize, const TData *x, TData *y)
{
    std::transform(x, x + nsize, y,
                   [](const TData &xi) { return std::abs(xi); });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
negKernel(const size_t nsize, const TData *x, TData *y)
{
    std::transform(x, x + nsize, y, std::negate<TData>());
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
sqrtKernel(const size_t nsize, const TData *x, TData *y)
{
    std::transform(x, x + nsize, y,
                   [](const TData &xi) { return std::sqrt(xi); });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
addKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](const TData &xi, const TData &yi) { return xi + yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
subKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](const TData &xi, const TData &yi) { return xi - yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
mulKernel(const size_t nsize, const TData alpha, const TData *x, TData *y)
{
    std::transform(x, x + nsize, y, [&alpha](TData xi) { return alpha * xi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
mulKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](TData xi, TData yi) { return xi * yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
divKernel(const size_t nsize, const TData alpha, const TData *x, TData *y)
{
    std::transform(x, x + nsize, y, [&alpha](TData xi) { return alpha / xi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
divKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](TData xi, TData yi) { return xi / yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
daxpyKernel(const size_t nsize, const TData alpha, const TData *x,
            const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z, [&](const TData &xi, const TData &yi) {
        return alpha * xi + yi;
    });
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceSumKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(x, x + nsize, initializer);
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceSumKernel(const size_t nsize, const uint8_t *mask, const TData *x,
                TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::inner_product(mask, mask + nsize, x, initializer);
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMaxKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? std::numeric_limits<TData>::min() : *out;
    *out = std::max(initializer, *(std::max_element(x, x + nsize)));
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMaxKernel(const size_t nsize, const uint8_t *mask, const TData *x,
                TData *out)
{
    if (init)
    {
        *out = std::numeric_limits<TData>::min();
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out = mask[i] ? std::max(*out, x[i]) : *out;
    }
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMinKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? std::numeric_limits<TData>::min() : *out;
    *out = std::min(initializer, *(std::min_element(x, x + nsize)));
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
reduceMinKernel(const size_t nsize, const uint8_t *mask, const TData *x,
                TData *out)
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

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
ddotKernel(const size_t nsize, const TData *x, const TData *y, TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::inner_product(x, x + nsize, y, initializer);
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
ddotKernel(const size_t nsize, const uint8_t *mask, const TData *x,
           const TData *y, TData *out)
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

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l1normKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(
        x, x + nsize, initializer,
        [](const TData &acc, const TData &val) { return acc + std::abs(val); });
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l1normKernel(const size_t nsize, const uint8_t *mask, const TData *x,
             TData *out)
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

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l2normKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(
        x, x + nsize, initializer,
        [](const TData &acc, const TData &val) { return acc + val * val; });
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
l2normKernel(const size_t nsize, const uint8_t *mask, const TData *x,
             TData *out)
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

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
lpnormKernel(const size_t nsize, const unsigned int p, const TData *x,
             TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(x, x + nsize, initializer,
                                        [&](const TData &acc, const TData &val) {
                               return acc + std::pow(std::abs(val), p);
                           });
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
lpnormKernel(const size_t nsize, const unsigned int p, const uint8_t *mask,
             const TData *x, TData *out)
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

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
linfnormKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? std::numeric_limits<TData>::min() : *out;
    *out              = std::accumulate(x, x + nsize, initializer,
                                        [](const TData &acc, const TData &val) {
                               return std::max(acc, std::abs(val));
                           });
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
linfnormKernel(const size_t nsize, const uint8_t *mask, const TData *x,
               TData *out)
{
    if (init)
    {
        *out = std::numeric_limits<TData>::min();
    }

    for (size_t i = 0; i < nsize; i++)
    {
        *out = mask[i] ? std::max(*out, std::abs(x[i])) : *out;
    }
}

} // namespace Nektar

#endif
