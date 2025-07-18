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

#include "Operators/LoopExecution/LoopExecution.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <type_traits>

namespace Nektar
{

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
negKernel(const size_t nsize, const TData *x, TData *y)
{
    std::transform(x, x + nsize, y, std::negate<TData>());
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
addKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](const TData &xi, const TData &yi) { return xi + yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
subKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](const TData &xi, const TData &yi) { return xi - yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
mulKernel(const size_t nsize, const TData alpha, const TData *x, TData *y)
{
    std::transform(x, x + nsize, y, [&alpha](TData xi) { return alpha * xi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
mulKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](TData xi, TData yi) { return xi * yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
divKernel(const size_t nsize, const TData alpha, const TData *x, TData *y)
{
    std::transform(x, x + nsize, y, [&alpha](TData xi) { return alpha / xi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
divKernel(const size_t nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](TData xi, TData yi) { return xi / yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
daxpyKernel(const size_t nsize, const TData alpha, const TData *x,
            const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z, [&](const TData &xi, const TData &yi) {
        return alpha * xi + yi;
    });
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
reduceSumKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(x, x + nsize, initializer);
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
reduceMaxKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? std::numeric_limits<TData>::min() : *out;
    *out = std::max(initializer, *(std::max_element(x, x + nsize)));
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
reduceMinKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? std::numeric_limits<TData>::min() : *out;
    *out = std::min(initializer, *(std::min_element(x, x + nsize)));
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
ddotKernel(const size_t nsize, const TData *x, const TData *y, TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::inner_product(x, x + nsize, y, initializer);
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
l1normKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(
        x, x + nsize, initializer,
        [](const TData &acc, const TData &val) { return acc + std::abs(val); });
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
l2normKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? 0.0 : *out;
    *out              = std::accumulate(
        x, x + nsize, initializer,
        [](const TData &acc, const TData &val) { return acc + val * val; });
}

template <typename ExecSpace, bool init, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
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
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                               void>::type
linfnormKernel(const size_t nsize, const TData *x, TData *out)
{
    TData initializer = init ? std::numeric_limits<TData>::min() : *out;
    *out              = std::accumulate(x, x + nsize, initializer,
                                        [](const TData &acc, const TData &val) {
                               return std::max(acc, std::abs(val));
                           });
}

} // namespace Nektar
