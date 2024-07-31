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

namespace Nektar //::Operators
{

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
negKernel(const unsigned int nsize, const TData *x, TData *y)
{
    std::transform(x, x + nsize, y, std::negate<TData>());
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
addKernel(const unsigned int nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](const TData &xi, const TData &yi) { return xi + yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
subKernel(const unsigned int nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](const TData &xi, const TData &yi) { return xi - yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
daxpyKernel(const unsigned int nsize, const TData alpha, const TData *x,
            const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z, [&](const TData &xi, const TData &yi) {
        return alpha * xi + yi;
    });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
divKernel(const unsigned int nsize, const TData *x, const TData *y, TData *z)
{
    std::transform(x, x + nsize, y, z,
                   [](TData xi, TData yi) { return xi / yi; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
reduceSumKernel(const unsigned int nsize, const TData *x, TData *out)
{
    *out = std::accumulate(x, x + nsize, (TData)0.0);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
reduceMaxKernel(const unsigned int nsize, const TData *x, TData *out)
{
    *out = *(std::max_element(x, x + nsize));
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
reduceMinKernel(const unsigned int nsize, const TData *x, TData *out)
{
    *out = *(std::min_element(x, x + nsize));
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
ddotKernel(const unsigned int nsize, const TData *x, const TData *y, TData *out)
{
    *out = std::inner_product(x, x + nsize, y, 0.0);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
l1normKernel(const unsigned int nsize, const TData *x, TData *out)
{
    *out = std::accumulate(
        x, x + nsize, (TData)0.0,
        [](const TData &acc, const TData &val) { return acc + std::abs(val); });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
l2normKernel(const unsigned int nsize, const TData *x, TData *out)
{
    *out = std::accumulate(
        x, x + nsize, (TData)0.0,
        [](const TData &acc, const TData &val) { return acc + val * val; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
lpnormKernel(const unsigned int nsize, const unsigned int p, const TData *x,
             TData *out)
{
    *out = std::accumulate(x, x + nsize, (TData)0.0,
                           [&](const TData &acc, const TData &val) {
                               return acc + std::pow(std::abs(val), p);
                           });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value, void>::type
linfnormKernel(const unsigned int nsize, const TData *x, TData *out)
{
    *out = std::accumulate(x, x + nsize, 0.0,
                           [](const TData &acc, const TData &val) {
                               return std::max(acc, std::abs(val));
                           });
}

} // namespace Nektar
