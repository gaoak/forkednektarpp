///////////////////////////////////////////////////////////////////////////////
//
// File: MathKokkosKernels.hpp
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

#if defined(NEKTAR_ENABLE_KOKKOS)

#include "Operators/LoopExecution/LoopExecution.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace Nektar
{

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
negKernel(const unsigned int nsize, const TData *x, TData *y)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, KOKKOS_LAMBDA(const unsigned int i) { y[i] = -x[i]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
addKernel(const unsigned int nsize, const TData *x, const TData *y, TData *z)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, KOKKOS_LAMBDA(const unsigned int i) { z[i] = x[i] + y[i]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
subKernel(const unsigned int nsize, const TData *x, const TData *y, TData *z)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, KOKKOS_LAMBDA(const unsigned int i) { z[i] = x[i] - y[i]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
daxpyKernel(const unsigned int nsize, const TData alpha, const TData *x,
            const TData *y, TData *z)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize,
        KOKKOS_LAMBDA(const unsigned int i) { z[i] = alpha * x[i] + y[i]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
divKernel(const unsigned int nsize, const TData *x, const TData *y, TData *z)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, KOKKOS_LAMBDA(const unsigned int i) { z[i] = x[i] / y[i]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
reduceSumKernel(const unsigned int nsize, const TData *x, TData *out)
{
    Nektar::parallel_reduce<ExecSpace, Nektar::ReduceSum<TData>>(
        0, nsize,
        KOKKOS_LAMBDA(const unsigned int i, TData &sum) { sum += x[i]; }, *out);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
reduceMaxKernel(const unsigned int nsize, const TData *x, TData *out)
{
    Nektar::parallel_reduce<ExecSpace, Nektar::ReduceMax<TData>>(
        0, nsize,
        KOKKOS_LAMBDA(const unsigned int i, TData &max) {
            max = std::max(max, x[i]);
        },
        *out);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
reduceMinKernel(const unsigned int nsize, const TData *x, TData *out)
{
    Nektar::parallel_reduce<ExecSpace, Nektar::ReduceMin<TData>>(
        0, nsize,
        KOKKOS_LAMBDA(const unsigned int i, TData &min) {
            min = std::min(min, x[i]);
        },
        *out);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
ddotKernel(const unsigned int nsize, const TData *x, const TData *y, TData *out)
{
    Nektar::parallel_reduce<ExecSpace, Nektar::ReduceSum<TData>>(
        0, nsize,
        KOKKOS_LAMBDA(const unsigned int i, TData &sum) { sum += x[i] * y[i]; },
        *out);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
l1normKernel(const unsigned int nsize, const TData *x, TData *out)
{
    Nektar::parallel_reduce<ExecSpace, Nektar::ReduceSum<TData>>(
        0, nsize,
        KOKKOS_LAMBDA(const unsigned int i, TData &sum) {
            sum += std::abs(x[i]);
        },
        *out);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
l2normKernel(const unsigned int nsize, const TData *x, TData *out)
{
    Nektar::parallel_reduce<ExecSpace, Nektar::ReduceSum<TData>>(
        0, nsize,
        KOKKOS_LAMBDA(const unsigned int i, TData &sum) { sum += x[i] * x[i]; },
        *out);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
lpnormKernel(const unsigned int nsize, const unsigned int p, const TData *x,
             TData *out)
{
    Nektar::parallel_reduce<ExecSpace, Nektar::ReduceSum<TData>>(
        0, nsize,
        KOKKOS_LAMBDA(const unsigned int i, TData &sum) {
            sum += std::pow(std::abs(x[i]), p);
        },
        *out);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
linfnormKernel(const unsigned int nsize, const TData *x, TData *out)
{
    Nektar::parallel_reduce<ExecSpace, Nektar::ReduceMax<TData>>(
        0, nsize,
        KOKKOS_LAMBDA(const unsigned int i, TData &max) {
            max = std::max(max, std::abs(x[i]));
        },
        *out);
}

} // namespace Nektar

#endif
