///////////////////////////////////////////////////////////////////////////////
//
// File: MathKernels.hpp
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

#include "Operators/CUDAMathKernels.cuh"
#include "Operators/LoopExecution.hpp"

#include <cstddef>
#include <type_traits>

// #define NEKTAR_USE_STD_TRANSFORM

namespace Nektar::Operators
{
template <typename ExecSpace, typename TData>
inline typename std::enable_if<
#if !defined(NEKTAR_USE_STD_TRANSFORM)
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
#endif
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
addKernel([[maybe_unused]] const size_t gridSize,
          [[maybe_unused]] const size_t blockSize, const size_t nloc,
          const TData *addend1, const TData *addend2, TData *sum)
{
    Nektar::parallel_for<ExecSpace>(
        0, nloc, KOKKOS_LAMBDA(int i) { sum[i] = addend1[i] + addend2[i]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
#if !defined(NEKTAR_USE_STD_TRANSFORM)
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
#endif
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
divKernel([[maybe_unused]] const size_t gridSize,
          [[maybe_unused]] const size_t blockSize, const size_t nloc,
          const TData *numerator, const TData *denominator, TData *quotient)
{
    Nektar::parallel_for<ExecSpace>(
        0, nloc,
        KOKKOS_LAMBDA(int i) { quotient[i] = numerator[i] / denominator[i]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
#if !defined(NEKTAR_USE_STD_TRANSFORM)
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
#endif
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
negKernel([[maybe_unused]] const size_t gridSize,
          [[maybe_unused]] const size_t blockSize, const size_t nloc,
          const TData *in, TData *out)
{
    Nektar::parallel_for<ExecSpace>(
        0, nloc, KOKKOS_LAMBDA(int i) { out[i] = -in[i]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
#if !defined(NEKTAR_USE_STD_TRANSFORM)
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
#endif
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
subKernel([[maybe_unused]] const size_t gridSize,
          [[maybe_unused]] const size_t blockSize, const size_t nloc,
          const TData *subtrahend, const TData *minuend, TData *difference)
{
    Nektar::parallel_for<ExecSpace>(
        0, nloc,
        KOKKOS_LAMBDA(int i) { difference[i] = subtrahend[i] - minuend[i]; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
daxpyKernel([[maybe_unused]] const size_t gridSize,
            [[maybe_unused]] const size_t blockSize, const unsigned int nsize,
            const TData alpha, const TData *x, const TData *y, TData *z)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, KOKKOS_LAMBDA(int i) { z[i] = alpha * x[i] + y[i]; });
}

template <typename ExecSpace, typename TData, int iBlockSize>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
dotKernel([[maybe_unused]] const size_t gridSize,
          [[maybe_unused]] const size_t blockSize, const unsigned int nsize,
          const TData *x, const TData *y, TData *out)
{
    NEKERROR(Nektar::ErrorUtil::efatal,
             "The MathKernels dotKernel is not implemented.");

    // Do something dumb to prevent warnings
    Nektar::parallel_for<ExecSpace>(
        0, nsize, KOKKOS_LAMBDA(int i) { out[i] = x[i] + y[i]; });
}

template <typename ExecSpace, typename TData, int iBlockSize>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::Serial>::value ||
        std::is_same<ExecSpace, NektarSpaces::AVX>::value ||
        std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value,
    void>::type
reduceKernel([[maybe_unused]] const size_t gridSize,
             [[maybe_unused]] const size_t blockSize, const unsigned int nsize,
             const TData *x, TData *out)
{
    NEKERROR(Nektar::ErrorUtil::efatal,
             "The MathKernels reduceKernel is not implemented.");

    // Do something dumb to prevent warnings
    Nektar::parallel_for<ExecSpace>(
        0, nsize, KOKKOS_LAMBDA(int i) { out[i] = x[i]; });
}

// Std transform versions
#if defined(NEKTAR_USE_STD_TRANSFORM)

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace,
                 NektarSpaces::Serial ||
                     std::is_same<ExecSpace, NektarSpaces::AVX>::value>::value,
    void>::type
addKernel([[maybe_unused]] const size_t gridSize,
          [[maybe_unused]] const size_t blockSize, const size_t nloc,
          const TData *addend1, const TData *addend2, TData *sum)
{
    std::transform(addend1, addend1 + nloc, addend2, sum,
                   [](const TData &x, const TData &y) { return x + y; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace,
                 NektarSpaces::Serial ||
                     std::is_same<ExecSpace, NektarSpaces::AVX>::value>::value,
    void>::type
divKernel([[maybe_unused]] const size_t gridSize,
          [[maybe_unused]] const size_t blockSize, const size_t nloc,
          const TData *numerator, const TData *denominator, TData *quotient)
{
    std::transform(numerator, numerator + nloc, denominator, quotient,
                   [](TData num, TData denom) { return num / denom; });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace,
                 NektarSpaces::Serial ||
                     std::is_same<ExecSpace, NektarSpaces::AVX>::value>::value,
    void>::type
negKernel([[maybe_unused]] const size_t gridSize,
          [[maybe_unused]] const size_t blockSize, const size_t nloc,
          const TData *in, TData *out)
{
    std::transform(in, in + nloc, out, std::negate<TData>());
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same<ExecSpace,
                 NektarSpaces::Serial ||
                     std::is_same<ExecSpace, NektarSpaces::AVX>::value>::value,
    void>::type
subKernel([[maybe_unused]] const size_t gridSize,
          [[maybe_unused]] const size_t blockSize, const size_t nloc,
          const TData *subtrahend, const TData *minuend, TData *difference)
{
    std::transform(subtrahend, subtrahend + nloc, minuend, difference,
                   [](const TData &x, const TData &y) { return x - y; });
}

#endif

} // namespace Nektar::Operators
