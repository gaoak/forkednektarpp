///////////////////////////////////////////////////////////////////////////////
//
// File: IMEXKernelLaunchers.hpp
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

namespace Nektar::Operators::detail
{

template <typename TData, unsigned int... ind, typename... TDatas>
NEK_DEVICE_INLINE static void ExtrapolateIMEXKernelImpl(
    const size_t idx, TData *__restrict inout,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... solutions)
{
    constexpr unsigned int nSolution = sizeof...(solutions);
    constexpr unsigned int intOrder  = (nSolution + 1) / 2;

    // 1st order
    if constexpr (intOrder == 1)
    {
        constexpr TData coeff[] = {1.0, 1.0};

        TData tmp = ((solutions[idx] * coeff[ind]) + ...);
        tmp += inout[idx] * coeff[nSolution];
        inout[idx] = tmp;
    }
    // 2nd order
    else if constexpr (intOrder == 2)
    {
        // Based on Karniadakis, Israeli and Orszag 1991, table IV
        constexpr TData coeff[] = {4.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0,
                                   -1.0 / 3.0};

        TData tmp = ((solutions[idx] * coeff[ind]) + ...);
        tmp += inout[idx] * coeff[nSolution];
        inout[idx] = tmp;
    }
    // 3rd order
    else if constexpr (intOrder == 3)
    {
        // Based on Karniadakis, Israeli and Orszag 1991, table IV
        constexpr TData coeff[] = {18.0 / 11.0, -18.0 / 11.0, 6.0 / 11.0,
                                   18.0 / 11.0, -9.0 / 11.0,  2.0 / 11.0};

        TData tmp = ((solutions[idx] * coeff[ind]) + ...);
        tmp += inout[idx] * coeff[nSolution];
        inout[idx] = tmp;
    }
    // 4th order
    else if constexpr (intOrder == 4)
    {
        // Based on Asher, Ruuth and Wetton 1995, equation (33)
        constexpr TData coeff[] = {48.0 / 25.0,  -72.0 / 25.0, 48.0 / 25.0,
                                   -12.0 / 25.0, 48.0 / 25.0,  -36.0 / 25.0,
                                   16.0 / 25.0,  -3.0 / 25.0};

        TData tmp = ((solutions[idx] * coeff[ind]) + ...);
        tmp += inout[idx] * coeff[nSolution];
        inout[idx] = tmp;
    }
}

// Kernel Launchers.
#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)
// Currently, argument pack can't be captured in a device lambda. Explicit
// kernel must be used (instead of nektar::parallel_for)
template <typename TData, typename... TDatas>
__global__ void ExtrapolateIMEXKernelLauncher(const size_t nsize,
                                              TData *__restrict inout,
                                              TDatas *__restrict... solutions)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        ExtrapolateIMEXKernelImpl(
            idx, inout,
            std::make_integer_sequence<unsigned int, sizeof...(solutions)>(),
            solutions...);
    }
}

template <typename ExecSpace, typename TData, typename... TDatas>
NEK_FORCE_INLINE static void ExtrapolateIMEXKernel(const size_t nsize,
                                                   TData *inout,
                                                   const TDatas *...solutions)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    ExtrapolateIMEXKernelLauncher<<<gridSize, blockSize>>>(nsize, inout,
                                                           solutions...);
    CHECK_LAST_HIPCUDA_ERROR();
}
#else
template <typename ExecSpace, typename TData, typename... TDatas>
NEK_FORCE_INLINE static void ExtrapolateIMEXKernel(const size_t nsize,
                                                   TData *inout,
                                                   const TDatas *...solutions)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            ExtrapolateIMEXKernelImpl(
                idx, inout,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(solutions)>(),
                solutions...);
        });
}
#endif

} // namespace Nektar::Operators::detail
