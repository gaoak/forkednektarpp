///////////////////////////////////////////////////////////////////////////////
//
// File: AdamsMoultonKernelLaunchers.hpp
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
NEK_DEVICE_INLINE static void ExtrapolateAdamsMoultonKernelImpl(
    const size_t idx, const TData dt, TData *__restrict inout,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... implicits)
{
    constexpr unsigned int intOrder = sizeof...(implicits) + 1;

    // 2nd Order
    if constexpr (intOrder == 2)
    {
        constexpr TData coeff[] = {1.0 / 2.0};

        inout[idx] += dt * ((implicits[idx] * coeff[ind]) + ...);
    }
    // 3rd Order
    else if constexpr (intOrder == 3)
    {
        constexpr TData coeff[] = {8.0 / 12.0, -1.0 / 12.0};

        inout[idx] += dt * ((implicits[idx] * coeff[ind]) + ...);
    }
    // 4th Order
    else if constexpr (intOrder == 4)
    {
        constexpr TData coeff[] = {19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0};

        inout[idx] += dt * ((implicits[idx] * coeff[ind]) + ...);
    }
}

// Kernel Launchers.
#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)
// Currently, argument pack can't be captured in a device lambda. Explicit
// kernel must be used (instead of nektar::parallel_for)
template <typename TData, typename... TDatas>
__global__ void ExtrapolateAdamsMoultonKernelLauncher(
    const size_t nsize, const TData dt, TData *__restrict inout,
    const TDatas *__restrict... implicits)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        ExtrapolateAdamsMoultonKernelImpl(
            idx, dt, inout,
            std::make_integer_sequence<unsigned int, sizeof...(implicits)>(),
            implicits...);
    }
}

template <typename ExecSpace, typename TData, typename... TDatas>
NEK_FORCE_INLINE static void ExtrapolateAdamsMoultonKernel(
    const size_t nsize, const TData dt, TData *inout,
    const TDatas *...implicits)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    ExtrapolateAdamsMoultonKernelLauncher<<<gridSize, blockSize>>>(
        nsize, dt, inout, implicits...);
    CHECK_LAST_HIPCUDA_ERROR();
}
#else
template <typename ExecSpace, typename TData, typename... TDatas>
NEK_FORCE_INLINE static void ExtrapolateAdamsMoultonKernel(
    const size_t nsize, const TData dt, TData *inout,
    const TDatas *...implicits)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            ExtrapolateAdamsMoultonKernelImpl(
                idx, dt, inout,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(implicits)>(),
                implicits...);
        });
}
#endif

} // namespace Nektar::Operators::detail
