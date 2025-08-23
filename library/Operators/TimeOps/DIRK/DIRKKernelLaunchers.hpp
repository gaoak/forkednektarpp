///////////////////////////////////////////////////////////////////////////////
//
// File: DIRKKernelLaunchers.hpp
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

template <unsigned int IntOrder, typename TData, unsigned int... ind,
          typename... TDatas>
NEK_DEVICE_INLINE static void StageSolutionDIRKKernelImpl(
    const size_t idx, TData *__restrict inout, const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... implicits)
{
    constexpr unsigned int stage    = sizeof...(implicits);
    constexpr unsigned int indStart = (stage * (stage - 1)) / 2;

    // 2nd order
    if constexpr (IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData lambda     = (2.0 - ConstSqrt2) / 2.0;

        // clang-format off
        constexpr TData coeff[] =
                { 1.0 - lambda };
        // clang-format on

        inout[idx] =
            solution[idx] + ((implicits[idx] * coeff[indStart + ind]) + ...);
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        constexpr TData lambda = 0.4358665215;

        // clang-format off
        constexpr TData coeff[] =
                {  // 1st stage
                   0.5 * (1.0 - lambda),
                   // 2nd stage
                   0.25 * (-6.0 * lambda * lambda + 16.0 * lambda - 1.0), 
                   0.25 * (6.0 * lambda * lambda - 20.0 * lambda + 5.0)
                };
        // clang-format on

        inout[idx] =
            solution[idx] + ((implicits[idx] * coeff[indStart + ind]) + ...);
    }
}

template <unsigned int IntOrder, typename TData, unsigned int... ind,
          typename... TDatas>
NEK_DEVICE_INLINE static void UpdateDIRKKernelImpl(
    const size_t idx, TData *__restrict inout, const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    const TDatas *__restrict... implicits)
{
    // 1st order
    if constexpr (IntOrder == 1)
    {
        constexpr TData coeff[] = {1.0};

        inout[idx] = solution[idx] + ((implicits[idx] * coeff[ind]) + ...);
    }
    // 2nd order
    else if constexpr (IntOrder == 2)
    {
        constexpr TData ConstSqrt2 = 1.414213562373095;
        constexpr TData lambda     = (2.0 - ConstSqrt2) / 2.0;

        constexpr TData coeff[] = {1.0 - lambda, lambda};

        inout[idx] = solution[idx] + ((implicits[idx] * coeff[ind]) + ...);
    }
    // 3rd order
    else if constexpr (IntOrder == 3)
    {
        constexpr TData lambda = 0.4358665215;

        constexpr TData coeff[] = {
            0.25 * (-6.0 * lambda * lambda + 16.0 * lambda - 1.0),
            0.25 * (6.0 * lambda * lambda - 20.0 * lambda + 5.0), lambda};

        inout[idx] = solution[idx] + ((implicits[idx] * coeff[ind]) + ...);
    }
}

// Kernel Launchers.
#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)
// Currently, argument pack can't be captured in a device lambda. Explicit
// kernel must be used (instead of nektar::parallel_for)
template <unsigned int IntOrder, typename TData, typename... TDatas>
__global__ void StageSolutionDIRKKernelLauncher(
    const size_t nsize, TData *__restrict inout,
    const TData *__restrict solution, const TDatas *__restrict... implicits)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        StageSolutionDIRKKernelImpl<IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(implicits)>(),
            implicits...);
    }
}

template <unsigned int IntOrder, typename TData, typename... TDatas>
__global__ void UpdateDIRKKernelLauncher(const size_t nsize,
                                         TData *__restrict inout,
                                         const TData *__restrict solution,
                                         const TDatas *__restrict... implicits)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        UpdateDIRKKernelImpl<IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(implicits)>(),
            implicits...);
    }
}

template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void StageSolutionDIRKKernel(const size_t nsize,
                                                     TData *inout,
                                                     const TData *solution,
                                                     const TDatas *...implicits)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    StageSolutionDIRKKernelLauncher<IntOrder>
        <<<gridSize, blockSize>>>(nsize, inout, solution, implicits...);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateDIRKKernel(const size_t nsize, TData *inout,
                                              const TData *solution,
                                              const TDatas *...implicits)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    UpdateDIRKKernelLauncher<IntOrder>
        <<<gridSize, blockSize>>>(nsize, inout, solution, implicits...);
    CHECK_LAST_HIPCUDA_ERROR();
}
#else
template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void StageSolutionDIRKKernel(const size_t nsize,
                                                     TData *inout,
                                                     const TData *solution,
                                                     const TDatas *...implicits)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            StageSolutionDIRKKernelImpl<IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(implicits)>(),
                implicits...);
        });
}

template <typename ExecSpace, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateDIRKKernel(const size_t nsize, TData *inout,
                                              const TData *solution,
                                              const TDatas *...implicits)
{
    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            UpdateDIRKKernelImpl<IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(implicits)>(),
                implicits...);
        });
}
#endif

} // namespace Nektar::Operators::detail
