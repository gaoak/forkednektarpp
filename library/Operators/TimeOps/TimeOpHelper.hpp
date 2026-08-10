///////////////////////////////////////////////////////////////////////////////
//
// File: TimeOpHelper.hpp
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

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

class Noscheme;

template <
    typename Scheme, unsigned int IntOrder, typename TData, unsigned int... ind,
    typename... TDatas,
    std::enable_if_t<std::is_same_v<Scheme, Noscheme>, bool> Enable = true>
NEK_DEVICE_INLINE static void UpdateStageKernelImpl(
    [[maybe_unused]] const size_t idx, [[maybe_unused]] TData *__restrict inout,
    [[maybe_unused]] const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    [[maybe_unused]] const TDatas *__restrict... solutions)
{
}

template <
    typename Scheme, unsigned int ImpStage, unsigned int ExpStage,
    unsigned int IntOrder, typename TData, unsigned int... ind,
    typename... TDatas,
    std::enable_if_t<std::is_same_v<Scheme, Noscheme>, bool> Enable = true>
NEK_DEVICE_INLINE static void UpdateStageKernelImpl(
    [[maybe_unused]] const size_t idx, [[maybe_unused]] TData *__restrict inout,
    [[maybe_unused]] const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    [[maybe_unused]] const TDatas *__restrict... solutions)
{
}

template <
    typename Scheme, typename TData, unsigned int... ind, typename... TDatas,
    std::enable_if_t<std::is_same_v<Scheme, Noscheme>, bool> Enable = true>
NEK_DEVICE_INLINE static void UpdateSolutionKernelImpl(
    [[maybe_unused]] const size_t idx, [[maybe_unused]] TData *__restrict inout,
    std::integer_sequence<unsigned int, ind...>,
    [[maybe_unused]] const TDatas *__restrict... solutions)
{
}

template <
    typename Scheme, unsigned int IntOrder, typename TData, unsigned int... ind,
    typename... TDatas,
    std::enable_if_t<std::is_same_v<Scheme, Noscheme>, bool> Enable = true>
NEK_DEVICE_INLINE static void UpdateSolutionKernelImpl(
    [[maybe_unused]] const size_t idx, [[maybe_unused]] TData *__restrict inout,
    [[maybe_unused]] const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    [[maybe_unused]] const TDatas *__restrict... solutions)
{
}

template <
    typename Scheme, unsigned int ImpStage, unsigned int ExpStage,
    unsigned int IntOrder, typename TData, unsigned int... ind,
    typename... TDatas,
    std::enable_if_t<std::is_same_v<Scheme, Noscheme>, bool> Enable = true>
NEK_DEVICE_INLINE static void UpdateSolutionKernelImpl(
    [[maybe_unused]] const size_t idx, [[maybe_unused]] TData *__restrict inout,
    [[maybe_unused]] const TData *__restrict solution,
    std::integer_sequence<unsigned int, ind...>,
    [[maybe_unused]] const TDatas *__restrict... solutions)
{
}

// Kernel Launchers.
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
// Currently, argument pack can't be captured in a device lambda. Explicit
// kernel must be used (instead of nektar::parallel_for)
template <typename Scheme, unsigned int IntOrder, typename TData,
          typename... TDatas>
__global__ void UpdateStageKernelLauncher(const size_t nsize,
                                          TData *__restrict inout,
                                          const TData *__restrict solution,
                                          const TDatas *__restrict... solutions)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        UpdateStageKernelImpl<Scheme, IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(solutions)>(),
            solutions...);
    }
}

template <typename Scheme, unsigned int ImpStage, unsigned int ExpStage,
          unsigned int IntOrder, typename TData, typename... TDatas>
__global__ void UpdateStageKernelLauncher(const size_t nsize,
                                          TData *__restrict inout,
                                          const TData *__restrict solution,
                                          const TDatas *__restrict... solutions)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        UpdateStageKernelImpl<Scheme, ImpStage, ExpStage, IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(solutions)>(),
            solutions...);
    }
}

template <typename Scheme, typename TData, typename... TDatas>
__global__ void UpdateSolutionKernelLauncher(
    const size_t nsize, TData *__restrict inout,
    const TDatas *__restrict... solutions)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        UpdateSolutionKernelImpl<Scheme>(
            idx, inout,
            std::make_integer_sequence<unsigned int, sizeof...(solutions)>(),
            solutions...);
    }
}

template <typename Scheme, unsigned int IntOrder, typename TData,
          typename... TDatas>
__global__ void UpdateSolutionKernelLauncher(
    const size_t nsize, TData *__restrict inout,
    const TData *__restrict solution, const TDatas *__restrict... solutions)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        UpdateSolutionKernelImpl<Scheme, IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(solutions)>(),
            solutions...);
    }
}

template <typename Scheme, unsigned int ImpStage, unsigned int ExpStage,
          unsigned int IntOrder, typename TData, typename... TDatas>
__global__ void UpdateSolutionKernelLauncher(
    const size_t nsize, TData *__restrict inout,
    const TData *__restrict solution, const TDatas *__restrict... solutions)
{
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        UpdateSolutionKernelImpl<Scheme, ImpStage, ExpStage, IntOrder>(
            idx, inout, solution,
            std::make_integer_sequence<unsigned int, sizeof...(solutions)>(),
            solutions...);
    }
}

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData, typename... TDatas>
NEK_FORCE_INLINE static void UpdateStageKernel(const unsigned int streamID,
                                               const size_t nsize, TData *inout,
                                               const TData *solution,
                                               const TDatas *...solutions)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    UpdateStageKernelLauncher<Scheme, IntOrder>
        <<<gridSize, blockSize, 0, stream>>>(nsize, inout, solution,
                                             solutions...);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename Scheme, unsigned int ImpStage,
          unsigned int ExpStage, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateStageKernel(const unsigned int streamID,
                                               const size_t nsize, TData *inout,
                                               const TData *solution,
                                               const TDatas *...solutions)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    UpdateStageKernelLauncher<Scheme, ImpStage, ExpStage, IntOrder>
        <<<gridSize, blockSize, 0, stream>>>(nsize, inout, solution,
                                             solutions...);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename Scheme, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateSolutionKernel(const unsigned int streamID,
                                                  const size_t nsize,
                                                  TData *inout,
                                                  const TDatas *...solutions)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    UpdateSolutionKernelLauncher<Scheme>
        <<<gridSize, blockSize, 0, stream>>>(nsize, inout, solutions...);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData, typename... TDatas>
NEK_FORCE_INLINE static void UpdateSolutionKernel(const unsigned int streamID,
                                                  const size_t nsize,
                                                  TData *inout,
                                                  const TData *solution,
                                                  const TDatas *...solutions)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    UpdateSolutionKernelLauncher<Scheme, IntOrder>
        <<<gridSize, blockSize, 0, stream>>>(nsize, inout, solution,
                                             solutions...);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename Scheme, unsigned int ImpStage,
          unsigned int ExpStage, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateSolutionKernel(const unsigned int streamID,
                                                  const size_t nsize,
                                                  TData *inout,
                                                  const TData *solution,
                                                  const TDatas *...solutions)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    UpdateSolutionKernelLauncher<Scheme, ImpStage, ExpStage, IntOrder>
        <<<gridSize, blockSize, 0, stream>>>(nsize, inout, solution,
                                             solutions...);
    CHECK_LAST_HIPCUDA_ERROR();
}

#else
template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData, typename... TDatas>
NEK_FORCE_INLINE static void UpdateStageKernel(const unsigned int streamID,
                                               const size_t nsize, TData *inout,
                                               const TData *solution,
                                               const TDatas *...solutions)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            UpdateStageKernelImpl<Scheme, IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(solutions)>(),
                solutions...);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename Scheme, unsigned int ImpStage,
          unsigned int ExpStage, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateStageKernel(const unsigned int streamID,
                                               const size_t nsize, TData *inout,
                                               const TData *solution,
                                               const TDatas *...solutions)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            UpdateStageKernelImpl<Scheme, ImpStage, ExpStage, IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(solutions)>(),
                solutions...);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename Scheme, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateSolutionKernel(const unsigned int streamID,
                                                  const size_t nsize,
                                                  TData *inout,
                                                  const TDatas *...solutions)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            UpdateSolutionKernelImpl<Scheme>(
                idx, inout,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(solutions)>(),
                solutions...);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename Scheme, unsigned int IntOrder,
          typename TData, typename... TDatas>
NEK_FORCE_INLINE static void UpdateSolutionKernel(const unsigned int streamID,
                                                  const size_t nsize,
                                                  TData *inout,
                                                  const TData *solution,
                                                  const TDatas *...solutions)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            UpdateSolutionKernelImpl<Scheme, IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(solutions)>(),
                solutions...);
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename Scheme, unsigned int ImpStage,
          unsigned int ExpStage, unsigned int IntOrder, typename TData,
          typename... TDatas>
NEK_FORCE_INLINE static void UpdateSolutionKernel(const unsigned int streamID,
                                                  const size_t nsize,
                                                  TData *inout,
                                                  const TData *solution,
                                                  const TDatas *...solutions)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, nsize, NEKTAR_LAMBDA(const size_t idx) {
            UpdateSolutionKernelImpl<Scheme, ImpStage, ExpStage, IntOrder>(
                idx, inout, solution,
                std::make_integer_sequence<unsigned int,
                                           sizeof...(solutions)>(),
                solutions...);
        });

    Nektar::LoopExecutionSetStreamID(0);
}
#endif

} // namespace Nektar::Operators::detail
