///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecutionHIPCUDA.hpp
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

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__))

#include <float.h>

namespace Nektar
{

static void *internalHIPCUDABuffer                 = nullptr;
static void *internalHIPCUDADeviceBuffer           = nullptr;
static void *internalHIPCUDAHostBuffer             = nullptr;
static unsigned int internalHIPCUDAMaxDataSizeByte = 16;

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_add(TData *const dest, const TData val)
{
    Nektar::atomic_add<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_sub(TData *const dest, const TData val)
{
    Nektar::atomic_sub<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_max(TData *const dest, const TData val)
{
    Nektar::atomic_max<Scope>(dest, val);
}

template <typename ExecSpace, typename Scope, typename TData>
NEK_DEVICE_INLINE
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    atomic_min(TData *const dest, const TData val)
{
    Nektar::atomic_min<Scope>(dest, val);
}

template <typename Functor>
__global__ void parallel_for(const size_t begin, const size_t end,
                             const Functor functor)
{
    size_t i = begin + blockDim.x * blockIdx.x + threadIdx.x;

    while (i < end)
    {
        functor(i);
        i += blockDim.x * gridDim.x;
    }
}

template <bool init, typename TData, typename Functor>
__global__ void reduceSumKernel(const size_t begin, const size_t end,
                                TData *buffer, const Functor functor)
{
    // Implementation based on reduce7 of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if (init && block.thread_rank() == 0)
    {
        buffer[block.group_index().x] = 0.0;
    }

    block.sync();

    for (size_t tid = begin + grid.thread_rank(); tid < end; tid += grid.size())
    {
        v += functor(tid);
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v += warp.shfl_down(v, 32);             // |
    v += warp.shfl_down(v, 16);             // |
    v += warp.shfl_down(v, 8);              // | warp level
    v += warp.shfl_down(v, 4);              // | reduce here
    v += warp.shfl_down(v, 2);              // |
    v += warp.shfl_down(v, 1);              // |
#endif

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
        atomicAdd(&buffer[block.group_index().x], v);
    }
}

template <bool init, typename TData, typename Functor>
__global__ void reduceMaxKernel(const size_t begin, const size_t end,
                                TData *buffer, const Functor functor)
{
    // Implementation based on reduce7 of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData min             = std::numeric_limits<TData>::min();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = min;

    if (init && block.thread_rank() == 0)
    {
        buffer[block.group_index().x] = min;
    }

    block.sync();

    for (size_t tid = begin + grid.thread_rank(); tid < end; tid += grid.size())
    {
        v = std::max(v, functor(tid));
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v = std::max(v, warp.shfl_down(v, 32)); // |
    v = std::max(v, warp.shfl_down(v, 16)); // |
    v = std::max(v, warp.shfl_down(v, 8));  // | warp level
    v = std::max(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::max(v, warp.shfl_down(v, 2));  // |
    v = std::max(v, warp.shfl_down(v, 1));  // |
#endif

    if (warp.thread_rank() == 0)
    {
        atomicMax(&buffer[block.group_index().x], v);
    }
}

template <bool init, typename TData, typename Functor>
__global__ void reduceMinKernel(const size_t begin, const size_t end,
                                TData *buffer, const Functor functor)
{
    // Implementation based on reduce7 of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData max             = std::numeric_limits<TData>::max();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = max;

    if (init && block.thread_rank() == 0)
    {
        buffer[block.group_index().x] = max;
    }

    block.sync();

    for (size_t tid = begin + grid.thread_rank(); tid < end; tid += grid.size())
    {
        v = std::min(v, functor(tid));
    }

    warp.sync();
#if defined(NEKTAR_ENABLE_CUDA)
    v = std::min(v, warp.shfl_down(v, 16)); // |
    v = std::min(v, warp.shfl_down(v, 8));  // | warp level
    v = std::min(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::min(v, warp.shfl_down(v, 2));  // |
    v = std::min(v, warp.shfl_down(v, 1));  // |
#elif defined(NEKTAR_ENABLE_HIP)
    v = std::min(v, warp.shfl_down(v, 32)); // |
    v = std::min(v, warp.shfl_down(v, 16)); // |
    v = std::min(v, warp.shfl_down(v, 8));  // | warp level
    v = std::min(v, warp.shfl_down(v, 4));  // | reduce here
    v = std::min(v, warp.shfl_down(v, 2));  // |
    v = std::min(v, warp.shfl_down(v, 1));  // |
#endif

    if (warp.thread_rank() == 0)
    {
        atomicMin(&buffer[block.group_index().x], v);
    }
}

// Launchers for the kernels
template <typename ExecSpace, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_for(const size_t begin, const size_t end, const Functor &functor)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = ((end - begin) + blockSize - 1u) / blockSize;

    parallel_for<<<gridSize, blockSize>>>(begin, end, functor);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, bool init, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const size_t begin, const size_t end, const Functor &functor,
                typename Reduction::value_type *out)
{
    using TData = typename Reduction::value_type;

    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalHIPCUDABuffer == nullptr)
    {
        const unsigned int internalHIPCUDABufferSize =
            internalHIPCUDAMaxDataSizeByte * gridSize;
        GetDeviceProperties::CheckGlobalMemoryUsage(internalHIPCUDABufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMalloc(&internalHIPCUDABuffer, internalHIPCUDABufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= internalHIPCUDABufferSize;
    }

    TData *buffer = (TData *)internalHIPCUDABuffer;
    if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
    {
        reduceSumKernel<true>
            <<<gridSize, blockSize>>>(begin, end, buffer, functor);
        CHECK_LAST_HIPCUDA_ERROR();
        reduceSumKernel<init>
            <<<1, gridSize>>>(0, gridSize, out, [=] __device__(const size_t i) {
                return buffer[i];
            });
        CHECK_LAST_HIPCUDA_ERROR();
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
        reduceMaxKernel<true>
            <<<gridSize, blockSize>>>(begin, end, buffer, functor);
        CHECK_LAST_HIPCUDA_ERROR();
        reduceMaxKernel<init>
            <<<1, gridSize>>>(0, gridSize, out, [=] __device__(const size_t i) {
                return buffer[i];
            });
        CHECK_LAST_HIPCUDA_ERROR();
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
        reduceMinKernel<true>
            <<<gridSize, blockSize>>>(begin, end, buffer, functor);
        CHECK_LAST_HIPCUDA_ERROR();
        reduceMinKernel<init>
            <<<1, gridSize>>>(0, gridSize, out, [=] __device__(const size_t i) {
                return buffer[i];
            });
        CHECK_LAST_HIPCUDA_ERROR();
    }
}

template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const size_t begin, const size_t end, const Functor &functor,
                typename Reduction::value_type &out)
{
    using TData = typename Reduction::value_type;

    if (internalHIPCUDAHostBuffer == nullptr)
    {
        GetDeviceProperties::CheckGlobalMemoryUsage(
            internalHIPCUDAMaxDataSizeByte);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaMallocHost(&internalHIPCUDAHostBuffer,
                                           internalHIPCUDAMaxDataSizeByte));
        CHECK_HIPCUDA_ERROR(cudaMalloc(&internalHIPCUDADeviceBuffer,
                                       internalHIPCUDAMaxDataSizeByte));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipHostMalloc(&internalHIPCUDAHostBuffer,
                                          internalHIPCUDAMaxDataSizeByte));
        CHECK_HIPCUDA_ERROR(hipMalloc(&internalHIPCUDADeviceBuffer,
                                      internalHIPCUDAMaxDataSizeByte));
#endif
        GetDeviceProperties::TotalGlobalMemory() -=
            internalHIPCUDAMaxDataSizeByte;
    }

    parallel_reduce<ExecSpace, true, Reduction>(
        begin, end, functor, (TData *)internalHIPCUDADeviceBuffer);

#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaMemcpyAsync(internalHIPCUDAHostBuffer,
                                        internalHIPCUDADeviceBuffer,
                                        sizeof(TData), cudaMemcpyDeviceToHost));
    CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(0));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipMemcpyAsync(internalHIPCUDAHostBuffer,
                                       internalHIPCUDADeviceBuffer,
                                       sizeof(TData), hipMemcpyDeviceToHost));
    CHECK_HIPCUDA_ERROR(hipStreamSynchronize(0));
#endif

    out = *(TData *)internalHIPCUDAHostBuffer;
}

} // namespace Nektar

#endif
