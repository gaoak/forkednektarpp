///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecutionCUDA.cuh
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

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

#include <cooperative_groups.h>
#include <cuda/std/limits>
#include <float.h>

namespace Nektar
{

static unsigned int cudaBufferSize = 0u;
static void *cudaBuffer            = nullptr;

namespace cg = cooperative_groups;

__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ float atomicMax_block(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS_block((int *)address, old, __float_as_int(val))) ==
            old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *)address, old,
                             __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

__device__ __forceinline__ double atomicMax_block(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS_block((unsigned long long *)address, old,
                                   __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ float atomicMin_block(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS_block((int *)address, old, __float_as_int(val))) ==
            old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *)address, old,
                             __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

__device__ __forceinline__ double atomicMin_block(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS_block((unsigned long long *)address, old,
                                   __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

template <typename ExecSpace, typename Scope, typename TData>
__device__ __forceinline__
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    atomic_add(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicAdd(dest, val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicAdd_block(dest, val);
    }
}

template <typename ExecSpace, typename Scope, typename TData>
__device__ __forceinline__
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    atomic_sub(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicAdd(dest, -val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicAdd_block(dest, -val);
    }
}

template <typename ExecSpace, typename Scope, typename TData>
__device__ __forceinline__
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    atomic_max(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicMax(dest, val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicMax_block(dest, val);
    }
}

template <typename ExecSpace, typename Scope, typename TData>
__device__ __forceinline__
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                            void>::type
    atomic_min(TData *const dest, const TData val)
{
    if constexpr (std::is_same_v<Scope, NektarSpaces::GlobalScope>)
    {
        atomicMin(dest, val);
    }
    else if constexpr (std::is_same_v<Scope, NektarSpaces::LocalScope>)
    {
        atomicMin_block(dest, val);
    }
}

template <typename Functor>
__global__ void parallel_for(const unsigned int begin, const unsigned int end,
                             const Functor functor)
{
    unsigned int i = begin + blockDim.x * blockIdx.x + threadIdx.x;

    while (i < end)
    {
        functor(i);
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData, typename Functor>
__global__ void reduceSumKernel(const unsigned int begin,
                                const unsigned int end, TData *buffer,
                                const Functor functor)
{
    // Implementation based on reduce7 of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = 0;

    if (block.thread_rank() == 0)
    {
        buffer[block.group_index().x] = 0.0;
    }

    block.sync();

    for (unsigned int tid = begin + grid.thread_rank(); tid < end;
         tid += grid.size())
    {
        functor(tid, v);
    }

    warp.sync();
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
        atomicAdd(&buffer[block.group_index().x], v);
    }
}

template <typename TData, typename Functor>
__global__ void reduceMaxKernel(const unsigned int begin,
                                const unsigned int end, TData *buffer,
                                const Functor functor)
{
    // Implementation based on reduce7 of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData min             = ::cuda::std::numeric_limits<TData>::min();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = min;

    if (block.thread_rank() == 0)
    {
        buffer[block.group_index().x] = min;
    }

    block.sync();

    for (unsigned int tid = begin + grid.thread_rank(); tid < end;
         tid += grid.size())
    {
        functor(tid, v);
    }

    warp.sync();
    v = max(v, warp.shfl_down(v, 16)); // |
    v = max(v, warp.shfl_down(v, 8));  // | warp level
    v = max(v, warp.shfl_down(v, 4));  // | reduce here
    v = max(v, warp.shfl_down(v, 2));  // |
    v = max(v, warp.shfl_down(v, 1));  // |

    if (warp.thread_rank() == 0)
    {
        atomicMax(&buffer[block.group_index().x], v);
    }
}

template <typename TData, typename Functor>
__global__ void reduceMinKernel(const unsigned int begin,
                                const unsigned int end, TData *buffer,
                                const Functor functor)
{
    // Implementation based on reduce7 of "Ansorge, R. (2022). Programming in
    // parallel with CUDA: a practical guide. Cambridge University Press."

    constexpr TData max             = ::cuda::std::numeric_limits<TData>::max();
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v    = max;

    if (block.thread_rank() == 0)
    {
        buffer[block.group_index().x] = max;
    }

    block.sync();

    for (unsigned int tid = begin + grid.thread_rank(); tid < end;
         tid += grid.size())
    {
        functor(tid, v);
    }

    warp.sync();
    v = min(v, warp.shfl_down(v, 16)); // |
    v = min(v, warp.shfl_down(v, 8));  // | warp level
    v = min(v, warp.shfl_down(v, 4));  // | reduce here
    v = min(v, warp.shfl_down(v, 2));  // |
    v = min(v, warp.shfl_down(v, 1));  // |

    if (warp.thread_rank() == 0)
    {
        atomicMin(&buffer[block.group_index().x], v);
    }
}

// Launchers for the kernels
template <typename ExecSpace, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                               void>::type
parallel_for(const unsigned int begin, const unsigned int end,
             const Functor &functor)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = ((end - begin) + blockSize - 1u) / blockSize;

    parallel_for<<<gridSize, blockSize>>>(begin, end, functor);
}

template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                               void>::type
parallel_reduce(const unsigned int begin, const unsigned int end,
                const Functor &functor, typename Reduction::value_type *out)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = ((end - begin) + blockSize - 1u) / blockSize;

    using TData = typename Reduction::value_type;

    if (cudaBuffer == nullptr)
    {
        cudaBufferSize = sizeof(TData) * gridSize;
        cudaMalloc(&cudaBuffer, cudaBufferSize);
    }

    TData *buffer = (TData *)cudaBuffer;

    TData *d_out;
    cudaMalloc((void **)&d_out, sizeof(TData));
    if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
    {
        reduceSumKernel<TData>
            <<<gridSize, blockSize>>>(begin, end, buffer, functor);
        reduceSumKernel<TData><<<1, gridSize>>>(
            0, gridSize, out, [=] __device__(const unsigned int i, TData &ans) {
                ans += buffer[i];
            });
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
        reduceMaxKernel<TData>
            <<<gridSize, blockSize>>>(begin, end, buffer, functor);
        reduceMaxKernel<TData><<<1, gridSize>>>(
            0, gridSize, out, [=] __device__(const unsigned int i, TData &ans) {
                ans = max(ans, buffer[i]);
            });
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
        reduceMinKernel<TData>
            <<<gridSize, blockSize>>>(begin, end, buffer, functor);
        reduceMinKernel<TData><<<1, gridSize>>>(
            0, gridSize, out, [=] __device__(const unsigned int i, TData &ans) {
                ans = min(ans, buffer[i]);
            });
    }
    cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

} // namespace Nektar

#endif
