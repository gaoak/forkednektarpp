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

static void *hipcudaBuffer = nullptr;

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

    constexpr TData min             = std::numeric_limits<TData>::min();
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

    constexpr TData max             = std::numeric_limits<TData>::max();
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
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_for(const unsigned int begin, const unsigned int end,
             const Functor &functor)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = ((end - begin) + blockSize - 1u) / blockSize;

    parallel_for<<<gridSize, blockSize>>>(begin, end, functor);
    CHECK_LAST_HIPCUDA_ERROR();
}

template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const unsigned int begin, const unsigned int end,
                const Functor &functor, typename Reduction::value_type *out)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    using TData = typename Reduction::value_type;

    if (hipcudaBuffer == nullptr)
    {
        const unsigned int hipcudaBufferSize = sizeof(TData) * (gridSize + 1);
        GetDeviceProperties::CheckGlobalMemoryUsage(hipcudaBufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaMalloc(&hipcudaBuffer, hipcudaBufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipMalloc(&hipcudaBuffer, hipcudaBufferSize));
#endif
        GetDeviceProperties::TotalGlobalMemory() -= hipcudaBufferSize;
    }

    TData *buffer = (TData *)hipcudaBuffer;
    TData *d_out  = (TData *)hipcudaBuffer + gridSize;
    if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
    {
        reduceSumKernel<<<gridSize, blockSize>>>(begin, end, buffer, functor);
        CHECK_LAST_HIPCUDA_ERROR();
        reduceSumKernel<<<1, gridSize>>>(
            0, gridSize, out, [=] __device__(const unsigned int i, TData &ans) {
                ans += buffer[i];
            });
        CHECK_LAST_HIPCUDA_ERROR();
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
        reduceMaxKernel<<<gridSize, blockSize>>>(begin, end, buffer, functor);
        CHECK_LAST_HIPCUDA_ERROR();
        reduceMaxKernel<<<1, gridSize>>>(
            0, gridSize, out, [=] __device__(const unsigned int i, TData &ans) {
                ans = max(ans, buffer[i]);
            });
        CHECK_LAST_HIPCUDA_ERROR();
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
        reduceMinKernel<<<gridSize, blockSize>>>(begin, end, buffer, functor);
        CHECK_LAST_HIPCUDA_ERROR();
        reduceMinKernel<<<1, gridSize>>>(
            0, gridSize, out, [=] __device__(const unsigned int i, TData &ans) {
                ans = min(ans, buffer[i]);
            });
        CHECK_LAST_HIPCUDA_ERROR();
    }
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(
        cudaMemcpy(out, d_out, sizeof(TData), cudaMemcpyDeviceToHost));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(
        hipMemcpy(out, d_out, sizeof(TData), hipMemcpyDeviceToHost));
#endif
}

} // namespace Nektar

#endif
