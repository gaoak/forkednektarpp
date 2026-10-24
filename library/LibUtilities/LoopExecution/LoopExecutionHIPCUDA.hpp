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

#if (defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)) ||           \
    (defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY))

#include <float.h>

namespace Nektar
{

// Parallel for kernel.
template <typename Functor,
          unsigned int blockSize = NektarSpaces::Device::defaultBlockSize>
__global__ __launch_bounds__(blockSize) void parallel_for(const size_t begin,
                                                          const size_t end,
                                                          const Functor functor)
{
    size_t i = begin + blockDim.x * blockIdx.x + threadIdx.x;

    while (i < end)
    {
        functor(i);
        i += blockDim.x * gridDim.x;
    }
}

// Parallel for launchers.
template <
    typename ExecSpace, typename Functor,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void parallel_for(const size_t begin, const size_t end,
                         const Functor &functor)
{
    // CUDA/HIP kernel launches require every grid dimension to be >= 1;
    // launching with gridDim.x == 0 (which naturally happens here whenever
    // the loop range is empty, e.g. end == begin) is rejected by the
    // runtime with an "invalid argument" error rather than silently doing
    // nothing. There is trivially no work to do in that case, so skip the
    // launch entirely instead.
    if (end <= begin)
    {
        return;
    }

    const unsigned int streamID  = internalLoopExecutionStreamID;
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = ((end - begin) + blockSize - 1u) / blockSize;

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    parallel_for<<<gridSize, blockSize, 0, stream>>>(begin, end, functor);
    CHECK_LAST_HIPCUDA_ERROR();
}

// Atomic combination of val into *dest by a reduction.
template <typename Reduction>
NEK_DEVICE_INLINE static void reduce_atomic(
    typename Reduction::value_type *const dest,
    const typename Reduction::value_type val)
{
    using TData = typename Reduction::value_type;
    if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
    {
        atomic_add<NektarSpaces::GlobalScope>(dest, val);
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
        atomic_max<NektarSpaces::GlobalScope>(dest, val);
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
        atomic_min<NektarSpaces::GlobalScope>(dest, val);
    }
}

// Partial results of nred values per thread block: functor(i, vals) writes
// those of index i to vals[0, nred), which Reduction combines. Result s of
// block b is in buffer[s * gridDim.x + b], which is set to the identity first
// if init.
template <bool init, typename Reduction, unsigned int nred, typename Functor,
          unsigned int blockSize = NektarSpaces::Device::reductionBlockSize()>
__global__ __launch_bounds__(blockSize) void reduceNKernel(
    const size_t begin, const size_t end,
    typename Reduction::value_type *buffer, const Functor functor)
{
    using TData = typename Reduction::value_type;

    constexpr TData identity        = reduce_identity<Reduction>();
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<warpsize>(block);
    TData v[nred];

    for (unsigned int s = 0; s < nred; ++s)
    {
        v[s] = identity;
    }

    if constexpr (init)
    {
        if (block.thread_rank() == 0)
        {
            for (unsigned int s = 0; s < nred; ++s)
            {
                buffer[s * gridDim.x + block.group_index().x] = identity;
            }
        }
    }

    block.sync();

    for (size_t tid = begin + grid.thread_rank(); tid < end; tid += grid.size())
    {
        TData vals[nred];
        functor(tid, vals);
        for (unsigned int s = 0; s < nred; ++s)
        {
            v[s] = reduce_join<Reduction>(v[s], vals[s]);
        }
    }

    warp.sync();
#pragma unroll
    for (unsigned int s = 0; s < nred; ++s)
    {
#pragma unroll
        for (unsigned int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            v[s] = reduce_join<Reduction>(v[s], warp.shfl_down(v[s], offset));
        }
    }

    // use atomics to combine over warps
    if (warp.thread_rank() == 0)
    {
        for (unsigned int s = 0; s < nred; ++s)
        {
            reduce_atomic<Reduction>(
                &buffer[s * gridDim.x + block.group_index().x], v[s]);
        }
    }
}

// Parallel reduction launchers without device-to-host copy.
template <
    typename ExecSpace, bool init, typename Reduction, typename Functor,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void parallel_reduce(const size_t begin, const size_t end,
                            const Functor &functor,
                            typename Reduction::value_type *out)
{
    using TData = typename Reduction::value_type;

    const unsigned int streamID  = internalLoopExecutionStreamID;
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(end - begin);

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceNKernel<true, Reduction, 1><<<gridSize, blockSize, 0, stream>>>(
        begin, end, buffer,
        [=] __device__(const size_t i, TData *vals) { vals[0] = functor(i); });
    CHECK_LAST_HIPCUDA_ERROR();
    reduceNKernel<init, Reduction, 1><<<1, blockSize, 0, stream>>>(
        0, gridSize, out,
        [=] __device__(const size_t i, TData *vals) { vals[0] = buffer[i]; });
    CHECK_LAST_HIPCUDA_ERROR();
}

// Parallel reduction launchers with device-to-host copy.
template <
    typename ExecSpace, typename Reduction, typename Functor,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void parallel_reduce(const size_t begin, const size_t end,
                            const Functor &functor,
                            typename Reduction::value_type &out)
{
    using TData = typename Reduction::value_type;

    const unsigned int streamID = internalLoopExecutionStreamID;
    if (internalHostBuffer == nullptr)
    {
        hostMallocPinned(&internalHostBuffer, internalMaxDataSizeByte);
        deviceMalloc(&internalDeviceBuffer, internalMaxDataSizeByte, streamID);
    }

    parallel_reduce<ExecSpace, true, Reduction>(begin, end, functor,
                                                (TData *)internalDeviceBuffer);

    deviceMemcpy<DeviceToHost>(internalHostBuffer, internalDeviceBuffer,
                               sizeof(TData), streamID);

    out = *(TData *)internalHostBuffer;
}

// Parallel reduction of nred values without device-to-host copy:
// functor(i, vals) writes those of index i to vals[0, nred), which Reduction
// combines into out[0, nred), together with the values already there unless
// init. At most 4 values, the partial results the reduction scratch holds.
template <
    typename ExecSpace, bool init, typename Reduction, unsigned int nred,
    typename Functor,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void parallel_reduce(const size_t begin, const size_t end,
                            const Functor &functor,
                            typename Reduction::value_type *out)
{
    using TData = typename Reduction::value_type;

    static_assert(nred <= 4, "parallel_reduce: at most 4 values.");

    const unsigned int streamID  = internalLoopExecutionStreamID;
    const unsigned int blockSize = NektarSpaces::Device::reductionBlockSize();
    const unsigned int gridSize =
        NektarSpaces::Device::reductionGridSize(end - begin);

    ASSERTL0(nred * gridSize * sizeof(TData) <= internalReductionBufferSize,
             "Too many values for the reduction buffer.");

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
#endif

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceNKernel<true, Reduction, nred>
        <<<gridSize, blockSize, 0, stream>>>(begin, end, buffer, functor);
    CHECK_LAST_HIPCUDA_ERROR();
    reduceNKernel<init, Reduction, nred><<<1, blockSize, 0, stream>>>(
        0, gridSize, out, [=] __device__(const size_t i, TData *vals) {
            for (unsigned int s = 0; s < nred; ++s)
            {
                vals[s] = buffer[s * gridSize + i];
            }
        });
    CHECK_LAST_HIPCUDA_ERROR();
}

} // namespace Nektar

#endif
