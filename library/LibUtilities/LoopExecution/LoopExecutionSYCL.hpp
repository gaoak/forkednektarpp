///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecutionSYCL.hpp
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

#if defined(NEKTAR_ENABLE_SYCL)

namespace Nektar
{

// Parallel for launchers.
template <
    typename ExecSpace, typename Functor,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
inline void parallel_for(const size_t begin, const size_t end,
                         const Functor &functor)
{
    const unsigned int streamID = internalLoopExecutionStreamID;
    sycl::queue &Q              = SYCLQueue::GetInstance(streamID);
    sycl::event e               = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        cgh.parallel_for(sycl::range<1>(end - begin),
                                       [=](sycl::id<1> indx) { functor(begin + indx); });
    });
    SYCLQueue::SetEvent(streamID, e);
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
        atomic_add<NektarSpaces::LocalScope>(dest, val);
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
        atomic_max<NektarSpaces::LocalScope>(dest, val);
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
        atomic_min<NektarSpaces::LocalScope>(dest, val);
    }
}

// Partial results of nred values per work-group: functor(i, vals) writes those
// of index i to vals[0, nred), which Reduction combines. Result s of
// work-group g is in buffer[s * gridSize + g], which is set to the identity
// first if init.
template <bool init, typename Reduction, unsigned int nred, typename Functor>
void reduceNKernel(const unsigned int gridSize, const unsigned int blockSize,
                   const size_t begin, const size_t end,
                   typename Reduction::value_type *buffer,
                   const Functor &functor)
{
    using TData = typename Reduction::value_type;

    constexpr TData identity = reduce_identity<Reduction>();

    const unsigned int streamID = internalLoopExecutionStreamID;
    sycl::queue &Q              = SYCLQueue::GetInstance(streamID);
    sycl::event e               = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        sycl::local_accessor<TData, 1> scratchpad(
            sycl::range<1>(nred * blockSize), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(gridSize * blockSize, blockSize),
            [=](sycl::nd_item<1> indx) SYCL_SUBGROUP_SIZE(
                NektarSpaces::Device::warpSize) {
                auto scratch =
                    scratchpad
                        .template get_multi_ptr<sycl::access::decorated::yes>()
                        .get();
                const size_t lid = indx.get_local_id(0);
                size_t gid       = begin + indx.get_global_id(0);

                if (init && lid == 0)
                {
                    for (unsigned int s = 0; s < nred; ++s)
                    {
                        buffer[s * gridSize + indx.get_group(0)] = identity;
                    }
                }

#if defined(__ADAPTIVECPP__)
                indx.barrier(sycl::access::fence_space::local_space);
#else
                sycl::group_barrier(indx.get_group(),
                                    sycl::memory_scope::work_group);
#endif

                TData tmp[nred];
                for (unsigned int s = 0; s < nred; ++s)
                {
                    tmp[s] = identity;
                }
                while (gid < end)
                {
                    TData vals[nred];
                    functor(gid, vals);
                    for (unsigned int s = 0; s < nred; ++s)
                    {
                        tmp[s] = reduce_join<Reduction>(tmp[s], vals[s]);
                    }
                    gid += indx.get_global_range(0);
                }
                for (unsigned int s = 0; s < nred; ++s)
                {
                    scratch[s * blockSize + lid] = tmp[s];
                }

#if defined(__ADAPTIVECPP__)
                indx.barrier(sycl::access::fence_space::local_space);
#else
                sycl::group_barrier(indx.get_group(),
                                    sycl::memory_scope::work_group);
#endif

                unsigned int n = NektarSpaces::Device::maximumBlockSize / 2;
                while (n > 0)
                {
                    if (blockSize > n && lid < n && lid + n < blockSize)
                    {
                        for (unsigned int s = 0; s < nred; ++s)
                        {
                            scratch[s * blockSize + lid] =
                                reduce_join<Reduction>(
                                    scratch[s * blockSize + lid],
                                    scratch[s * blockSize + lid + n]);
                        }
                    }
#if defined(__ADAPTIVECPP__)
                    indx.barrier(sycl::access::fence_space::local_space);
#else
                    sycl::group_barrier(indx.get_group(),
                                        sycl::memory_scope::work_group);
#endif
                    n /= 2;
                }

                if (lid == 0)
                {
                    for (unsigned int s = 0; s < nred; ++s)
                    {
                        reduce_atomic<Reduction>(buffer + s * gridSize +
                                                     indx.get_group(0),
                                                 scratch[s * blockSize]);
                    }
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
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

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceNKernel<true, Reduction, 1>(
        gridSize, blockSize, begin, end, buffer,
        [=](const size_t i, TData *vals) { vals[0] = functor(i); });
    reduceNKernel<init, Reduction, 1>(
        1, gridSize, 0, gridSize, out,
        [=](const size_t i, TData *vals) { vals[0] = buffer[i]; });
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

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalReductionBufferSize, streamID);
    }

    TData *buffer = (TData *)internalMemoryBufferMap[streamID];
    reduceNKernel<true, Reduction, nred>(gridSize, blockSize, begin, end,
                                         buffer, functor);
    reduceNKernel<init, Reduction, nred>(
        1, blockSize, 0, gridSize, out, [=](const size_t i, TData *vals) {
            for (unsigned int s = 0; s < nred; ++s)
            {
                vals[s] = buffer[s * gridSize + i];
            }
        });
}

} // namespace Nektar

#endif
