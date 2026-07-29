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
template <typename ExecSpace, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_for(const size_t begin, const size_t end, const Functor &functor)
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

// Reduction kernels.
template <bool init, typename TData, typename Functor>
void reduceSumKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t begin, const size_t end, TData *buffer,
                     const Functor &functor)
{
    const unsigned int streamID = internalLoopExecutionStreamID;
    sycl::queue &Q              = SYCLQueue::GetInstance(streamID);
    sycl::event e               = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        sycl::local_accessor<TData, 1> scratchpad(sycl::range<1>(blockSize),
                                                                cgh);
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
                    buffer[indx.get_group(0)] = 0.0;
                }

#if defined(__ADAPTIVECPP__)
                indx.barrier(sycl::access::fence_space::local_space);
#else
                sycl::group_barrier(indx.get_group(),
                                    sycl::memory_scope::work_group);
#endif

                TData tmp = 0.0;
                while (gid < end)
                {
                    tmp += functor(gid);
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

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
                        scratch[lid] += scratch[lid + n];
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
                    atomic_add<NektarSpaces::LocalScope>(
                        buffer + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData, typename Functor>
void reduceMaxKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t begin, const size_t end, TData *buffer,
                     const Functor &functor)
{
    constexpr TData min = std::numeric_limits<TData>::lowest();

    const unsigned int streamID = internalLoopExecutionStreamID;
    sycl::queue &Q              = SYCLQueue::GetInstance(streamID);
    sycl::event e               = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        sycl::local_accessor<TData, 1> scratchpad(sycl::range<1>(blockSize),
                                                                cgh);
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
                    buffer[indx.get_group(0)] = min;
                }

#if defined(__ADAPTIVECPP__)
                indx.barrier(sycl::access::fence_space::local_space);
#else
                sycl::group_barrier(indx.get_group(),
                                    sycl::memory_scope::work_group);
#endif

                TData tmp = min;
                while (gid < end)
                {
                    tmp = sycl::fmax(tmp, functor(gid));
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

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
                        scratch[lid] =
                            sycl::fmax(scratch[lid], scratch[lid + n]);
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
                    atomic_max<NektarSpaces::LocalScope>(
                        buffer + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

template <bool init, typename TData, typename Functor>
void reduceMinKernel(const unsigned int gridSize, const unsigned int blockSize,
                     const size_t begin, const size_t end, TData *buffer,
                     const Functor &functor)
{
    constexpr TData max = std::numeric_limits<TData>::max();

    const unsigned int streamID = internalLoopExecutionStreamID;
    sycl::queue &Q              = SYCLQueue::GetInstance(streamID);
    sycl::event e               = Q.submit([=](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        sycl::local_accessor<TData, 1> scratchpad(sycl::range<1>(blockSize),
                                                                cgh);
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
                    buffer[indx.get_group(0)] = max;
                }

#if defined(__ADAPTIVECPP__)
                indx.barrier(sycl::access::fence_space::local_space);
#else
                sycl::group_barrier(indx.get_group(),
                                    sycl::memory_scope::work_group);
#endif

                TData tmp = max;
                while (gid < end)
                {
                    tmp = sycl::fmin(tmp, functor(gid));
                    gid += indx.get_global_range(0);
                }
                scratch[lid] = tmp;

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
                        scratch[lid] =
                            sycl::fmin(scratch[lid], scratch[lid + n]);
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
                    atomic_min<NektarSpaces::LocalScope>(
                        buffer + indx.get_group(0), scratch[0]);
                }
            });
    });
    SYCLQueue::SetEvent(streamID, e);
}

// Parallel reduction launchers without device-to-host copy.
template <typename ExecSpace, bool init, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const size_t begin, const size_t end, const Functor &functor,
                typename Reduction::value_type *out)
{
    using TData = typename Reduction::value_type;

    const unsigned int streamID  = internalLoopExecutionStreamID;
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = NektarSpaces::Device::maximumBlockSize;

    if (internalMemoryBufferMap.find(streamID) == internalMemoryBufferMap.end())
    {
        const unsigned int internalMemoryBufferSize =
            internalMaxDataSizeByte * gridSize;
        deviceMalloc(&internalMemoryBufferMap[streamID],
                     internalMemoryBufferSize, streamID);
    }

    if constexpr (std::is_same_v<Reduction, Nektar::ReduceSum<TData>>)
    {
        TData *buffer = (TData *)internalMemoryBufferMap[streamID];
        reduceSumKernel<true>(gridSize, blockSize, begin, end, buffer, functor);
        reduceSumKernel<init>(1, gridSize, 0, gridSize, out,
                              [=](const size_t i) { return buffer[i]; });
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMax<TData>>)
    {
        TData *buffer = (TData *)internalMemoryBufferMap[streamID];
        reduceMaxKernel<true>(gridSize, blockSize, begin, end, buffer, functor);
        reduceMaxKernel<init>(1, gridSize, 0, gridSize, out,
                              [=](const size_t i) { return buffer[i]; });
    }
    else if constexpr (std::is_same_v<Reduction, Nektar::ReduceMin<TData>>)
    {
        TData *buffer = (TData *)internalMemoryBufferMap[streamID];
        reduceMinKernel<true>(gridSize, blockSize, begin, end, buffer, functor);
        reduceMinKernel<init>(1, gridSize, 0, gridSize, out,
                              [=](const size_t i) { return buffer[i]; });
    }
}

// Parallel reduction launchers with device-to-host copy.
template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
parallel_reduce(const size_t begin, const size_t end, const Functor &functor,
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

} // namespace Nektar

#endif
