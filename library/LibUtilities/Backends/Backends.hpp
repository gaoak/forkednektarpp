///////////////////////////////////////////////////////////////////////////////
//
// File: Backends.hpp
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

#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>

#include <type_traits>

#include <float.h>
#include <iostream>
#include <limits.h>
#include <string>
#include <unordered_map>
#include <utility>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

#if defined(NEKTAR_ENABLE_CUDA)
#include "LibUtilities/Backends/CUDA_Host_API.hpp"
#elif defined(NEKTAR_ENABLE_HIP)
#include "LibUtilities/Backends/HIP_Host_API.hpp"
#elif defined(NEKTAR_ENABLE_SYCL)
#include "LibUtilities/Backends/SYCL_Host_API.hpp"
#endif

namespace Nektar
{

#if defined(NEKTAR_ENABLE_DEVICE)
extern std::unordered_map<unsigned int, void *> internalMemoryBufferMap;
extern void *internalDeviceBuffer;
extern void *internalHostBuffer;
extern unsigned int internalMaxDataSizeByte;
// Bytes of each stream's reduction scratch in internalMemoryBufferMap:
// internalMaxDataSizeByte, two partial sums of double, per thread block of the
// largest reduction grid.
extern unsigned int internalReductionBufferSize;
#endif

namespace NektarSpaces
{

// Memory space.
// Used to refer to any data in host memory.
struct HostSpace
{
};
#if defined(NEKTAR_ENABLE_DEVICE)
// Used to refer to any data in device memory.
struct DeviceSpace
{
};
#else
// No specific GPU so the device is the host.
using DeviceSpace = HostSpace;
#endif

// Execution space.
struct Serial
{
    static inline const std::string name = "Serial";
    using memory_space                   = NektarSpaces::HostSpace;
};

struct AVX
{
    static inline const std::string name = "AVX";
    using memory_space                   = NektarSpaces::HostSpace;
};

struct Device
{
    static inline const std::string name = "Device";
    using memory_space                   = NektarSpaces::DeviceSpace;
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
    static constexpr unsigned int warpSize         = 32u;
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 1024u;
    static constexpr unsigned int warpSize         = 64u;
#elif defined(SYCL_ENABLE_INTEL)
    static constexpr unsigned int defaultBlockSize = 128u;
    static constexpr unsigned int maximumBlockSize = 1024u;
    static constexpr unsigned int warpSize         = 32u;
#elif defined(SYCL_ENABLE_CPU)
    static constexpr unsigned int defaultBlockSize = 256u;
    static constexpr unsigned int maximumBlockSize = 256u;
#if defined(__ADAPTIVECPP__) || defined(__DPCPP_COMPILER_NATIVE_CPU)
    static constexpr unsigned int warpSize = 1u;
#elif defined(__DPCPP_COMPILER)
    static constexpr unsigned int warpSize = 8u;
#elif defined(NEKTAR_USE_SIMSYCL)
    // SimSYCL simulates a sub-group of 32 by default.
    static constexpr unsigned int warpSize = 32u;
#else
    static constexpr unsigned int warpSize = 1u;
#endif
#else
    static constexpr unsigned int defaultBlockSize = 1u;
    static constexpr unsigned int maximumBlockSize = 1u;
    static constexpr unsigned int warpSize         = 1u;
#endif
};

#if defined(__ADAPTIVECPP__)
#define SYCL_SUBGROUP_SIZE(x)
#elif defined(__DPCPP_COMPILER)
#define SYCL_SUBGROUP_SIZE(x) [[sycl::reqd_sub_group_size(x)]]
#else
#define SYCL_SUBGROUP_SIZE(x)
#endif

// Host memory alignment
#if defined(NEKTAR_ENABLE_SIMD)
static constexpr size_t host_memory_alignment =
    tinysimd::simd<double>::alignment;
#else
static constexpr size_t host_memory_alignment =
    __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#endif

// Vector width
template <typename ExecSpace, typename TData> struct vector_width
{
};

template <typename TData> struct vector_width<Serial, TData>
{
    static constexpr unsigned int value = 1u;
};

template <typename TData> struct vector_width<AVX, TData>
{
    static constexpr unsigned int value = tinysimd::simd<TData>::width;
};

template <typename TData> struct vector_width<Device, TData>
{
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA) ||                \
    defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP) ||                  \
    defined(SYCL_ENABLE_INTEL)
    static_assert(
        Device::warpSize % tinysimd::simd<TData>::width == 0,
        "AVX/Device back-ends interoperability requires device vector width "
        "(warpsize) to be integer multiple of SIMD vector width");
#endif
    static constexpr unsigned int value = Device::warpSize;
};

template <typename TData> struct max_vector_width
{
    // Use maximum vector width for back-ends interoperability.
    static constexpr unsigned int value = std::max(
        vector_width<AVX, TData>::value, vector_width<Device, TData>::value);
};

template <typename TData>
static unsigned int GetVectorWidth(const std::string &execName)
{
    if (execName == "Serial")
    {
        return NektarSpaces::vector_width<NektarSpaces::Serial, TData>::value;
    }
    else if (execName == "AVX")
    {
        return NektarSpaces::vector_width<NektarSpaces::AVX, TData>::value;
    }
    else if (execName == "Device")
    {
        return NektarSpaces::vector_width<NektarSpaces::Device, TData>::value;
    }
    else
    {
        return 0;
    }
}

} // namespace NektarSpaces

#if defined(__CUDACC__) || defined(__NEK_HIPCC__) || defined(NEKTAR_ENABLE_SYCL)
#define DEVICE_COMPILE_ONLY
#endif

// NEK_RESTRICT
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#define NEK_RESTRICT __restrict__
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#define NEK_RESTRICT __restrict__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEK_RESTRICT __restrict__
#else
#define NEK_RESTRICT __restrict__
#endif

// NEK_HOSTDEVICE_INLINE
// Used to define a generic function that can be used on both the
// host or the dvice . All generic functions must be prefixed by the
// NEK_HOSTDEVICE_INLINE decorator.
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#define NEK_HOSTDEVICE_INLINE __host__ __device__ __forceinline__
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#define NEK_HOSTDEVICE_INLINE __host__ __device__ __forceinline__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEK_HOSTDEVICE_INLINE NEK_FORCE_INLINE
#else
#define NEK_HOSTDEVICE_INLINE NEK_FORCE_INLINE
#endif

// NEK_DEVICE_INLINE
// Used to define a device function (e.g. a function launched
// from a kernel function and executing on the device). All
// device functions must be prefixed by the NEK_DEVICE_INLINE
// decorator.
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_INLINE __device__ __forceinline__
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_INLINE __device__ __forceinline__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEK_DEVICE_INLINE NEK_FORCE_INLINE
#else
#define NEK_DEVICE_INLINE NEK_FORCE_INLINE
#endif

// NEK_DEVICE_KERNEL
// Used to define a kernel function (e.g. a function launched
// from the host and executing on the device). All kernel
// functions must be prefixed by the NEK_DEVICE_KERNEL decorator.
#if defined(NEKTAR_ENABLE_CUDA) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_KERNEL __global__
#elif defined(NEKTAR_ENABLE_HIP) && defined(DEVICE_COMPILE_ONLY)
#define NEK_DEVICE_KERNEL __global__
#elif defined(NEKTAR_ENABLE_SYCL)
#define NEK_DEVICE_KERNEL NEK_FORCE_INLINE
#else
#define NEK_DEVICE_KERNEL NEK_FORCE_INLINE
#endif

template <typename TData> std::string DataTypeToString(void)
{
    if constexpr (std::is_same_v<TData, float>)
    {
        return "float";
    }
    else if constexpr (std::is_same_v<TData, double>)
    {
        return "double";
    }
}

// const_if metafunction return "const T" type if B = true and "T" type
// otherwise.
template <bool B, typename TData = void> struct const_if
{
    typedef TData type;
};

template <class TData> struct const_if<true, TData>
{
    typedef const TData type;
};

template <bool B, typename TData> struct data_type_if
{
    typedef TData type;
};

template <typename TData> struct data_type_if<true, TData>
{
    typedef tinysimd::simd<TData> type;
};

template <bool B, typename TData> struct simd_type_if
{
    typedef tinysimd::scalarT<TData> type;
};

template <typename TData> struct simd_type_if<true, TData>
{
    typedef tinysimd::simd<TData> type;
};

[[maybe_unused]] static inline void nekSetDevice(
    [[maybe_unused]] unsigned int device_rank)
{
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
#elif defined(NEKTAR_ENABLE_SYCL)
    internalSYCLDeviceId = device_rank;
#else
    // Do nothing
#endif
}

[[maybe_unused]] static inline unsigned int nekGetDevice()
{
#if defined(NEKTAR_ENABLE_CUDA)
    int device_rank = 0;
    CHECK_HIPCUDA_ERROR(cudaGetDevice(&device_rank));
    return device_rank;
#elif defined(NEKTAR_ENABLE_HIP)
    int device_rank = 0;
    CHECK_HIPCUDA_ERROR(hipGetDevice(&device_rank));
    return device_rank;
#elif defined(NEKTAR_ENABLE_SYCL)
    return internalSYCLDeviceId;
#else
    return 0;
#endif
}

[[maybe_unused]] static inline unsigned int nekGetNumDevice()
{
    int num_device = 0;
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaGetDeviceCount(&num_device));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipGetDeviceCount(&num_device));
#elif defined(SYCL_ENABLE_CUDA) || defined(SYCL_ENABLE_HIP) ||                 \
    defined(SYCL_ENABLE_INTEL)
    num_device = sycl::device::get_devices(sycl::info::device_type::gpu).size();
#elif defined(NEKTAR_USE_SIMSYCL)
    num_device = 1;
#endif
    return num_device;
}

[[maybe_unused]] static inline void nekDeviceSynchronize(void)
{
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaDeviceSynchronize());
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipDeviceSynchronize());
#elif defined(NEKTAR_ENABLE_SYCL)
    auto &Queues = SYCLQueue::GetAllInstances();
    for (auto &item : Queues)
    {
        item.second->wait();
    }
#endif
}

[[maybe_unused]] static inline void nekStreamSynchronize(
    [[maybe_unused]] const unsigned int streamID)
{
#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
    CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(stream));
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
    CHECK_HIPCUDA_ERROR(hipStreamSynchronize(stream));
#elif defined(NEKTAR_ENABLE_SYCL)
    SYCLQueue::GetInstance(streamID).wait();
#endif
}

template <
    typename ExecSpace,
    std::enable_if_t<!std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
static void SetStreamDependencies([[maybe_unused]] unsigned int streamID,
                                  [[maybe_unused]] unsigned int eventID)
{
}

template <
    typename ExecSpace,
    std::enable_if_t<!std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
static void SetStreamDependencies(
    [[maybe_unused]] unsigned int streamID,
    [[maybe_unused]] std::vector<unsigned int> &eventIDs)
{
}

template <
    typename ExecSpace,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
static void SetStreamDependencies([[maybe_unused]] unsigned int streamID,
                                  [[maybe_unused]] unsigned int eventID)
{
#if defined(NEKTAR_ENABLE_CUDA)
    if (streamID != eventID)
    {
        auto stream = CUDAStream::GetInstance(streamID);
        auto e      = CUDAStream::GetEvent(eventID);
        if (e != nullptr)
        {
            CHECK_HIPCUDA_ERROR(cudaStreamWaitEvent(stream, e));
        }
    }
#elif defined(NEKTAR_ENABLE_HIP)
    if (streamID != eventID)
    {
        auto stream = HIPStream::GetInstance(streamID);
        auto e      = HIPStream::GetEvent(eventID);
        if (e != nullptr)
        {
            CHECK_HIPCUDA_ERROR(hipStreamWaitEvent(stream, e));
        }
    }
#elif defined(NEKTAR_ENABLE_SYCL)
    if (streamID != eventID)
    {
        [[maybe_unused]] sycl::event event = SYCLQueue::GetEvent(eventID);
#if defined(__ADAPTIVECPP__)
        sycl::event e =
            SYCLQueue::GetInstance(streamID).submit([&](sycl::handler &cgh) {
                cgh.depends_on(event);
                cgh.AdaptiveCpp_enqueue_custom_operation(
                    [=]([[maybe_unused]] sycl::interop_handle ih) {});
            });
        SYCLQueue::SetEvent(streamID, e);
#elif defined(__DPCPP_COMPILER)
        sycl::event e = SYCLQueue::GetInstance(streamID).submit(
            [&](sycl::handler &cgh) { cgh.ext_oneapi_barrier({event}); });
        SYCLQueue::SetEvent(streamID, e);
#endif
    }
#endif
}

template <
    typename ExecSpace,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
static void SetStreamDependencies(
    [[maybe_unused]] unsigned int streamID,
    [[maybe_unused]] std::vector<unsigned int> &eventIDs)
{
#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
    for (auto eventID : eventIDs)
    {
        if (streamID != eventID)
        {
            auto e = CUDAStream::GetEvent(eventID);
            if (e != nullptr)
            {
                CHECK_HIPCUDA_ERROR(cudaStreamWaitEvent(stream, e));
            }
        }
    }
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
    for (auto eventID : eventIDs)
    {
        if (streamID != eventID)
        {
            auto e = HIPStream::GetEvent(eventID);
            if (e != nullptr)
            {
                CHECK_HIPCUDA_ERROR(hipStreamWaitEvent(stream, e));
            }
        }
    }
#elif defined(NEKTAR_ENABLE_SYCL)
    std::vector<sycl::event> events;
    for (auto eventID : eventIDs)
    {
        if (streamID != eventID)
        {
            events.push_back(SYCLQueue::GetEvent(eventID));
        }
    }
#if defined(__ADAPTIVECPP__)
    sycl::event e =
        SYCLQueue::GetInstance(streamID).submit([&](sycl::handler &cgh) {
            cgh.depends_on(events);
            cgh.AdaptiveCpp_enqueue_custom_operation(
                [=]([[maybe_unused]] sycl::interop_handle ih) {});
        });
    SYCLQueue::SetEvent(streamID, e);
#elif defined(__DPCPP_COMPILER)
    sycl::event e = SYCLQueue::GetInstance(streamID).submit(
        [&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(events); });
    SYCLQueue::SetEvent(streamID, e);
#endif
#endif
}

/// Record the current point of stream @p streamID for SetStreamDependencies.
/// SYCL queues record their last submission themselves.
template <typename ExecSpace>
static void RecordStreamEvent([[maybe_unused]] const unsigned int streamID)
{
    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CUDAStream::RecordEvent(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
        HIPStream::RecordEvent(streamID);
#endif
    }
}

/// Whether stream @p streamID has been created, and so may hold work.
template <typename ExecSpace>
static bool StreamExists([[maybe_unused]] const unsigned int streamID)
{
    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        return CUDAStream::GetAllInstances().count(streamID) > 0;
#elif defined(NEKTAR_ENABLE_HIP)
        return HIPStream::GetAllInstances().count(streamID) > 0;
#elif defined(NEKTAR_ENABLE_SYCL)
        return SYCLQueue::GetAllInstances().count(streamID) > 0;
#endif
    }
    return false;
}

/**
 * @brief Order the work queued next on stream @p streamID after the work
 * already queued on each of @p streamIDs.
 *
 * A stream never created holds no work and is skipped, as is, on CUDA and
 * HIP, a join onto the default stream, which already orders against the
 * numbered streams, created blocking. SYCL on the CPU maps every id to one
 * in-order queue, and the host runs in submission order: neither has
 * anything to order.
 */
template <typename ExecSpace>
static void JoinStreams(
    [[maybe_unused]] const unsigned int streamID,
    [[maybe_unused]] const std::vector<unsigned int> &streamIDs)
{
    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)
        if (streamID == 0)
        {
            return;
        }

        for (const auto id : streamIDs)
        {
            if (id != streamID && StreamExists<ExecSpace>(id))
            {
                RecordStreamEvent<ExecSpace>(id);
                SetStreamDependencies<ExecSpace>(streamID, id);
            }
        }
#elif defined(NEKTAR_ENABLE_SYCL) && !defined(SYCL_ENABLE_CPU)
        // One barrier on the last submission of every queue holding work.
        std::vector<unsigned int> waitOn;
        for (const auto id : streamIDs)
        {
            if (id != streamID && StreamExists<ExecSpace>(id))
            {
                waitOn.push_back(id);
            }
        }

        if (!waitOn.empty())
        {
            SetStreamDependencies<ExecSpace>(streamID, waitOn);
        }
#endif
    }
}

/// Order the work queued next on stream @p streamID after the work already
/// queued on stream @p producerID; JoinStreams() with one stream.
template <typename ExecSpace>
static void JoinStreams(const unsigned int streamID,
                        const unsigned int producerID)
{
    JoinStreams<ExecSpace>(streamID, std::vector<unsigned int>{producerID});
}

/**
 * @brief Order the work queued next on each of @p streamIDs after the work
 * already queued on stream @p streamID.
 *
 * Skipped under the same conditions as JoinStreams(): nothing is ordered
 * after a stream never created, or, on CUDA and HIP, after the default
 * stream.
 */
template <typename ExecSpace>
static void ReleaseStreams(
    [[maybe_unused]] const unsigned int streamID,
    [[maybe_unused]] const std::vector<unsigned int> &streamIDs)
{
    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)
        if (streamID == 0 || !StreamExists<ExecSpace>(streamID))
        {
            return;
        }

        bool recorded = false;
        for (const auto id : streamIDs)
        {
            if (id == streamID)
            {
                continue;
            }

            if (!recorded)
            {
                RecordStreamEvent<ExecSpace>(streamID);
                recorded = true;
            }

            SetStreamDependencies<ExecSpace>(id, streamID);
        }
#elif defined(NEKTAR_ENABLE_SYCL) && !defined(SYCL_ENABLE_CPU)
        if (!StreamExists<ExecSpace>(streamID))
        {
            return;
        }

        for (const auto id : streamIDs)
        {
            if (id != streamID)
            {
                SetStreamDependencies<ExecSpace>(id, streamID);
            }
        }
#endif
    }
}

} // namespace Nektar
