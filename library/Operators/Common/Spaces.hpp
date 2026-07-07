///////////////////////////////////////////////////////////////////////////////
//
// File: Spaces.hpp
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
#include "Operators/Common/Backends/CUDA_Host_API.hpp"
#elif defined(NEKTAR_ENABLE_HIP)
#include "Operators/Common/Backends/HIP_Host_API.hpp"
#elif defined(NEKTAR_ENABLE_SYCL)
#include "Operators/Common/Backends/SYCL_Host_API.hpp"
#elif defined(NEKTAR_ENABLE_DEVICEONHOST)
#include "Operators/Common/Backends/DeviceOnHost_Host_API.hpp"
#endif

#if defined(__CUDACC__) || defined(__NEK_HIPCC__) ||                           \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
#define DEVICE_COMPILE_ONLY
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
    static constexpr unsigned int maximumBlockSize = 1024u;
    static constexpr unsigned int warpSize         = 1u;
#else
    static constexpr unsigned int defaultBlockSize = 1u;
    static constexpr unsigned int maximumBlockSize = 1u;
    static constexpr unsigned int warpSize         = 1u;
#endif
};

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

// Memory scope for atomic.
struct GlobalScope
{
};

struct LocalScope
{
};

} // namespace NektarSpaces

namespace Nektar
{

#if defined(NEKTAR_ENABLE_DEVICE)
extern std::unordered_map<unsigned int, void *> internalMemoryBufferMap;
extern void *internalDeviceBuffer;
extern void *internalHostBuffer;
extern unsigned int internalMaxDataSizeByte;
#endif

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
    int num_device = 1;
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaGetDeviceCount(&num_device));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipGetDeviceCount(&num_device));
#elif defined(SYCL_ENABLE_CUDA) || defined(SYCL_ENABLE_HIP) ||                 \
    defined(SYCL_ENABLE_INTEL)
    num_device = sycl::device::get_devices(sycl::info::device_type::gpu).size();
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

} // namespace Nektar

#include "Operators/Common/Backends/DeviceOnHost_Device_API.hpp"
#include "Operators/Common/Backends/HIPCUDA_Device_API.hpp"
#include "Operators/Common/Backends/SYCL_Device_API.hpp"
