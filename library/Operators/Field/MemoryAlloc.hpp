///////////////////////////////////////////////////////////////////////////////
//
// File: MemoryAlloc.hpp
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

#include "Operators/Common/DeviceProperties.hpp"

namespace Nektar
{

// MemoryCopy
struct HostToHost
{
};
struct DeviceToHost
{
};
struct HostToDevice
{
};
struct DeviceToDevice
{
};

// Memory allocation type
enum MemAllocType
{
    ePageable,
    ePinned
};

template <typename TData>
void deviceFillKernelLauncher(TData *dst, const TData val, const size_t size);

inline static unsigned int nekGetDeviceCount(void)
{
    //  Current design assumes one MPI rank per device architecture. The number
    //  of GPU devices is limited by the number of MPI ranks.
    int num_rank = 1;
#if defined(NEKTAR_USE_MPI)
    MPI_Comm_size(MPI_COMM_WORLD, &num_rank);
#endif

    int num_device = 1;
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaGetDeviceCount(&num_device));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipGetDeviceCount(&num_device));
#elif defined(SYCL_ENABLE_CUDA) || defined(SYCL_ENABLE_HIP) ||                 \
    defined(SYCL_ENABLE_INTEL)
    num_device = sycl::device::get_devices(sycl::info::device_type::gpu).size();
#endif
    return std::min(num_rank, num_device);
}

inline static void nekSetDevice([[maybe_unused]] unsigned int device_rank)
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

inline static unsigned int nekGetDevice()
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

template <typename TData>
inline void hostMalloc(TData **src, const size_t size,
                       [[maybe_unused]] const size_t alignment)
{
    if (size > 0)
    {
        *src = static_cast<TData *>(
            ::operator new[](size, std::align_val_t(alignment)));
    }
    else
    {
        *src = nullptr;
    }
}

template <typename TData>
inline void hostMallocPinned(TData **src, const size_t size,
                             [[maybe_unused]] const size_t alignment)
{
    if (size > 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaMallocHost((void **)src, size));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipHostMalloc((void **)src, size));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        *src           = (TData *)sycl::malloc_host(size, Q);
#else
        *src = static_cast<TData *>(
            ::operator new[](size, std::align_val_t(alignment)));
#endif
    }
    else
    {
        src = nullptr;
    }
}

template <typename TData>
inline void deviceMalloc(TData **src, const size_t size,
                         [[maybe_unused]] const size_t alignment)
{
    if (size > 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        GetDeviceProperties::CheckGlobalMemoryUsage(size);
        CHECK_HIPCUDA_ERROR(cudaMalloc((void **)src, size));
        GetDeviceProperties::TotalGlobalMemory() -= size;
#elif defined(NEKTAR_ENABLE_HIP)
        GetDeviceProperties::CheckGlobalMemoryUsage(size);
        CHECK_HIPCUDA_ERROR(hipMalloc((void **)src, size));
        GetDeviceProperties::TotalGlobalMemory() -= size;
#elif defined(NEKTAR_ENABLE_SYCL)
        GetDeviceProperties::CheckGlobalMemoryUsage(size);
        sycl::queue &Q = SYCLQueue::GetInstance();
        *src           = (TData *)sycl::malloc_device(size, Q);
        GetDeviceProperties::TotalGlobalMemory() -= size;
#else
        *src = static_cast<TData *>(
            ::operator new[](size, std::align_val_t(alignment)));
#endif
    }
    else
    {
        *src = nullptr;
    }
}

template <typename TData>
inline void hostFree(TData *src, [[maybe_unused]] const size_t alignment)
{
    if (src == nullptr)
    {
        return;
    }

    operator delete[](src, std::align_val_t(alignment));
}

template <typename TData>
inline void hostFreePinned(TData *src, [[maybe_unused]] const size_t alignment)
{
    if (src == nullptr)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaFreeHost(src));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipFreeHost(src));
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.wait();
    sycl::free(src, Q);
#else
    operator delete[](src, std::align_val_t(alignment));
#endif
}

template <typename TData>
inline void deviceFree(TData *src, [[maybe_unused]] const size_t size,
                       [[maybe_unused]] const size_t alignment)
{
    if (src == nullptr)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaFree(src));
    GetDeviceProperties::TotalGlobalMemory() += size;
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipFree(src));
    GetDeviceProperties::TotalGlobalMemory() += size;
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.wait();
    sycl::free(src, Q);
    GetDeviceProperties::TotalGlobalMemory() += size;
#else
    operator delete[](src, std::align_val_t(alignment));
#endif
}

template <typename TData>
inline void deviceMemset(TData *dst, const int val, const size_t size)
{
    if (size == 0)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaMemsetAsync((void *)dst, val, size));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipMemsetAsync((void *)dst, val, size));
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.memset((void *)dst, val, size);
#else
    memset((void *)dst, val, size);
#endif
}

template <typename TData>
inline void deviceFill(TData *dst, const TData val, const size_t size)
{
    if (size == 0)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    deviceFillKernelLauncher(dst, val, size);
#elif defined(NEKTAR_ENABLE_HIP)
    deviceFillKernelLauncher(dst, val, size);
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.fill(dst, val, size);
#else
    std::fill(dst, dst + size, val);
#endif
}

template <typename MemCopy, typename TData>
inline void deviceMemcpy(
    TData *dst, const TData *src, const size_t size,
    [[maybe_unused]] const MemAllocType memAllocType = ePageable)
{
    if (size == 0)
    {
        return;
    }

    if constexpr (std::is_same_v<MemCopy, HostToHost>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyHostToHost));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipMemcpy(dst, src, size, hipMemcpyHostToHost));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size).wait();
#else
        memcpy(dst, src, size);
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, DeviceToHost>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size).wait();
#else
        memcpy(dst, src, size);
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, HostToDevice>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        if (memAllocType == ePinned)
        {
            CHECK_HIPCUDA_ERROR(
                cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice));
        }
        else
        {
            CHECK_HIPCUDA_ERROR(
                cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
        }
#elif defined(NEKTAR_ENABLE_HIP)
        if (memAllocType == ePinned)
        {
            CHECK_HIPCUDA_ERROR(
                hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice));
        }
        else
        {
            CHECK_HIPCUDA_ERROR(
                hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
        }
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        if (memAllocType == ePinned)
        {
            Q.memcpy(dst, src, size);
        }
        else
        {
            Q.memcpy(dst, src, size).wait();
        }
#else
        memcpy(dst, src, size);
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, DeviceToDevice>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size);
#else
        memcpy(dst, src, size);
#endif
    }
}

} // namespace Nektar
