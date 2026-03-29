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
#include <cstring>

namespace Nektar
{

#if defined(NEKTAR_ENABLE_CUDA)
extern bool isSetDeviceMemoryPool;
#elif defined(NEKTAR_ENABLE_HIP)
extern bool isSetDeviceMemoryPool;
#endif

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
    eHostPageable,
    eHostPinned,
    eDeviceMemoryPool,
    eHostPinnedDeviceMemoryPool
};

static inline void SetDeviceMemoryPool(void)
{
#if defined(NEKTAR_ENABLE_CUDA)
    if (!isSetDeviceMemoryPool)
    {
        auto device = nekGetDevice();
        cudaMemPool_t mempool;
        CHECK_HIPCUDA_ERROR(cudaDeviceGetDefaultMemPool(&mempool, device));
        uint64_t threshold = UINT64_MAX;
        CHECK_HIPCUDA_ERROR(cudaMemPoolSetAttribute(
            mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }
#elif defined(NEKTAR_ENABLE_HIP)
    if (!isSetDeviceMemoryPool)
    {
        auto device = nekGetDevice();
        hipMemPool_t mempool;
        CHECK_HIPCUDA_ERROR(hipDeviceGetDefaultMemPool(&mempool, device));
        uint64_t threshold = UINT64_MAX;
        CHECK_HIPCUDA_ERROR(hipMemPoolSetAttribute(
            mempool, hipMemPoolAttrReleaseThreshold, &threshold));
    }
#endif
}

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
inline void hostMallocPinned(TData **src, const size_t size)
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
        *src = (TData *)malloc(size);
#endif
    }
    else
    {
        src = nullptr;
    }
}

template <typename TData>
inline void deviceMalloc(
    TData **src, const size_t size,
    [[maybe_unused]] const MemAllocType memAllocType = eHostPageable)
{
    if (size > 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        // Note: Do not use DeviceMemoryPool for MPI communication buffer.
        if (memAllocType == eDeviceMemoryPool ||
            memAllocType == eHostPinnedDeviceMemoryPool)
        {
            SetDeviceMemoryPool();
            GetDeviceProperties::CheckGlobalMemoryUsage(size);
            CHECK_HIPCUDA_ERROR(cudaMallocAsync((void **)src, size, 0));
            GetDeviceProperties::TotalGlobalMemory() -= size;
        }
        else
        {
            GetDeviceProperties::CheckGlobalMemoryUsage(size);
            CHECK_HIPCUDA_ERROR(cudaMalloc((void **)src, size));
            GetDeviceProperties::TotalGlobalMemory() -= size;
        }
#elif defined(NEKTAR_ENABLE_HIP)
        // Note: Do not use DeviceMemoryPool for MPI communication buffer.
        if (memAllocType == eDeviceMemoryPool ||
            memAllocType == eHostPinnedDeviceMemoryPool)
        {
            SetDeviceMemoryPool();
            GetDeviceProperties::CheckGlobalMemoryUsage(size);
            CHECK_HIPCUDA_ERROR(hipMallocAsync((void **)src, size, 0));
            GetDeviceProperties::TotalGlobalMemory() -= size;
        }
        else
        {
            GetDeviceProperties::CheckGlobalMemoryUsage(size);
            CHECK_HIPCUDA_ERROR(hipMalloc((void **)src, size));
            GetDeviceProperties::TotalGlobalMemory() -= size;
        }
#elif defined(NEKTAR_ENABLE_SYCL)
        GetDeviceProperties::CheckGlobalMemoryUsage(size);
        sycl::queue &Q = SYCLQueue::GetInstance();
        *src           = (TData *)sycl::malloc_device(size, Q);
        GetDeviceProperties::TotalGlobalMemory() -= size;
#else
        *src = (TData *)malloc(size);
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

template <typename TData> inline void hostFreePinned(TData *src)
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
    free(src);
#endif
}

template <typename TData>
inline void deviceFree(
    TData *src, [[maybe_unused]] const size_t size,
    [[maybe_unused]] const MemAllocType memAllocType = eHostPageable)
{
    if (src == nullptr)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    if (memAllocType == eDeviceMemoryPool ||
        memAllocType == eHostPinnedDeviceMemoryPool)
    {
        CHECK_HIPCUDA_ERROR(cudaFreeAsync(src, 0));
        GetDeviceProperties::TotalGlobalMemory() += size;
    }
    else
    {
        CHECK_HIPCUDA_ERROR(cudaFree(src));
        GetDeviceProperties::TotalGlobalMemory() += size;
    }
#elif defined(NEKTAR_ENABLE_HIP)
    if (memAllocType == eDeviceMemoryPool ||
        memAllocType == eHostPinnedDeviceMemoryPool)
    {
        CHECK_HIPCUDA_ERROR(hipFreeAsync(src, 0));
        GetDeviceProperties::TotalGlobalMemory() += size;
    }
    else
    {
        CHECK_HIPCUDA_ERROR(hipFree(src));
        GetDeviceProperties::TotalGlobalMemory() += size;
    }
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.wait();
    sycl::free(src, Q);
    GetDeviceProperties::TotalGlobalMemory() += size;
#else
    free(src);
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
    CHECK_HIPCUDA_ERROR(cudaMemsetAsync((void *)dst, val, size, 0));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipMemsetAsync((void *)dst, val, size, 0));
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
inline void deviceMemcpy(TData *dst, const TData *src, const size_t size)
{
    if (size == 0)
    {
        return;
    }

    if constexpr (std::is_same_v<MemCopy, HostToHost>)
    {
        memcpy(dst, src, size);
    }
    else if constexpr (std::is_same_v<MemCopy, DeviceToHost>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, 0));
        CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(0));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, 0));
        CHECK_HIPCUDA_ERROR(hipStreamSynchronize(0));
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
        CHECK_HIPCUDA_ERROR(
            cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 0));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, 0));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size);
#if defined(SYCL_ENABLE_CPU)
        Q.wait();
#endif
#else
        memcpy(dst, src, size);
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, DeviceToDevice>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, 0));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(
            hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, 0));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size);
#else
        memcpy(dst, src, size);
#endif
    }
}

} // namespace Nektar
