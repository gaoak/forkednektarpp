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
void deviceFillKernelLauncher(TData *dst, const TData val, const size_t size,
                              const unsigned int streamID);

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
        sycl::queue &Q = SYCLQueue::GetInstance(0);
        *src           = (TData *)sycl::malloc_host(size, Q);
        Q.wait();
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
    [[maybe_unused]] const unsigned int streamID,
    [[maybe_unused]] const MemAllocType memAllocType = eHostPageable)
{
    if (size > 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        // Note: Do not use DeviceMemoryPool for MPI communication buffer.
        if (memAllocType == eDeviceMemoryPool ||
            memAllocType == eHostPinnedDeviceMemoryPool)
        {
            auto stream = CUDAStream::GetInstance(streamID);
            SetDeviceMemoryPool();
            GetDeviceProperties::CheckGlobalMemoryUsage(size);
            CHECK_HIPCUDA_ERROR(cudaMallocAsync((void **)src, size, stream));
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
            auto stream = HIPStream::GetInstance(streamID);
            SetDeviceMemoryPool();
            GetDeviceProperties::CheckGlobalMemoryUsage(size);
            CHECK_HIPCUDA_ERROR(hipMallocAsync((void **)src, size, stream));
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
        sycl::queue &Q = SYCLQueue::GetInstance(streamID);
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
    sycl::queue &Q = SYCLQueue::GetInstance(0);
    Q.wait();
    sycl::free(src, Q);
    Q.wait();
#else
    free(src);
#endif
}

template <typename TData>
inline void deviceFree(
    TData *src, [[maybe_unused]] const size_t size,
    [[maybe_unused]] const unsigned int streamID,
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
        auto stream = CUDAStream::GetInstance(streamID);
        CHECK_HIPCUDA_ERROR(cudaFreeAsync(src, stream));
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
        auto stream = HIPStream::GetInstance(streamID);
        CHECK_HIPCUDA_ERROR(hipFreeAsync(src, stream));
        GetDeviceProperties::TotalGlobalMemory() += size;
    }
    else
    {
        CHECK_HIPCUDA_ERROR(hipFree(src));
        GetDeviceProperties::TotalGlobalMemory() += size;
    }
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    Q.wait();
    sycl::free(src, Q);
    Q.wait();
    GetDeviceProperties::TotalGlobalMemory() += size;
#else
    free(src);
#endif
}

template <typename TData>
inline void deviceMemset(TData *dst, const int val, const size_t size,
                         [[maybe_unused]] const unsigned int streamID)
{
    if (size == 0)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    auto stream = CUDAStream::GetInstance(streamID);
    CHECK_HIPCUDA_ERROR(cudaMemsetAsync((void *)dst, val, size, stream));
#elif defined(NEKTAR_ENABLE_HIP)
    auto stream = HIPStream::GetInstance(streamID);
    CHECK_HIPCUDA_ERROR(hipMemsetAsync((void *)dst, val, size, stream));
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    Q.memset((void *)dst, val, size);
#else
    memset((void *)dst, val, size);
#endif
}

template <typename TData>
inline void deviceFill(TData *dst, const TData val, const size_t size,
                       [[maybe_unused]] const unsigned int streamID)
{
    if (size == 0)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    deviceFillKernelLauncher(dst, val, size, streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    deviceFillKernelLauncher(dst, val, size, streamID);
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    Q.fill(dst, val, size);
#else
    std::fill(dst, dst + size, val);
#endif
}

template <typename MemCopy, typename TData>
inline void deviceMemcpy(TData *dst, const TData *src, const size_t size,
                         [[maybe_unused]] const unsigned int streamID)
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
        auto stream = CUDAStream::GetInstance(streamID);
        CHECK_HIPCUDA_ERROR(
            cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
        CHECK_HIPCUDA_ERROR(cudaStreamSynchronize(stream));
#elif defined(NEKTAR_ENABLE_HIP)
        auto stream = HIPStream::GetInstance(streamID);
        CHECK_HIPCUDA_ERROR(
            hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, stream));
        CHECK_HIPCUDA_ERROR(hipStreamSynchronize(stream));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance(streamID);
        Q.memcpy(dst, src, size).wait();
#else
        memcpy(dst, src, size);
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, HostToDevice>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        auto stream = CUDAStream::GetInstance(streamID);
        CHECK_HIPCUDA_ERROR(
            cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
#elif defined(NEKTAR_ENABLE_HIP)
        auto stream = HIPStream::GetInstance(streamID);
        CHECK_HIPCUDA_ERROR(
            hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance(streamID);
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
        auto stream = CUDAStream::GetInstance(streamID);
        CHECK_HIPCUDA_ERROR(
            cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
#elif defined(NEKTAR_ENABLE_HIP)
        auto stream = HIPStream::GetInstance(streamID);
        CHECK_HIPCUDA_ERROR(
            hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance(streamID);
        Q.memcpy(dst, src, size);
#else
        memcpy(dst, src, size);
#endif
    }
}

} // namespace Nektar
