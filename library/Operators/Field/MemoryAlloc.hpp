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

template <typename TData>
void hostMalloc(TData *&src, const size_t size,
                [[maybe_unused]] const size_t alignment)
{
    if (size > 0)
    {
        src = static_cast<TData *>(
            ::operator new[](size, std::align_val_t(alignment)));
    }
    else
    {
        src = nullptr;
    }
}

template <typename TData>
void hostMallocPinned(TData *&src, const size_t size,
                      [[maybe_unused]] const size_t alignment)
{
    if (size > 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaMallocHost((void **)&src, size));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipHostMalloc((void **)&src, size));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        src            = (TData *)sycl::malloc_host(size, Q);
#else
        src = static_cast<TData *>(
            ::operator new[](size, std::align_val_t(alignment)));
#endif
    }
    else
    {
        src = nullptr;
    }
}

template <typename TData>
void deviceMalloc(TData *&src, const size_t size,
                  [[maybe_unused]] const size_t alignment,
                  [[maybe_unused]] const unsigned int device_rank)
{
    if (size > 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
        GetDeviceProperties::CheckGlobalMemoryUsage(size);
        CHECK_HIPCUDA_ERROR(cudaMalloc((void **)&src, size));
        GetDeviceProperties::TotalGlobalMemory() -= size;
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
        GetDeviceProperties::CheckGlobalMemoryUsage(size);
        CHECK_HIPCUDA_ERROR(hipMalloc((void **)&src, size));
        GetDeviceProperties::TotalGlobalMemory() -= size;
#elif defined(NEKTAR_ENABLE_SYCL)
        GetDeviceProperties::CheckGlobalMemoryUsage(size);
        sycl::queue &Q = SYCLQueue::GetInstance();
        src            = (TData *)sycl::malloc_device(size, Q);
        GetDeviceProperties::TotalGlobalMemory() -= size;
#else
        src = static_cast<TData *>(
            ::operator new[](size, std::align_val_t(alignment)));
#endif
    }
    else
    {
        src = nullptr;
    }
}

template <typename TData>
void hostFree(TData *&src, [[maybe_unused]] const size_t alignment)
{
    if (src == nullptr)
    {
        return;
    }

    operator delete[](src, std::align_val_t(alignment));

    src = nullptr;
}

template <typename TData>
void hostFreePinned(TData *&src, [[maybe_unused]] const size_t alignment)
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
    sycl::free(src, Q);
#else
    operator delete[](src, std::align_val_t(alignment));
#endif

    src = nullptr;
}

template <typename TData>
void deviceFree(TData *&src, [[maybe_unused]] const size_t size,
                [[maybe_unused]] const size_t alignment,
                [[maybe_unused]] const unsigned int device_rank)
{
    if (src == nullptr)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
    CHECK_HIPCUDA_ERROR(cudaFree(src));
    GetDeviceProperties::TotalGlobalMemory() += size;
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
    CHECK_HIPCUDA_ERROR(hipFree(src));
    GetDeviceProperties::TotalGlobalMemory() += size;
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    sycl::free(src, Q);
    GetDeviceProperties::TotalGlobalMemory() += size;
#else
    operator delete[](src, std::align_val_t(alignment));
#endif

    src = nullptr;
}

template <typename TData>
void deviceMemset(TData *dst, const int val, const size_t size,
                  [[maybe_unused]] const unsigned int device_rank)
{
    if (size == 0)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
    CHECK_HIPCUDA_ERROR(cudaMemset((void *)dst, val, size));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
    CHECK_HIPCUDA_ERROR(hipMemset((void *)dst, val, size));
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.memset((void *)dst, val, size).wait();
#else
    memset((void *)dst, val, size);
#endif
}

template <typename TData>
void deviceFill(TData *dst, const TData val, const size_t size,
                [[maybe_unused]] const unsigned int device_rank)
{
    if (size == 0)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
    thrust::fill(dst, dst + size, val);
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
    thrust::fill(dst, dst + size, val);
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.fill(dst, val, size).wait();
#else
    std::fill(dst, dst + size, val);
#endif
}

template <typename MemCopy, typename TData>
void deviceMemcpy(TData *dst, const TData *src, const size_t size,
                  [[maybe_unused]] const unsigned int device_rank)
{
    if (size == 0)
    {
        return;
    }

    if constexpr (std::is_same_v<MemCopy, HostToHost>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyHostToHost));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
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
        CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
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
        CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size).wait();
#else
        memcpy(dst, src, size);
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, DeviceToDevice>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size).wait();
#else
        memcpy(dst, src, size);
#endif
    }
}

} // namespace Nektar
