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

#include "Operators/Common/Spaces.hpp"

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
void hostMallocPinned(TData *&src, const unsigned int size,
                      [[maybe_unused]] const unsigned int alignment)
{
    if (size > 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(
            cudaMallocHost((void **)&src, size * sizeof(TData)));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipHostMalloc((void **)&src, size * sizeof(TData)));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        src            = sycl::malloc_host<TData>(size, Q);
#else
        src = static_cast<TData *>(::operator new[](
            size * sizeof(TData), std::align_val_t(alignment)));
#endif
    }
    else
    {
        src = nullptr;
    }
}

template <typename TData>
void deviceMalloc(TData *&src, const unsigned int size,
                  [[maybe_unused]] const unsigned int device_rank)
{
    if (size > 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(cudaMalloc((void **)&src, size * sizeof(TData)));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(hipMalloc((void **)&src, size * sizeof(TData)));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        src            = sycl::malloc_device<TData>(size, Q);
#else
        src = (TData *)malloc(size * sizeof(TData));
#endif
    }
    else
    {
        src = nullptr;
    }
}

template <typename TData>
void hostFreePinned(TData *&src, [[maybe_unused]] const unsigned int alignment)
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
void deviceFree(TData *&src, [[maybe_unused]] const unsigned int device_rank)
{
    if (src == nullptr)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
    CHECK_HIPCUDA_ERROR(cudaFree(src));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
    CHECK_HIPCUDA_ERROR(hipFree(src));
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    sycl::free(src, Q);
#else
    free(src);
#endif

    src = nullptr;
}

template <typename TData>
void deviceMemset(TData *dst, const int val, const unsigned int size,
                  [[maybe_unused]] const unsigned int device_rank)
{
    if (size == 0)
    {
        return;
    }

#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
    CHECK_HIPCUDA_ERROR(cudaMemset((void *)dst, val, size * sizeof(TData)));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
    CHECK_HIPCUDA_ERROR(hipMemset((void *)dst, val, size * sizeof(TData)));
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.memset((void *)dst, val, size * sizeof(TData)).wait();
#else
    memset((void *)dst, val, size * sizeof(TData));
#endif
}

template <typename TData>
void deviceFill(TData *dst, const TData val, const unsigned int size,
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
void deviceMemcpy(TData *dst, const TData *src, const unsigned int size,
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
        CHECK_HIPCUDA_ERROR(
            cudaMemcpy(dst, src, size * sizeof(TData), cudaMemcpyHostToHost));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(
            hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyHostToHost));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size * sizeof(TData)).wait();
#else
        memcpy(dst, src, size * sizeof(TData));
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, DeviceToHost>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(
            cudaMemcpy(dst, src, size * sizeof(TData), cudaMemcpyDeviceToHost));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(
            hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyDeviceToHost));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size * sizeof(TData)).wait();
#else
        memcpy(dst, src, size * sizeof(TData));
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, HostToDevice>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(
            cudaMemcpy(dst, src, size * sizeof(TData), cudaMemcpyHostToDevice));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(
            hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyHostToDevice));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size * sizeof(TData)).wait();
#else
        memcpy(dst, src, size * sizeof(TData));
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, DeviceToDevice>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_HIPCUDA_ERROR(cudaSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(cudaMemcpy(dst, src, size * sizeof(TData),
                                       cudaMemcpyDeviceToDevice));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_HIPCUDA_ERROR(hipSetDevice(device_rank));
        CHECK_HIPCUDA_ERROR(
            hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyDeviceToDevice));
#elif defined(NEKTAR_ENABLE_SYCL)
        sycl::queue &Q = SYCLQueue::GetInstance();
        Q.memcpy(dst, src, size * sizeof(TData)).wait();
#else
        memcpy(dst, src, size * sizeof(TData));
#endif
    }
}

} // namespace Nektar
