///////////////////////////////////////////////////////////////////////////////
//
// File: DeviceMemory.hpp
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

#ifdef NEKTAR_ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/fill.h>
#elif defined(NEKTAR_ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(NEKTAR_ENABLE_SYCL)
#include "Operators/Utils/SYCLQueue.hpp"
#elif defined(NEKTAR_ENABLE_KOKKOS)
#endif

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
void deviceMalloc(TData *&src, const unsigned int size)
{
    if (size > 0)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        cudaMalloc((void **)&src, size * sizeof(TData));
#elif defined(NEKTAR_ENABLE_HIP)
        hipMalloc((void **)&src, size * sizeof(TData));
#elif defined(NEKTAR_ENABLE_SYCL)
        src = sycl::malloc_device<TData>(size, SYCLQueue::GetInstance());
#elif defined(NEKTAR_ENABLE_KOKKOS)
        src = (TData *)
            Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(
                size * sizeof(TData));
#else
        src = (TData *)malloc(size * sizeof(TData));
#endif
    }
    else
    {
        src = nullptr;
    }
}

template <typename TData> void deviceFree(TData *src)
{
#if defined(NEKTAR_ENABLE_CUDA)
    cudaFree(src);
#elif defined(NEKTAR_ENABLE_HIP)
    hipFree(src);
#elif defined(NEKTAR_ENABLE_SYCL)
    sycl::free(src, SYCLQueue::GetInstance());
#elif defined(NEKTAR_ENABLE_KOKKOS)
    Kokkos::kokkos_free(src);
#else
    free(src);
#endif
}

template <typename TData>
void deviceMemset(TData *dst, const int val, const unsigned int size)
{
#if defined(NEKTAR_ENABLE_CUDA)
    cudaMemset(dst, val, size * sizeof(TData));
#elif defined(NEKTAR_ENABLE_HIP)
    hipMemset(dst, val, size * sizeof(TData));
#elif defined(NEKTAR_ENABLE_SYCL)
    SYCLQueue::GetInstance().memset(dst, val, size * sizeof(TData)).wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
    // Create an unmanage Kokkos view from the raw pointer.
    Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> dstView(dst, size);
    // Deep copy the val to the device view.
    Kokkos::deep_copy(dstView, val);
#else
    memset(dst, val, size * sizeof(TData));
#endif
}

template <typename TData>
void deviceFill(TData *dst, const TData val, const unsigned int size)
{
#if defined(NEKTAR_ENABLE_CUDA)
    thrust::fill(dst, dst + size, val);
#elif defined(NEKTAR_ENABLE_HIP)
    hipLaunchKernelGGL(fill_, blocks, threads, 0, 0, size, dst,
                       val); // TODO: implement fill_ kernel
#elif defined(NEKTAR_ENABLE_SYCL)
    SYCLQueue::GetInstance().fill(dst, val, size).wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
    // Create an unmanage Kokkos view from the raw pointer.
    Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> dstView(dst, size);
    // Deep copy the val to the device view.
    Kokkos::deep_copy(dstView, val);
#else
    std::fill(dst, dst + size, val);
#endif
}

template <typename MemCopy, typename TData>
void deviceMemcpy(TData *dst, const TData *src, const unsigned int size)
{
    if constexpr (std::is_same_v<MemCopy, HostToHost>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        cudaMemcpy(dst, src, size * sizeof(TData), cudaMemcpyHostToHost);
#elif defined(NEKTAR_ENABLE_HIP)
        hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyHostToHost);
#elif defined(NEKTAR_ENABLE_SYCL)
        SYCLQueue::GetInstance().memcpy(dst, src, size * sizeof(TData)).wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
        // Create unmanage Kokkos views from the raw pointers.
        TData *v_src = const_cast<TData *>(src);
        Kokkos::View<TData *, Kokkos::HostSpace> srcView(v_src, size);
        Kokkos::View<TData *, Kokkos::HostSpace> dstView(dst, size);
        // Deep copy the host view to the device view.
        Kokkos::deep_copy(dstView, srcView);
#else
        memcpy(dst, src, size * sizeof(TData));
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, DeviceToHost>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        cudaMemcpy(dst, src, size * sizeof(TData), cudaMemcpyDeviceToHost);
#elif defined(NEKTAR_ENABLE_HIP)
        hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyDeviceToHost);
#elif defined(NEKTAR_ENABLE_SYCL)
        SYCLQueue::GetInstance().memcpy(dst, src, size * sizeof(TData)).wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
        // Create unmanage Kokkos views from the raw pointers.
        TData *v_src = const_cast<TData *>(src);
        Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> srcView(v_src,
                                                                     size);
        Kokkos::View<TData *, Kokkos::HostSpace> dstView(dst, size);
        // Deep copy the host view to the device view.
        Kokkos::deep_copy(dstView, srcView);
#else
        memcpy(dst, src, size * sizeof(TData));
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, HostToDevice>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        cudaMemcpy(dst, src, size * sizeof(TData), cudaMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
        hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyHostToDevice);
#elif defined(NEKTAR_ENABLE_SYCL)
        SYCLQueue::GetInstance().memcpy(dst, src, size * sizeof(TData)).wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
        // Create unmanage Kokkos views from the raw pointers.
        TData *v_src = const_cast<TData *>(src);
        Kokkos::View<TData *, Kokkos::HostSpace> srcView(v_src, size);
        Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> dstView(dst, size);
        // Deep copy the host view to the device view.
        Kokkos::deep_copy(dstView, srcView);
#else
        memcpy(dst, src, size * sizeof(TData));
#endif
    }
    else if constexpr (std::is_same_v<MemCopy, DeviceToDevice>)
    {
#if defined(NEKTAR_ENABLE_CUDA)
        cudaMemcpy(dst, src, size * sizeof(TData), cudaMemcpyDeviceToDevice);
#elif defined(NEKTAR_ENABLE_HIP)
        hipMemcpy(dst, src, size * sizeof(TData), hipMemcpyDeviceToDevice);
#elif defined(NEKTAR_ENABLE_SYCL)
        SYCLQueue::GetInstance().memcpy(dst, src, size * sizeof(TData)).wait();
#elif defined(NEKTAR_ENABLE_KOKKOS)
        // Create unmanage Kokkos views from the raw pointers.
        TData *v_src = const_cast<TData *>(src);
        Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> srcView(v_src,
                                                                     size);
        Kokkos::View<TData *, Kokkos::DefaultExecutionSpace> dstView(dst, size);
        // Deep copy the host view to the device view.
        Kokkos::deep_copy(dstView, srcView);
#else
        memcpy(dst, src, size * sizeof(TData));
#endif
    }
}

} // namespace Nektar
