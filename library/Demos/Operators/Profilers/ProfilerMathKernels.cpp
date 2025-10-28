///////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerMathKernels.cpp
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

#include <cstdio>
#include <iomanip>
#include <iostream>

#include "Operators/MathKernels/Math.hpp"
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Timer.h>
#include <Operators/Field/Field.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;
using vec_t = tinysimd::simd<double>;

template <typename TData, bool warmup = false>
void ProfilerReduction(const size_t size)
{
    // Initialization.
    Math math;
    Timer timer;
    const unsigned int ntests = 40;
    auto x    = MemoryRegion<TData>("x", size, vec_t::alignment);
    auto y    = MemoryRegion<TData>("y", size, vec_t::alignment);
    auto xptr = x.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    auto yptr = y.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    for (size_t i = 0; i < size; i++)
    {
        xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191)));
        yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516)));
    }
    x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();

    // Serial.
    math = Math("Serial");
    TData result_serial;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        result_serial = math.ddot(x, y);
        ASSERTL0((result_serial > 0.0), "Error!");
    }
    timer.Stop();
    TData time_serial = timer.Elapsed().count() / ntests;

    // AVX.
#if defined(NEKTAR_ENABLE_SIMD)
    math = Math("AVX");
    TData result_avx;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        result_avx = math.ddot(x, y);
        ASSERTL0((result_avx > 0.0), "Error!");
    }
    timer.Stop();
    TData time_avx = timer.Elapsed().count() / ntests;
    // Device.
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    math = Math("Device");
    TData result_device;
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        result_device = math.ddot(x, y);
        ASSERTL0((result_device > 0.0), "Error!");
    }
    timer.Stop();
    TData time_device = timer.Elapsed().count() / ntests;
#endif

    // Display results.
    if (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 2 * sizeof(TData) * 1e-9 * size / time_serial
                  << " "
#if defined(NEKTAR_ENABLE_SIMD)
                  << 2 * sizeof(TData) * 1e-9 * size / time_avx << " "
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
                  << 2 * sizeof(TData) * 1e-9 * size / time_device
#endif
                  << std::endl;
    }
}

template <typename TData, bool warmup = false>
void ProfilerDaxpy(const size_t size)
{
    // Initialization.
    Math math;
    Timer timer;
    const unsigned int ntests = 40;
    auto x    = MemoryRegion<TData>("x", size, vec_t::alignment);
    auto y    = MemoryRegion<TData>("y", size, vec_t::alignment);
    auto z    = MemoryRegion<TData>("z", size, vec_t::alignment);
    auto xptr = x.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    auto yptr = y.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    for (size_t i = 0; i < size; i++)
    {
        xptr[i] = i % 13 + (0.2 + 0.00001 * (i % (100191)));
        yptr[i] = i % 42 + (0.1 + 0.00008 * (i % (280516)));
    }
    x.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    y.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>();
    z.template GetPtr<NektarSpaces::DeviceSpace, WriteOnly>();

    // Serial.
    math = Math("Serial");
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        math.daxpy(3.2, x, y, z);
    }
    timer.Stop();
    TData time_serial = timer.Elapsed().count() / ntests;

    // AVX.
#if defined(NEKTAR_ENABLE_SIMD)
    math = Math("AVX");
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        math.daxpy(3.2, x, y, z);
    }
    timer.Stop();
    TData time_avx = timer.Elapsed().count() / ntests;
    // SYCL.
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    math = Math("Device");
    timer.Start();
    for (unsigned int t = 0; t < ntests; ++t)
    {
        math.daxpy(3.2, x, y, z);
    }
    nekDeviceSynchronize();
    timer.Stop();
    TData time_device = timer.Elapsed().count() / ntests;
#endif

    // Display results.
    if (!warmup)
    {
        std::cout << std::setprecision(10);
        std::cout << "Size " << size
                  << " GB/s: " << 3 * sizeof(TData) * 1e-9 * size / time_serial
                  << " "
#if defined(NEKTAR_ENABLE_SIMD)
                  << 3 * sizeof(TData) * 1e-9 * size / time_avx << " "
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
                  << 3 * sizeof(TData) * 1e-9 * size / time_device
#endif
                  << std::endl;
    }
}

int main(void)
{
    // Print GPU properties
#if defined(NEKTAR_ENABLE_CUDA)
    cudaDeviceProp prop;
    CHECK_HIPCUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n", prop.name);
#if CUDART_VERSION >= 13000
    int memoryClockRate;
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
    printf("  Memory Clock Rate (KHz): %d\n", memoryClockRate);
#else
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
#endif
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Total Global Memory (bytes): %ld\n", prop.totalGlobalMem);
    printf("  Shared Memory per Block (bytes): %ld\n", prop.sharedMemPerBlock);
    printf("  Shared Memory per Multiprocessor (bytes): %ld\n",
           prop.sharedMemPerMultiprocessor);
#if CUDART_VERSION >= 13000
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
#else
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
#endif
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
#elif defined(NEKTAR_ENABLE_HIP)
    hipDeviceProp_t prop;
    CHECK_HIPCUDA_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Total Global Memory (bytes): %ld\n", prop.totalGlobalMem);
    printf("  Shared Memory per Block (bytes): %ld\n", prop.sharedMemPerBlock);
    printf("  Shared Memory per Multiprocessor (bytes): %ld\n",
           prop.sharedMemPerMultiprocessor);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
#elif defined(NEKTAR_ENABLE_SYCL) && !defined(SYCL_ENABLE_CPU) &&              \
    defined(__INTEL_LLVM_COMPILER)
    auto device =
        SYCLQueue::GetInstance().get_info<sycl::info::queue::device>();
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n",
           device.get_info<sycl::info::device::name>().c_str());
    printf(
        "  Memory Clock Rate (KHz): %d\n",
        1000 *
            device
                .get_info<sycl::ext::intel::info::device::memory_clock_rate>());
    printf("  Memory Bus Width (bits): %d\n",
           device.get_info<sycl::ext::intel::info::device::memory_bus_width>());
    printf("  Total Global Memory (bytes): %ld\n",
           device.get_info<sycl::info::device::global_mem_size>());
    printf("  Shared Memory per Block (bytes): %ld\n",
           device.get_info<sycl::info::device::local_mem_size>());
    printf(
        "  Peak Memory Bandwidth (GB/s): %f\n",
        2.0 *
            device
                .get_info<sycl::ext::intel::info::device::memory_clock_rate>() *
            (device
                 .get_info<sycl::ext::intel::info::device::memory_bus_width>() /
             8) /
            1.0e3);
    printf("  Number of multiprocessors: %d\n",
           device.get_info<sycl::info::device::max_compute_units>());
#elif defined(NEKTAR_ENABLE_SYCL)
    auto device =
        SYCLQueue::GetInstance().get_info<sycl::info::queue::device>();
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n",
           device.get_info<sycl::info::device::name>().c_str());
    printf(
        "  Total Global Memory (bytes): %ld\n",
        (long unsigned)device.get_info<sycl::info::device::global_mem_size>());
    printf(
        "  Shared Memory per Block (bytes): %ld\n",
        (long unsigned)device.get_info<sycl::info::device::local_mem_size>());
    printf("  Number of multiprocessors: %d\n",
           device.get_info<sycl::info::device::max_compute_units>());
#endif

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Math Kernel Profiler : Reduction " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "                      Serial";
#if defined(NEKTAR_ENABLE_SIMD)
    std::cout << "      AVX";
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    std::cout << "      Device";
#endif
    std::cout << std::endl;

    // Warm-up.
    ProfilerReduction<double, true>(2 << 24);

    // Benchmark.
    for (unsigned int i = 0; i < 24; i++)
    {
        ProfilerReduction<double>(4 << i);
    }

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Math Kernel Profiler : daxpy     " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "                      Serial";
#if defined(NEKTAR_ENABLE_SIMD)
    std::cout << "      AVX";
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||             \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
    std::cout << "      Device";
#endif
    std::cout << std::endl;

    // Warm-up.
    ProfilerDaxpy<double, true>(2 << 24);

    // Benchmark.
    for (unsigned int i = 0; i < 24; i++)
    {
        ProfilerDaxpy<double>(2 << i);
    }
}
